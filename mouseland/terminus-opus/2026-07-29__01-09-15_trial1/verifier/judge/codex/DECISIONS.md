# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `data/beh/Imaging_Exp_info.npy` as the master index, builds one unique `session_id` per mouse/date/block, then for each session searches all behavior files associated with that session's experiment types, loads the matching spike file and retinotopy file, and processes trials session by session. Unlike the reference, it reloads behavior files per session instead of grouping sessions by behavior file, and it only uses the first matching behavior variant.

ii. 
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
session_ids = sorted(all_sessions.keys())
```
```python
def load_beh_for_session(sess_id, exp_types):
    results = []
    for exp_type in exp_types:
        beh_path = f'data/beh/Beh_{exp_type}.npy'
        if os.path.exists(beh_path):
            beh_all = np.load(beh_path, allow_pickle=True).item()
            if sess_id in beh_all:
                results.append((beh_all[sess_id], sess_id))
            for key in sorted(beh_all.keys()):
                if key.startswith(sess_id + '_swap'):
                    results.append((beh_all[key], key))
```
```python
beh_results = load_beh_for_session(sess_id, exp_types)
beh, beh_key = beh_results[0]
spk = load_spk(mname, datexp, blk)
retino = load_retino(mname, datexp)
```

iii. In `CONVERSION_NOTES.md` Step 6 and Step 10, the AI says it loads sessions from `exp_info`, processes each session by loading neural, behavioral, and retinotopy data, and notes that 13 sessions were skipped because it could not match their swap-session behavior keys.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mname`. The AI collects all unique mouse names from processed sessions, sorts them, and stores per-session indices into that list.

ii. 
```python
sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
sessions[sess_id] = {
    'mname': s['mname'],
    'datexp': s['datexp'],
    'blk': s['blk'],
```
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. `CONVERSION_NOTES.md` Step 2 records 19 subjects and Step 9 reports 19 subjects in the converted data, so the AI treated `mname` as the subject identifier throughout.

## 1-c. How are the data split into sessions?

i. A session is one unique `mname_datexp_blk` combination. The AI deduplicates the repeated entries in `Imaging_Exp_info.npy` by that session id and stores all experiment types seen for the same session.

ii. 
```python
sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if sess_id not in sessions:
    sessions[sess_id] = {
        'mname': s['mname'],
        'datexp': s['datexp'],
        'blk': s['blk'],
        'exp_types': [],
```
```python
sessions[sess_id]['exp_types'].append(exp_type)
```

iii. In the trajectory and `CONVERSION_NOTES.md` Step 6, the AI describes building a list of 89 unique sessions from `exp_info` and then processing them one by one.

## 1-d. How are the data split into trials?

i. Trials are defined from the rounded trial start frame `StartFr` up to the rounded gray-space entry frame `GrayFr`. For each trial, the AI slices contiguous frame ranges `[start_fr:end_fr)` and keeps the whole available corridor segment, with variable trial length.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
end_fr = min(start_fr + n_timepoints, gray_fr)
actual_frames = end_fr - start_fr
```
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 10, the AI explicitly says it extracts the corridor portion from `StartFr` to `GrayFr` because that is when the visual stimulus is present.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with fewer than 2 available frames or with frame bounds outside the spike matrix, and it drops sessions with fewer than 2 valid trials.

ii. 
```python
if avail_frames < 2:
    return None
...
if actual_frames < 2:
    return None
...
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```
```python
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. `CONVERSION_NOTES.md` Step 10 says one trial with fewer than 2 frames was excluded and that sessions need at least two valid trials for the decoder format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `spks` in the per-session neural `.npy` file, concatenated across imaging planes. Brain-region labels come from `iarea` in the retinotopy `.npz` file.

ii. 
```python
dat = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```
```python
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return {'iarea': dtrans['iarea'], 'neu_ar_idx': ix}
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 5 describe neural data as deconvolved calcium traces loaded from `spks`, with retinotopy used for area assignment.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, slices each trial from `StartFr` to `GrayFr`, and casts the resulting per-trial matrices to `float16`. It does not pad to a fixed length and does not otherwise transform the traces.

ii. 
```python
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. In `CONVERSION_NOTES.md` Step 6 and Step 10, the AI says it uses the corridor portion of each trial and stores neural data as `float16` to reduce file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons out by area. It assigns neurons to `V1`, `mHV`, `lHV`, `aHV`, or an extra `unassigned` category, and keeps all neurons.

ii. 
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)
for i, region in enumerate(brain_regions[:-1]):
    if region in region_map:
        neuron_region_idx[region_map[region]] = i
```

iii. `CONVERSION_NOTES.md` Step 1 says there is no explicit neuron quality filtering in the reference code, and Step 10 claims "No neuron filtering" was consistent with the reference, so the AI chose to retain unassigned neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to corridor entry by using `StartFr` as time zero and extracting frames until `GrayFr`. The alignment is therefore trial-start aligned, but trial durations remain variable and are not padded/truncated to a common 32-frame window.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
end_fr = min(start_fr + n_timepoints, gray_fr)
```
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the AI says trials are aligned to corridor entry and uses the corridor portion because that is where the task-relevant visual stimulus occurs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one imaging frame at 3.17 Hz, stored as `1000 / 3.17` ms per bin. No temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```
```python
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FRAME_RATE,
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 3 identify the frame rate as 3.17 Hz, and the AI uses that directly as the binning grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives `time_to_sound_cue` from `SoundFr`, `StartFr`, and the fixed frame rate. It does not use `ft` timestamps.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
sound_fr = beh['SoundFr'][trial_idx]
sound_frame_offset = sound_fr - start_fr
```
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps "SoundFr - frame" to `time_to_sound_cue`, and the trajectory repeatedly describes using frame offsets from trial start.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes a per-frame signed offset `(current_frame_index - sound_frame_offset) / FRAME_RATE` within each trial. This uses frame counts rather than interpolated timestamps, and it yields negative values before the cue and positive values after the cue.

ii. 
```python
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. In the code comments the AI states this is the signed time from the current frame to the sound cue, and `CONVERSION_NOTES.md` Step 5 summarizes the transformation as converting sound-frame offsets into time in seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The AI computes `time_to_sound_cue` on the same per-trial frame sequence extracted for the neural data, so it has the same time axis and length as each trial's neural matrix.

ii. 
```python
trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
actual_frames = trial_data['actual_frames']
neural = trial_data['neural']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The trajectory plan in Step 31 says inputs are constructed after extracting per-trial windows aligned to corridor entry, so the AI intended all inputs to share the same window as `neural`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `day_of_training` is derived from `mname` and `datexp` in `Imaging_Exp_info.npy`. The AI gathers all dates seen for one mouse and finds the index of the current session date in sorted order.

ii. 
```python
for exp_type, sessions in exp_info.items():
    for s in sessions:
        if s['mname'] == mname:
            dates.add(s['datexp'])
sorted_dates = sorted(dates)
return sorted_dates.index(datexp)
```

iii. `CONVERSION_NOTES.md` Step 5 says `datexp` is mapped to a chronological day index, and the trajectory describes this as the mouse's training day.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI counts unique recording dates per mouse, starting at 0, and broadcasts that scalar across all time bins of every trial from that session.

ii. 
```python
sorted_dates = sorted(dates)
return sorted_dates.index(datexp)
```
```python
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes `day_of_training` as a chronological day index derived from session date, not from block count or session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives `time_since_trial_start` from `StartFr` and the fixed frame rate. It does not use `ft`.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps "frame - StartFr" to `time_since_trial_start`, and the trajectory describes using trial-start frame offsets.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI sets the first kept frame of each trial to time 0 and increments by `1 / FRAME_RATE` seconds for each later frame. Because extraction starts at `StartFr`, there are no negative pre-start bins.

ii. 
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI summarizes this variable as time in seconds computed from frame offset relative to trial start.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The AI computes `time_since_trial_start` over exactly the same extracted frame window as each trial's neural data, so the input length matches the neural time axis.

ii. 
```python
neural = trial_data['neural']
actual_frames = trial_data['actual_frames']
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The trajectory and code structure show that this input is built after `extract_trial_data`, so it shares the same per-trial alignment as the neural slice.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `beh['isRew']`.

ii. 
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `isRew` to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI casts the trial-level boolean to float and broadcasts it across all time bins in the trial. There is no further processing.

ii. 
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI describes this variable as a binary per-trial flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus label directly from `beh['WallName']`.

ii. 
```python
wall_names = beh['WallName']
stim_name = str(wall_names[t])
```

iii. `CONVERSION_NOTES.md` Step 5 maps `WallName` to the visual-stimulus output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps the raw wall-name strings as categories, collects all unique stimulus names seen across processed sessions, sorts them, and encodes each trial by the index of its wall-name string. The label is then repeated across time within the trial.

ii. 
```python
stim_name = str(wall_names[t])
...
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```
```python
out = np.zeros((4, n_t), dtype=np.int64)
out[0, :] = stim_idx
```

iii. `CONVERSION_NOTES.md` Step 5 says `WallName` becomes a categorical variable, and Step 12 later justifies the high decoder accuracy partly by noting there are many visual-stimulus classes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`, using lick frame numbers assigned to the current trial.

ii. 
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `LickFr` to licking, and the code uses `LickTrind` to keep only licks from the current trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI rounds each lick frame to the nearest integer frame, subtracts `start_fr`, and writes a binary 1 into that frame of the trial-local licking vector. Multiple licks in one frame collapse to 1.

ii. 
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. `CONVERSION_NOTES.md` Step 5 summarizes the output as binary per frame, and the AI's implementation comment says "Licking - binary per frame."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The AI aligns licking to the neural data by converting lick frame numbers into offsets within the same `[StartFr:GrayFr)` trial window used for `neural`.

ii. 
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
...
frame_offset = lf_int - start_fr
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0
```

iii. The trajectory and code structure show that licking is built inside `extract_trial_data`, alongside the neural slice for the same trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `beh['ft_Pos']` over the extracted trial frames.

ii. 
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_Pos` to the position-bin output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI digitizes the framewise corridor positions into four bins using edges `[0, 10, 20, 30, 40]` in VR units, clips any overshoot into the last bin, and stores the result as integer categories.

ii. 
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
```
```python
def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly plans four 1 m bins from 0-40 VR units.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholding uses hard-coded edges `[0, 10, 20, 30, 40]`, corresponding to four 1 m bins.

ii. 
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```

iii. `CONVERSION_NOTES.md` Step 5 says position is discretized into four equal-length 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The AI aligns position by taking `ft_Pos` from the same `[StartFr:GrayFr)` frame slice as the neural data, so it remains framewise synchronized to the neural time axis.

ii. 
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. The code constructs position inside `extract_trial_data`, using the same indices used for the neural matrix.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `beh['ft_RunSpeed']` over the extracted trial frames.

ii. 
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_RunSpeed` to the speed-bin output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI performs a first pass over all selected sessions, collects `ft_RunSpeed` values from `StartFr` to `GrayFr` for every trial, computes global percentile edges at 0/25/50/75/100%, and then digitizes each trial's speed values against those global edges.

ii. 
```python
for t in range(ntrials):
    start_fr = int(np.round(beh['StartFr'][t]))
    gray_fr = int(np.round(beh['GrayFr'][t]))
    if start_fr < len(beh['ft_RunSpeed']) and gray_fr <= len(beh['ft_RunSpeed']):
        trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]
        speeds.append(trial_speed)
```
```python
flat_speeds = np.concatenate(all_speeds)
flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
```
```python
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly describes a two-pass design where Pass 1 collects running speeds for quartile computation and Pass 2 applies those bins during session processing.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholding uses percentile-based edges computed from the aggregated speed distribution across all processed sessions, with `np.digitize` applied to each frame.

ii. 
```python
speed_bin_edges = compute_speed_bin_edges(all_speeds)
```
```python
def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 6 say running speed should be split into quartile bins, and the AI implemented that with percentile edges over collected speeds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The AI aligns speed by taking `ft_RunSpeed` from the same extracted frame slice as the neural data and then discretizing that per-frame series.

ii. 
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
```

iii. The code derives speed inside `extract_trial_data`, so it stays on the same frame grid as the neural slice.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles edge cases by rounding fractional frame indices, skipping trials with fewer than 2 frames or out-of-bounds frame ranges, ignoring licks outside the kept trial window, and skipping sessions with missing behavior or too few valid trials. It does not explicitly clip all behavior streams to the imaged frame count before trial extraction.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
...
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```
```python
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```
```python
if beh_results is None:
    print(f'  WARNING: No behavioral data found for {sess_id}')
    return None
```

iii. `CONVERSION_NOTES.md` Step 9 and Step 10 explicitly mention skipped sessions with unmatched swap behavior keys and one excluded short trial, and Step 10 says it rounded float frame indices.

## 12-a. What are the most time-consuming steps of the code?

i. The AI's implementation makes the expensive steps neural-file loading and the two full dataset passes, especially Pass 1 collecting running speeds and Pass 2 loading spikes again while processing sessions.

ii. 
```python
for i, sess_id in enumerate(session_ids):
    sess_info = all_sessions[sess_id]
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
```
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk)
t_load = time.time() - t0
```

iii. `CONVERSION_NOTES.md` Step 6 is explicit about the two-pass design, and Step 11 reports per-session load times, indicating the AI regarded loading and repeated passes as the runtime bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several Python loops unvectorized: trial-by-trial speed collection, trial-by-trial extraction, the per-lick loop inside each trial, repeated per-trial construction of small arrays, and repeated loops over trials again during final assembly.

ii. 
```python
for t in range(ntrials):
    start_fr = int(np.round(beh['StartFr'][t]))
    gray_fr = int(np.round(beh['GrayFr'][t]))
    ...
```
```python
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```
```python
for trial_out in result['outputs']:
    stim_idx = stim_to_idx[trial_out['stim_name']]
    n_t = len(trial_out['licking'])
    out = np.zeros((4, n_t), dtype=np.int64)
```

iii. The AI did not document vectorization opportunities in `CONVERSION_NOTES.md`; the need to vectorize is inferred directly from the implemented loops.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats several scans: it rescans `exp_info` to compute training day for every session, reloads and rescans behavior files for every session, does a full first pass only to collect speed values, and then processes the sessions again in the second pass.

ii. 
```python
for exp_type, sessions in exp_info.items():
    for s in sessions:
        if s['mname'] == mname:
            dates.add(s['datexp'])
```
```python
for exp_type in exp_types:
    beh_path = f'data/beh/Beh_{exp_type}.npy'
    if os.path.exists(beh_path):
        beh_all = np.load(beh_path, allow_pickle=True).item()
```
```python
# Pass 1
speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
...
# Pass 2
result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly documents the two-pass design; the other repeated work is evident from the code rather than from a written justification.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores behavior matches for all session variants but discards all but `beh_results[0]`; it performs a full first pass just to throw away the collected speed arrays after computing global edges; it imports `scipy.interpolate` and defines `get_stimulus_category` and `get_brain_region_indices` without using them; and it records `n_total_frames` without using it downstream.

ii. 
```python
beh_results = load_beh_for_session(sess_id, exp_types)
...
beh, beh_key = beh_results[0]
```
```python
all_speeds = []
...
speed_bin_edges = compute_speed_bin_edges(all_speeds)
...
del all_speeds
```
```python
from scipy import interpolate
...
def get_brain_region_indices(iarea, brain_regions):
...
def get_stimulus_category(wall_name):
...
n_total_frames = spk.shape[1]
```

iii. The AI justified the two-pass speed computation in `CONVERSION_NOTES.md` Step 6, but the discarded alternative behavior matches and unused helper code are not justified there.
