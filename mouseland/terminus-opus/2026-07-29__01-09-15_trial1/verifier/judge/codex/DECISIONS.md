# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script first loads `data/beh/Imaging_Exp_info.npy`, builds one `session_id` per `(mname, datexp, blk)`, and stores the experiment types associated with that session. At processing time it looks through the corresponding `Beh_<exp_type>.npy` files for either the base `session_id` key or swap-suffixed keys, then uses only the first behavioral match it finds. Spikes are loaded per session from `data/spk/<session_id>_neural_data.npy`, and retinotopy is loaded per session from `data/retinotopy/<mname>_<datexp>_trans.npz`.

ii. 
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
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
spk = load_spk(mname, datexp, blk)
retino = load_retino(mname, datexp)
```

iii. In `CONVERSION_NOTES.md`, the AI says the script "Loads all sessions from exp_info" and later notes that 13 sessions were skipped because swap behavior keys were not handled cleanly. The trajectory shows the AI understood that swap sessions shared neural recordings with base sessions, but then accepted the 76-session dataset for practical reasons instead of fully refactoring the loader.

## 1-b. How are the data split into subjects?

i. Subjects are defined by mouse name (`mname`). After all sessions are processed, the script takes the sorted unique mouse IDs from the surviving session results and uses them to build `subjects` and `subject_idx`.

ii. 
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
```

```python
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. `CONVERSION_NOTES.md` states that the converted data contain 19 subjects and that sessions are organized per mouse. The trajectory also describes the dataset as 19 mice and uses mouse identity as the subject key throughout.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` triple, encoded as `"<mname>_<datexp>_<blk>"`. The helper that builds the session table deduplicates entries across experiment types by that ID and aggregates a list of experiment types per session.

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

iii. The AI's notes say there are 89 unique neural sessions and explain that some behavior sessions are stored as swap variants of the same neural recording. The trajectory shows the AI explicitly treated session identity as mouse/date/block.

## 1-d. How are the data split into trials?

i. Trials are defined by rounded `StartFr` and `GrayFr` boundaries. For each trial, the script takes a contiguous slice from `start_fr` to `gray_fr`, capped at `start_fr + 1000` frames, and treats that as the trial window. It does not use `ft_trInd` or `ft_CorrSpc` to pick per-frame trial membership.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
```

```python
end_fr = min(start_fr + n_timepoints, gray_fr)
actual_frames = end_fr - start_fr
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps `spks` to "Extract per-trial from StartFr to GrayFr" and says the corridor portion is the relevant trial segment. The trajectory similarly says the corridor portion from `StartFr` to `GrayFr` is the trial window because that is where the visual stimulus is present.

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped only if they have fewer than 2 frames, if the rounded window falls outside the neural recording, or if a session ends up with fewer than 2 valid trials. Very long trials are not dropped using a dataset-level percentile rule; they are only truncated by the hard `n_timepoints=1000` cap.

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
trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
...
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. `CONVERSION_NOTES.md` says trials with `<2` frames are excluded and says the maximum trial length is capped at 1000 frames. There is no note describing a dataset-wide traversal-length outlier filter, which matches the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the `spks` arrays in each session's neural `.npy` file. Brain-region labels are derived from `iarea` in the retinotopy `.npz` file.

ii. 
```python
dat = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```

```python
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
ix = neu_area_ID(dtrans['iarea'])
return {'iarea': dtrans['iarea'], 'neu_ar_idx': ix}
```

iii. `CONVERSION_NOTES.md` Step 1 identifies `load_spk`, `load_retino`, and `neu_area_ID` as the key reference functions, and Step 5 maps `spks` directly to `neural`.

## 2-b. How is the `neural` data processed?

i. The script concatenates imaging planes, slices each trial window directly from the session array, and casts the result to `float16`. There is no additional denoising, deconvolution, temporal smoothing, or resampling.

ii. 
```python
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```

```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. In `CONVERSION_NOTES.md`, the AI says the neural data are already deconvolved Suite2p traces and says "Neural data stored as float16 to reduce file size." It also explicitly notes that the reference code does not require further neuron-quality filtering in its interpretation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not drop neurons outside the four visual areas. Instead, it assigns each neuron to one of `['V1', 'mHV', 'lHV', 'aHV', 'unassigned']`, leaving unmatched neurons in an `"unassigned"` category that is still kept.

ii. 
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
...
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)
for i, region in enumerate(brain_regions[:-1]):
    if region in region_map:
        neuron_region_idx[region_map[region]] = i
```

iii. `CONVERSION_NOTES.md` Step 10 says "No neuron filtering: Consistent with reference code." The trajectory and notes therefore show that the AI deliberately chose to keep all neurons and only annotate region identity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial start, operationalized as corridor entry at `StartFr`. Each trial begins at the rounded `StartFr` and ends at the rounded `GrayFr`, so all trial arrays start at trial onset but have variable duration.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
```

```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. `CONVERSION_NOTES.md` repeatedly states that temporal alignment is based on trial start or corridor entry, and the trajectory explicitly says the corridor portion from `StartFr` to `GrayFr` should be used because that is when the stimulus is present.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal bin size is one imaging frame at 3.17 Hz, i.e. about 315.5 ms per bin. No extra temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

```python
'time_bin_size': TIME_BIN_MS,
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 3 record the 3.17 Hz frame rate from the reference notebook and paper, and the metadata expose that frame duration directly.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` together with the frame index relative to `StartFr`. The script does not use the behavioral timestamp vector `ft` for this variable.

ii. 
```python
sound_fr = beh['SoundFr'][trial_idx]
sound_frame_offset = sound_fr - start_fr
```

```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps `SoundFr` to `time_to_sound_cue` and describes the transform as "Time in seconds." The trajectory describes the variable as a signed time relative to the cue, using frame offsets.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The script computes a cue offset in frames, then converts each within-trial frame index to seconds as `(current_frame_index - sound_offset) / FRAME_RATE`. This yields negative values before the cue and positive values after it.

ii. 
```python
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The AI's code comments describe this as "Signed time from current frame to sound cue," and `CONVERSION_NOTES.md` treats it as a time-in-seconds transform from `SoundFr`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed for exactly the same extracted trial window as the neural data, using one value per frame in `actual_frames`.

ii. 
```python
neural = trial_data['neural']
...
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. The notes and code both organize all trialwise signals on the same per-frame trial slice, so the AI's intended justification is simple framewise alignment within each extracted window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and `datexp` in `Imaging_Exp_info.npy`. The session date is used to place each session in chronological order within mouse.

ii. 
```python
for exp_type, sessions in exp_info.items():
    for s in sessions:
        if s['mname'] == mname:
            dates.add(s['datexp'])
```

```python
sorted_dates = sorted(dates)
return sorted_dates.index(datexp)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `datexp` to `day_of_training`, and the trajectory describes it as a "chronological day index for this mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the script collects all unique session dates, sorts them, and returns the zero-based index of the current date. That scalar is then repeated across all time bins of the trial.

ii. 
```python
sorted_dates = sorted(dates)
return sorted_dates.index(datexp)
```

```python
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. The AI's notes explicitly call this a chronological day index, not a block index, so its justification is that training day should correspond to the order of recorded dates for each mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from trial start frame `StartFr` and the within-trial frame index. The script does not use the `ft` timestamp vector.

ii. 
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
```

```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes this mapping as "frame - StartFr" converted to time in seconds, which is exactly what the code does.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script assumes the first extracted frame is time zero and assigns each later frame a value of `i / FRAME_RATE` seconds, where `i` is the within-trial frame index.

ii. 
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The justification in the notes is that the signal should represent elapsed time from corridor entry, so a frame-count-to-seconds conversion is sufficient under the constant frame-rate assumption.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It has one value per extracted neural frame and is built on the same `actual_frames` length as the trial's neural matrix.

ii. 
```python
neural = trial_data['neural']
...
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. The AI's overall design is to compute all inputs directly on the extracted trial slice, so the intended alignment is one-to-one with neural bins.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, using one trial-level value per trial.

ii. 
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `isRew` directly to `reward_availability` with no extra derived source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No substantive transform is applied. The script converts the trial's `isRew` value to float and broadcasts it across all time bins in that trial.

ii. 
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The notes describe this variable simply as a binary indicator of rewarded corridor membership, so the justification is straightforward broadcasting of an existing trial label.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, taking the raw wall/stimulus name for each trial.

ii. 
```python
wall_names = beh['WallName']
...
stim_name = str(wall_names[t])
```

iii. `CONVERSION_NOTES.md` Step 5 maps `WallName` to `visual_stimulus`, and the trajectory later discusses the decoder having 13 visual-stimulus classes, which reflects the raw names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script keeps the raw `WallName` labels rather than collapsing them to four texture families. It collects all unique names across processed sessions, sorts them, maps each name to an integer category, and repeats that category across all time bins of the trial.

ii. 
```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```

```python
stim_idx = stim_to_idx[trial_out['stim_name']]
out[0, :] = stim_idx
```

iii. The trajectory records that the AI interpreted the decoder target as having 13 stimulus classes and later judged the resulting 13-class decoder accuracy as acceptable. That is the clearest evidence of its justification for keeping raw wall names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and filtered by `LickTrind` so that only lick events belonging to the current trial are used.

ii. 
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The AI's notes map `LickFr` to a binary per-frame licking signal, and the code adds `LickTrind` to localize events to each trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the script creates a zero-filled per-frame vector, rounds each lick frame to the nearest integer frame, shifts it relative to `start_fr`, and sets the corresponding bin to 1 if the lick falls inside the extracted window.

ii. 
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. `CONVERSION_NOTES.md` Step 5 describes this variable as "Binary per frame." The AI therefore justified the transform as turning event times into a categorical framewise signal.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by building the lick vector on the same trial window defined for the neural slice, with one bin per neural frame in `actual_frames`.

ii. 
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
...
licking = np.zeros(actual_frames, dtype=np.float32)
```

iii. The notes consistently describe the trial window as the common alignment frame for neural, input, and output variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos` within each extracted trial window.

ii. 
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_Pos` directly to the position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized from VR units into four 1 m bins spanning the 4 m textured corridor.

ii. 
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
...
def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```

iii. `CONVERSION_NOTES.md` Step 2 says 1 VR unit is 0.1 m and the textured corridor is 40 VR units, and Step 5 explicitly plans four 1 m bins from that geometry.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories use fixed thresholds at 10, 20, 30, and 40 VR units, corresponding to 1 m spatial bins. `np.digitize` assigns each frame to bins 0 through 3, then clips any out-of-range value back into that range.

ii. 
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
```

```python
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
return bins.astype(np.int64)
```

iii. The AI's planned mapping in `CONVERSION_NOTES.md` says "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins," which is exactly the thresholding scheme implemented.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position vector is sliced over the same `start_fr:end_fr` window as the neural trial, so it has one value per neural frame.

ii. 
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. The script's general alignment strategy is framewise co-slicing of all time-varying variables, and the notes describe the same common trial window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed` over the extracted trial windows.

ii. 
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_RunSpeed` directly to the speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first performs a full pass over the selected sessions to collect all trial speed traces from `StartFr:GrayFr`, concatenates them, computes global 0/25/50/75/100 percentile edges, and then digitizes each frame's speed by those thresholds.

ii. 
```python
all_speeds = []
for i, sess_id in enumerate(session_ids):
    sess_info = all_sessions[sess_id]
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
    if speeds is not None:
        all_speeds.extend(speeds)
speed_bin_edges = compute_speed_bin_edges(all_speeds)
```

```python
def compute_speed_bin_edges(all_speeds):
    flat_speeds = np.concatenate(all_speeds)
    flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
    edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
    return edges
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says "Pass 1: Collect running speeds for quartile computation," and Step 5 says speed should be put into four quartile bins. The trajectory shows this extra first pass was a deliberate design choice.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the global percentile edges returned by `compute_speed_bin_edges`. `np.digitize` applies the inner three edges, then clips the result into categories 0 through 3.

ii. 
```python
def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. The AI's notes and trajectory consistently describe the output as quartile-binned speed, and the code implements quartiles via global percentile thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced from the same per-trial frame window as the neural data, so it is one value per neural frame.

ii. 
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. As with position and licking, the AI's intended justification is common framewise alignment across all extracted trial signals.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles edge cases by skipping sessions with no matched behavioral data, skipping trials shorter than 2 frames, skipping trials whose rounded window falls outside the neural recording, and ignoring `NaN` values only when computing speed quartiles. It rounds floating frame indices for `StartFr`, `GrayFr`, and lick frames rather than interpolating. It does not trim all behavioral streams globally to the imaged frame count; instead it rejects out-of-range trials.

ii. 
```python
if beh_results is None:
    print(f'  WARNING: No behavioral data found for {sess_id}')
    return None
```

```python
if avail_frames < 2:
    return None
...
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

```python
flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
```

iii. `CONVERSION_NOTES.md` explicitly mentions skipped swap sessions, rounded float frame indices, exclusion of one `<2`-frame trial, and speed-edge computation over collected running speeds. Those notes match the limited error-handling logic in the script.

## 12-a. What are the most time-consuming steps of the code?

i. The heaviest work is loading the large spike files during session processing. A second meaningful cost is the extra dataset-wide pass used only to collect running speeds for quartile thresholds.

ii. 
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk)
t_load = time.time() - t0
```

```python
for i, sess_id in enumerate(session_ids):
    sess_info = all_sessions[sess_id]
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
```

iii. The conversion notes report roughly 10 to 11 seconds of load time for sample sessions and explicitly structure the script into "Pass 1" and "Pass 2," which shows the AI understood both spike I/O and the extra speed pass as major costs.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain fully Python-level: looping over trials to extract windows, looping over licks within each trial, building `time_to_cue` and `time_since_start` with list comprehensions, and rescanning many behavior files session-by-session during lookup.

ii. 
```python
for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
```

```python
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The AI did not document a vectorization plan for these loops. Its notes emphasize getting the conversion working and adding a separate speed-collection pass, not optimizing these inner loops.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats behavioral-file lookup and loading for each session, performs two passes over the sessions for speed binning and then actual conversion, and recomputes the training-day ordering by rescanning `exp_info` for every processed session.

ii. 
```python
for exp_type in exp_types:
    beh_path = f'data/beh/Beh_{exp_type}.npy'
    if os.path.exists(beh_path):
        beh_all = np.load(beh_path, allow_pickle=True).item()
```

```python
for i, sess_id in enumerate(session_ids):
    ...
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
...
for i, sess_id in enumerate(session_ids):
    ...
    result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

```python
for exp_type, sessions in exp_info.items():
    for s in sessions:
        if s['mname'] == mname:
            dates.add(s['datexp'])
```

iii. `CONVERSION_NOTES.md` Step 6 names the explicit first and second passes, and the trajectory shows the AI focused on correctness and pragmatic completion rather than reducing repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is that `load_beh_for_session` may load and deduplicate multiple behavioral variants for a session, but `process_session` immediately discards all but the first match. That means additional lookup work is performed without affecting the final converted data.

ii. 
```python
results = []
...
if sess_id in beh_all:
    results.append((beh_all[sess_id], sess_id))
for key in sorted(beh_all.keys()):
    if key.startswith(sess_id + '_swap'):
        results.append((beh_all[key], key))
```

```python
# Use the first (primary) behavioral data
beh, beh_key = beh_results[0]
```

iii. The trajectory shows the AI understood that swap-session handling was more complex and decided not to fully refactor around it. The result is a loader that does extra work gathering variants and then throws most of that work away.
