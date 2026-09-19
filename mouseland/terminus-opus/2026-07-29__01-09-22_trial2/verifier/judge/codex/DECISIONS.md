# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `data/beh/Imaging_Exp_info.npy` as a master index, deduplicates sessions by `mname_datexp_blk`, then processes each session by loading one spike file from `data/spk`, one retinotopy file from `data/retinotopy`, and one behavior dictionary from the relevant `Beh_<exp_type>.npy` file.

ii. 
```python
def get_all_sessions():
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    session_map = {}
    for exp_type in info.keys():
        for entry in info[exp_type]:
            session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
            if session_id not in session_map:
                session_map[session_id] = (exp_type, entry, beh_key)
```
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
```

iii. In `CONVERSION_NOTES.md`, the AI says "All 89 sessions included" and that each unique neural recording is one session. The trajectory also shows it deliberately used the first occurrence of sessions that appeared under multiple experiment types.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname`. The output `subjects` list is built incrementally in the order each subject first appears during session processing, and each session stores the matching `subject_idx`.

ii. 
```python
sessions.append({'session_id': session_id, 'mname': entry['mname'],
                 'datexp': entry['datexp'], 'blk': entry['blk'],
                 'exp_type': exp_type, 'beh_key': beh_key, 'entry': entry})
```
```python
all_subjects = []
...
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
...
'subjects': all_subjects, 'subject_idx': np.array(all_subject_idx),
```

iii. The notes repeatedly describe the dataset as 89 sessions from 19 mice, and the split by mouse name is the direct way the agent used to preserve that structure.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` recording. If the same recording appears under multiple experiment types, only the first occurrence is kept.

ii. 
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. In the trajectory the AI states that one neural recording should correspond to one session, even when the session appears under multiple behavior experiment types. `CONVERSION_NOTES.md` says "Each unique neural recording = one session."

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd`, but the AI keeps only frames where the mouse is marked as moving (`ft_move > 0`). It treats the remaining running frames for each trial as the trial window and drops trials with fewer than two kept frames.

ii. 
```python
ft_trInd = beh['ft_trInd'][:n_frames_neural]
ft_move = beh['ft_move'][:n_frames_neural]
vr_move = ft_move > 0
...
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    if len(valid_frame_indices) < 2:
        continue
```

iii. The notes say "Frame filtering: Only VR-moving frames (ft_move > 0, matches reference)" and "Trial curation: All trials included, frames filtered for VR-moving." In the trajectory the AI explicitly decided to keep running-only frames because it believed the paper analyzed only running periods.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a dedicated whole-trial outlier filter. Instead, it filters frames within each trial to running frames and then skips any trial with fewer than two remaining frames. It also discards any session with fewer than two surviving trials.

ii. 
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    if len(valid_frame_indices) < 2:
        continue
```
```python
valid_count = len(neural_trials)
...
if valid_count < 2:
    return None
```

iii. `CONVERSION_NOTES.md` states "All trials included, frames filtered for VR-moving." The AI's trajectory reflects that it saw frame-level filtering as sufficient trial curation and did not add the reference solution's long-trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the `spks` arrays in each session's `*_neural_data.npy` file, concatenated across planes. Region labels come from `iarea` in the retinotopy file.

ii. 
```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes summarize the reference loading pattern as "Concatenates spks across imaging planes" and "Loads iarea for brain region assignment," and the AI mirrored that structure.

## 2-b. How is the `neural` data processed?

i. The AI concatenates spike planes, filters neurons by retinotopy, extracts only running frames for each trial, and stores each trial as `float32`.

ii. 
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
```
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes say "spks (concat) -> neural | Filter neurons, extract per-trial running frames" and list "Only running timepoints used" as a processing detail. The AI also chose `float32`, likely to avoid precision issues after earlier decoder-format debugging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopy. The AI keeps neurons whose `iarea` is not `-1` and not `7`, then maps the surviving codes into four coarse visual regions.

ii. 
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
iarea_filtered = iarea[neuron_mask]
```
```python
def neu_area_ID(iarea):
    idx = {}
    idx['V1'] = iarea == 8
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
```

iii. The notes explicitly justify this as matching the reference code: "Exclude iarea==-1 and iarea==7 (outside visual cortex)." The trajectory also shows the AI deciding to follow that exact filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says alignment is to trial start / corridor entry, but in code each trial is actually aligned to the first kept running frame in that trial, because non-running frames are removed before the trial arrays are built.

ii. 
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes describe the temporal alignment event as "Trial start (corridor entry)," but the trajectory shows the AI becoming concerned about long pauses and intentionally using running-only frames. That justification led it to an implicit first-running-frame alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging frame resolution at 3.17 Hz, recorded in metadata as `1000 / 3.17` ms. No temporal rebinning or resampling is applied.

ii. 
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```
```python
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FRAME_RATE,
```

iii. The notes say the frame rate is 3.17 Hz and repeatedly treat imaging frames as the native analysis grid. The AI did not add any temporal aggregation step.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `ft`. `SoundFr` gives the sound-cue frame index for each trial and `ft` provides frame timestamps.

ii. 
```python
ft = beh['ft'][:n_frames_neural]
sound_fr = beh['SoundFr']
```
```python
s_fr_int = int(np.floor(s_fr))
s_fr_frac = s_fr - s_fr_int
if 0 <= s_fr_int < len(ft) - 1:
    sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
```

iii. The notes map `SoundFr, ft` directly onto `input[0]: time_to_sound_cue`, and the trajectory explicitly identifies `SoundFr` as the relevant event timing variable.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI interpolates the fractional `SoundFr` onto the `ft` time axis and then computes `ft_trial - sound_time` in seconds. If `SoundFr` is `NaN`, it fills the whole trace with zeros.

ii. 
```python
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
else:
    s_fr_int = int(np.floor(s_fr))
    s_fr_frac = s_fr - s_fr_int
    if 0 <= s_fr_int < len(ft) - 1:
        sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
    elif s_fr_int >= len(ft) - 1:
        sound_time = ft[-1]
    else:
        sound_time = ft[0]
    time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes say "Time from frame to sound cue (seconds)," and the trajectory shows the AI switching to actual frame times after noticing irregular gaps in frame indices. Its implementation uses signed elapsed time relative to the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same per-trial `ft_trial` vector used to select neural frames, so it is aligned to the same running-only frame indices as the neural data.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
ft_trial = ft[valid_frame_indices]
...
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI's general alignment rule was to compute all time-varying streams on the same frame indices used for the neural trial matrices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's mouse name and session date, specifically `mname` and `datexp` from the session records built out of `Imaging_Exp_info.npy`.

ii. 
```python
mouse_sessions[s['mname']].append((i, s))
...
datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))
```

iii. The notes map `datexp` to `day_of_training`, and the session metadata assembled from the master index supply the per-mouse calendar dates needed for that computation.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by calendar date. The AI then assigns `day_of_training` as the number of elapsed calendar days since that mouse's first recorded session, and broadcasts that scalar across each trial.

ii. 
```python
def compute_day_of_training(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((i, s))
    day_of_training = np.zeros(len(sessions))
    for mname, msessions in mouse_sessions.items():
        dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))) for idx, s in msessions]
        dates.sort(key=lambda x: x[1])
        first_date = dates[0][1]
        for idx, date in dates:
            day_of_training[idx] = (date - first_date).days
```
```python
input_trial[1] = np.float32(day_val)
```

iii. `CONVERSION_NOTES.md` explicitly states "Days since first session per mouse." The AI chose real elapsed days rather than an ordinal session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `ft` and from the first retained frame of each trial after running-frame filtering. The code does not use `StartFr`.

ii. 
```python
ft = beh['ft'][:n_frames_neural]
...
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes map `ft` to `time_since_trial_start` and describe it as "Elapsed time from trial start." The trajectory shows the AI redefining "trial start" operationally as the first kept frame in the filtered trial trace.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI subtracts the timestamp of the first kept frame in the trial from every retained frame time and converts the difference from MATLAB days to seconds.

ii. 
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. In the trajectory the AI explains that it switched away from frame-index differences because removing non-running frames created large gaps; it therefore used actual timestamps instead.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on `ft_trial`, which is indexed by the same `valid_frame_indices` used for the per-trial neural array, so it is aligned to the same running-only time bins as the neural data.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI's alignment strategy was to derive all time-varying signals from the same filtered frame index vector per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial `isRew` flag in the behavior data.

ii. 
```python
is_rew = beh['isRew']
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The notes map `isRew` to `reward_availability` with no extra transformation beyond conversion to 0/1.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No substantial processing is applied. The flag is converted to `1.0` or `0.0` and broadcast across all time bins of the trial.

ii. 
```python
input_trial = np.zeros((4, n_t), dtype=np.float32)
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. `CONVERSION_NOTES.md` describes this simply as "1=rewarded, 0=not," so the AI treated it as a per-trial binary context variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, with the stimulus lookup table created from the union of `UniqWalls` across all sessions.

ii. 
```python
for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(wn))
...
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```
```python
wall_name = beh['WallName']
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. The notes map `WallName` to `visual_stimulus_category`, and the trajectory shows the AI deciding to preserve all observed wall names rather than collapsing them.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse stimuli to four base categories. Instead, it enumerates all unique wall names across the dataset, maps each wall name to an integer class, and broadcasts the per-trial class across all time bins in that trial.

ii. 
```python
all_stim_names = get_all_stim_names(all_sessions)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```
```python
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
...
output_trial[0] = stim_idx
```
```python
'output_values': [all_stim_names, ['no_lick', 'lick'], ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ['Q1', 'Q2', 'Q3', 'Q4']],
```

iii. The notes and final review explicitly celebrate "15 categories" for visual stimulus accuracy, so this was a deliberate decision rather than an accident.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the list of lick frame indices for the session.

ii. 
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
```

iii. The notes map `LickFr` directly onto a binary per-frame licking output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds each lick's fractional frame index to the nearest integer frame, drops out-of-range licks, and marks those frames as 1 in a binary per-frame vector.

ii. 
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary
```

iii. The notes summarize this as "Binary per frame." The AI also added a guard for empty lick arrays.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by selecting the same `valid_frame_indices` used for neural data, so the lick trace is on the same running-only per-trial grid.

ii. 
```python
lick_trial = lick_binary[valid_frame_indices]
...
output_trial[1] = lick_trial
```

iii. The AI consistently aligned all per-frame outputs by indexing them with the same trial-specific frame vector used for the neural matrix.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the per-frame position signal.

ii. 
```python
ft_Pos = beh['ft_Pos'][:n_frames_neural]
...
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The notes map `ft_Pos` to `position_bin`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI divides position by 15 decimeters and clips to 4 categories, effectively making four 1.5 m bins across the full 6 m corridor.

ii. 
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```
```python
'output_values': [all_stim_names, ['no_lick', 'lick'], ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ['Q1', 'Q2', 'Q3', 'Q4']],
```

iii. The notes say "4 bins of 15dm over 60dm." In the trajectory the AI explicitly says it chose 15 dm bins because the full corridor is 6 m and it wanted equal coverage of that full space.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are assigned by `floor(position / 15.0)` and then clipped into the range 0 to 3.

ii. 
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The trajectory shows the AI changed this after noticing that mapping the full 6 m corridor into 1 m bins left the grey zone collapsed into the last category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by indexing `ft_Pos` with the same `valid_frame_indices` vector used for the neural data, so it is on the same running-only frame grid.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
...
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The AI's general alignment convention was to compute all time-varying outputs on the neural trial's selected frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, the per-frame running speed signal.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The notes map `ft_RunSpeed` directly to `running_speed_bin`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes one set of global quartile thresholds from all sessions using only frames with `ft_move > 0`. It optionally subsamples to 5000 speed values per session before concatenation. Within each trial it bins frame speeds by those global percentile boundaries.

ii. 
```python
def compute_global_speed_quartiles(sessions):
    all_speeds = []
    ...
    vr_move = beh['ft_move'] > 0
    speeds = beh['ft_RunSpeed'][vr_move]
    if len(speeds) > 5000:
        speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
    all_speeds.append(speeds)
    ...
    quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The notes call this out explicitly: "Speed bins: Global quartile-based (computed from all sessions)." The trajectory also shows the AI optimizing this to avoid loading neural files during quartile estimation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the three percentile cut points returned by `np.percentile(all_speeds, [25, 50, 75])`, and `np.digitize` converts each frame's speed into category 0, 1, 2, or 3.

ii. 
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The notes say the bins are quartile based. Unlike the position bins, the thresholds are data-driven rather than fixed distances.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by taking `ft_RunSpeed` on the same `valid_frame_indices` used for the neural trial matrix, so it is on the same running-only per-trial grid.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. As with the other time-varying streams, the AI aligns running speed by direct indexing onto the trial's selected neural frames.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates all behavior time series to the number of neural frames, drops out-of-range lick events, handles empty `LickFr`, treats `NaN` sound cues as all-zero `time_to_sound_cue`, and clips sound times that would interpolate beyond the recorded frame range.

ii. 
```python
ft = beh['ft'][:n_frames_neural]
ft_trInd = beh['ft_trInd'][:n_frames_neural]
ft_Pos = beh['ft_Pos'][:n_frames_neural]
ft_move = beh['ft_move'][:n_frames_neural]
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
```
```python
if len(lick_fr) == 0:
    return lick_binary
idx = idx[(idx >= 0) & (idx < n_frames)]
```
```python
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
else:
    ...
    elif s_fr_int >= len(ft) - 1:
        sound_time = ft[-1]
    else:
        sound_time = ft[0]
```

iii. The notes mention behavior-neural length mismatches and the trajectory shows the AI adding explicit guards after inspecting edge cases. Its approach was to coerce questionable values into valid bins rather than fail hard.

## 12-a. What are the most time-consuming steps of the code?

i. The code is dominated by loading huge neural data files session by session, plus the per-trial extraction loop over large neural matrices. The AI also added a global speed-quartile pass over all sessions before session processing.

ii. 
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    ...
    neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```
```python
speed_quartiles = compute_global_speed_quartiles(all_sessions)
```

iii. The notes report 73.1 minutes for the full run and list memory-saving optimizations around session-by-session processing. In the trajectory the AI explicitly describes neural file loading and the trial extraction loop as the main runtime bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization targets are the per-trial loop that repeatedly builds masks and slices the giant neural matrix, and the dataset-wide loops that repeatedly scan behavior files for stimulus names and speed quartiles.

ii. 
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```
```python
for session in sessions:
    ...
    for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
        all_stim.add(str(wn))
```
```python
for i, session in enumerate(sessions):
    ...
    speeds = beh['ft_RunSpeed'][vr_move]
```

iii. The trajectory explicitly says the "trial extraction loop" was the main bottleneck during sample conversion and that speed-quartile computation also needed optimization.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats several dataset-wide passes over behavior data: once to enumerate session metadata, once to compute day-of-training, once to gather all stimulus names, once to compute speed quartiles, and again during actual session processing. Behavior files are also reloaded per session in `process_session`.

ii. 
```python
all_sessions = get_all_sessions()
day_of_training = compute_day_of_training(all_sessions)
all_stim_names = get_all_stim_names(all_sessions)
speed_quartiles = compute_global_speed_quartiles(all_sessions)
```
```python
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
```

iii. The notes mention `get_all_stim_names`, global quartile computation, and then full session processing as separate stages. This repeated scanning was part of the AI's attempt to keep the conversion simple while managing memory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extra artifacts that are not needed by downstream decoder training: a complete `stim_to_idx` metadata mapping for 15 wall identities, dataset-wide speed quartile boundaries, optional diagnostic plotting code, and a temp-file cache pass that only exists to work around memory usage during conversion.

ii. 
```python
'speed_quartile_boundaries': speed_quartiles.tolist(),
'session_info': session_infos, 'stim_to_idx': stim_to_idx,
'neural_data_type': 'Suite2p deconvolved fluorescence traces',
'deconvolution_decay_s': 0.75,
'neuron_filter': 'Excluded neurons outside visual cortex (iarea==-1 or iarea==7)',
'frame_filter': 'Only VR-moving frames (ft_move > 0)',
```
```python
if args.show_processing and i < 2:
    plot_processing(result, session, speed_quartiles, stim_to_idx, i)
```
```python
temp_fn = f'cache/session_{i:03d}.pkl'
...
for tf in temp_files:
    with open(tf, 'rb') as f:
        sd = pickle.load(f)
```

iii. The notes call out the temp-file approach and extra validation tooling as optimizations, but none of these extra artifacts are used by the decoder itself. They reflect the AI's effort to make its own workflow manageable rather than requirements of the target format.
