# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads session metadata from `data/beh/Imaging_Exp_info.npy`, deduplicates to one entry per unique neural recording (`mname_datexp_blk`), then loads per-session neural data from `data/spk`, retinotopy from `data/retinotopy`, and behavior from the matching `Beh_<exp_type>.npy` file. It then iterates through all `ntrials` in the selected behavior record.

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
ntrials = beh['ntrials']
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent states that all 89 sessions are included and that each unique neural recording is treated as one session. Trajectory steps 21-25 show the same reasoning: use all 89 unique recordings and load matching behavior plus retinotopy for each.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse name `mname`. Unique subject names are accumulated in first-seen order while sessions are processed, and each session stores the index of its mouse in that list.

ii. 
```python
sessions.append({'session_id': session_id, 'mname': entry['mname'],
                 'datexp': entry['datexp'], 'blk': entry['blk'],
                 'exp_type': exp_type, 'beh_key': beh_key, 'entry': entry})
```

```python
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. `CONVERSION_NOTES.md` Step 2 reports 19 unique subjects and Step 9 says the converted dataset also has 19 subjects. The trajectory repeatedly refers to `mname` as the subject identity.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique recording identifier `mname_datexp_blk`. If the same recording appears in multiple experiment lists or has `stimtype` variants, the first encountered entry is kept and later duplicates are ignored.

ii. 
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. `CONVERSION_NOTES.md` Step 5 says "Each unique neural recording = one session". Trajectory steps 22-23 make the same argument: recordings can appear in multiple experiment types, but each neural file should only become one decoder session.

## 1-d. How are the data split into trials?

i. Within each session, trials are defined by `beh['ntrials']` and frame-level trial assignments `beh['ft_trInd']`. For each trial index, the agent keeps only frames where `ft_trInd == trial_idx` and `ft_move > 0`, then uses those kept frames as the trial time axis.

ii. 
```python
ntrials = beh['ntrials']
ft_trInd = beh['ft_trInd'][:n_frames_neural]
ft_move = beh['ft_move'][:n_frames_neural]
vr_move = ft_move > 0

for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

iii. `CONVERSION_NOTES.md` Step 5 says the neural data are extracted as per-trial running frames. Trajectory steps 24-28 justify using raw frame-level data aligned to trial start rather than the notebook's position interpolation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not filtered by any behavioral or stimulus QC beyond the running-frame filter. A trial is dropped only if fewer than two running frames remain after `ft_move > 0` filtering. Entire sessions are dropped only if fewer than two such trials remain.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
if len(valid_frame_indices) < 2:
    continue
...
if valid_count < 2:
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 says "All trials included, frames filtered for VR-moving" and Step 10 repeats that all trials were processed. The trajectory shows the agent added the `<2` frame / `<2` trial rules to satisfy decoder format requirements rather than because of a reference-paper curation rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the `spks` arrays stored in each `*_neural_data.npy` file, concatenated across imaging planes.

ii. 
```python
def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. `CONVERSION_NOTES.md` Step 1 identifies `load_spk` as the reference loading function and Step 3 identifies the neural data as Suite2p deconvolved fluorescence traces. The trajectory step 16 explicitly notes that `spks` is a list of per-plane arrays concatenated into a neuron-by-frame matrix.

## 2-b. How is the `neural` data processed?

i. The agent keeps the native deconvolved traces, concatenates planes, filters neurons by `iarea`, and then slices each trial into a variable-length sequence of running frames. It does not perform the reference notebook's position-based interpolation into 60 spatial bins.

ii. 
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `spks` to neural data via neuron filtering and per-trial running-frame extraction. Trajectory step 24 explicitly says the agent chose time-aligned raw frames instead of the reference `get_interpPos_spk` pipeline because the task requested temporal alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using retinotopy labels: neurons with `iarea == -1` or `iarea == 7` are excluded as outside the four target visual-region groups. Frames are filtered to running frames with `ft_move > 0`.

ii. 
```python
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
...
vr_move = ft_move > 0
valid_mask = (ft_trInd == trial_idx) & vr_move
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 5 both call out these two filters as the key curation rules copied from the reference code. The trajectory step 33 says the agent intentionally matched the `iarea` filter from the authors' utilities.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent claims alignment to trial start / corridor entry, but operationally it aligns each trial to the first retained running frame in that trial. It does not use `StartFr`, and once non-running frames are removed the trial begins at the first `ft_trInd == trial_idx` frame that also has `ft_move > 0`.

ii. 
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
...
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. `metadata['temporal_alignment_event']` is set to "Trial start (corridor entry)", and trajectory steps 24 and 28 explain that the agent intended a trial-start-aligned representation. The code-level choice was justified in the trajectory as using the kept running frames as the aligned trial sequence.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging frame rate, 3.17 Hz, corresponding to about 315.46 ms per time bin. No extra temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FRAME_RATE,
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 3 both identify 3.17 Hz as the native calcium imaging rate. Trajectory steps 17-18 show the agent checking the units of `ft` and deciding to stay at the native frame resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `beh['SoundFr']` for the cue location in frame units and `beh['ft']` for frame timestamps.

ii. 
```python
ft = beh['ft'][:n_frames_neural]
sound_fr = beh['SoundFr']
...
s_fr = sound_fr[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `SoundFr, ft` to `time_to_sound_cue`. Trajectory steps 17, 27, and 28 discuss using `SoundFr` plus frame timestamps rather than cue position.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The agent converts fractional `SoundFr` into a cue time by interpolating between adjacent `ft` timestamps, then computes a signed continuous series `ft_trial - sound_time` in seconds. If `SoundFr` is `NaN`, it fills the whole vector with zeros.

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

iii. `CONVERSION_NOTES.md` Step 5 says the transform is "Time from frame to sound cue (seconds)" using actual MATLAB-datenum frame times. Trajectory steps 49-50 show the agent explicitly revising this computation to use `ft` rather than frame indices.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed only on the same `valid_frame_indices` used for `neural_trial`, so it shares the same variable-length per-trial time axis as the neural matrix.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
ft_trial = ft[valid_frame_indices]
...
input_trial[0] = time_to_sound
```

iii. The agent's general alignment decision in trajectory steps 24 and 28 was to have all trial-varying variables share the same retained-frame index. `CONVERSION_NOTES.md` Step 5 presents `time_to_sound_cue` as a per-frame input alongside neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session metadata fields `mname` and `datexp` in `Imaging_Exp_info.npy`.

ii. 
```python
def compute_day_of_training(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((i, s))
    ...
    dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))) for idx, s in msessions]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `datexp` to `day_of_training`. Trajectory step 28 explains why the agent preferred calendar date differences over the coarse before/after-learning indicators in experiment labels.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by recording date and each session is assigned the integer number of days elapsed since that mouse's first recording date. The value is then broadcast across all timepoints in the trial.

ii. 
```python
dates.sort(key=lambda x: x[1])
first_date = dates[0][1]
for idx, date in dates:
    day_of_training[idx] = (date - first_date).days
...
input_trial[1] = np.float32(day_val)
```

iii. The trajectory step 28 explicitly says the agent chose "days from the first session for each mouse". `CONVERSION_NOTES.md` Step 5 summarizes the same decision.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not derived from any raw-data variable, because the agent did not create an `environment_type` input at all.

ii. 
```python
'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
```

iii. `CONVERSION_NOTES.md` Step 5 lists only four decoder inputs, matching the task instructions. The trajectory never proposes an additional environment-type input.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The variable is omitted from the converted dataset.

ii. 
```python
input_trial = np.zeros((4, n_t), dtype=np.float32)
input_trial[0] = time_to_sound
input_trial[1] = np.float32(day_val)
input_trial[2] = time_since_start
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The omission follows the agent's interpretation of the Decoder Task section, which only asked for four inputs. `CONVERSION_NOTES.md` Step 5 documents exactly those four.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the frame timestamp vector `beh['ft']` after selecting the frames retained for a given trial.

ii. 
```python
ft = beh['ft'][:n_frames_neural]
...
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft` to `time_since_trial_start`. Trajectory step 50 says the agent switched to actual `ft` timestamps because using frame indices was incorrect when running frames had gaps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent subtracts the first kept frame time in the trial from every kept frame time and converts the MATLAB-datenum difference from days to seconds.

ii. 
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Trajectory steps 49-50 document the justification: once non-running frames were removed, frame indices no longer gave correct elapsed time, so the computation was changed to use real timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned on the same kept running frames used for `neural_trial`, with zero defined at the first retained frame of that trial.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
...
input_trial[2] = time_since_start
```

iii. The trajectory step 28 says the agent wanted all trial-varying quantities to share the frame-wise trial alignment. `CONVERSION_NOTES.md` Step 5 presents this as a time-varying per-trial signal.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `beh['isRew']`.

ii. 
```python
is_rew = beh['isRew']
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `isRew` to `reward_availability` and defines it as `1=rewarded, 0=not`. The trajectory step 21 also reasons that unsupervised and naive sessions would contribute zeros here.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts each trial's `isRew` boolean into `1.0` or `0.0` and repeats that scalar across every timepoint in the trial.

ii. 
```python
input_trial = np.zeros((4, n_t), dtype=np.float32)
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. `CONVERSION_NOTES.md` Step 5 describes this as a discrete per-trial variable. The trajectory uses the same interpretation when deciding to include all sessions.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial wall/stimulus labels `beh['WallName']`, with the set of possible categories collected from `beh['UniqWalls']` across all sessions.

ii. 
```python
for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(wn))
...
wall_name = beh['WallName']
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `WallName` to `visual_stimulus_category`. The sample conversion log in the trajectory also lists the discovered global stimulus vocabulary.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent builds a global sorted list of stimulus names, maps each name to an integer index, and then fills the entire trial time axis with that constant category index.

ii. 
```python
all_stim_names = get_all_stim_names(all_sessions)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
output_trial = np.zeros((4, n_t), dtype=np.int32)
output_trial[0] = stim_idx
```

iii. `CONVERSION_NOTES.md` Step 5 states "Map to index" for `WallName`. The trajectory step 27 says the decoder output should be per-trial stimulus category, which motivated broadcasting it across timepoints.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `beh['LickFr']`, converted into a recording-length binary lick vector and then indexed at the kept trial frames.

ii. 
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    ...
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
```

```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
lick_trial = lick_binary[valid_frame_indices]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `LickFr` to a binary per-frame licking output. Trajectory step 27 explicitly notes `LickFr` as the relevant frame-level source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent rounds lick frame indices to the nearest integer frame, clips them to valid range, marks those frames as 1, and leaves all others at 0.

ii. 
```python
idx = np.round(lick_fr).astype(int)
idx = idx[(idx >= 0) & (idx < n_frames)]
lick_binary[idx] = 1
```

iii. The trajectory shows the agent deciding on a binary per-frame licking target to match the decoder specification. `CONVERSION_NOTES.md` Step 5 summarizes that same binary representation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The trial lick vector is obtained by slicing the recording-length binary lick array with the exact same `valid_frame_indices` used for the neural trial.

ii. 
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
lick_trial = lick_binary[valid_frame_indices]
output_trial[1] = lick_trial
```

iii. The agent's general frame-sharing alignment decision is described in trajectory steps 24 and 28. `CONVERSION_NOTES.md` Step 5 presents licking as a time-varying output paired to trial frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level position signal `beh['ft_Pos']`.

ii. 
```python
ft_Pos = beh['ft_Pos'][:n_frames_neural]
...
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_Pos` to `position_bin`. The trajectory steps 24 and 64-68 show the agent thinking through corridor length and binning.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent does not keep continuous position. It divides `ft_Pos` by `15.0`, floors the result, clips to `[0, 3]`, and stores the resulting integer category per kept frame.

ii. 
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
output_trial[2] = pos_bins
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 10 explicitly describe "4 equal bins of 15dm over 60dm". Trajectory step 64 discusses the skew this creates and why the agent moved to 15 dm bins over the full 6 m trial.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Category boundaries are `0-15 dm`, `15-30 dm`, `30-45 dm`, and `45-60 dm`, implemented by `floor(ft_Pos / 15)` and clipping to 0-3. The metadata labels these bins as `0-1.5m`, `1.5-3m`, `3-4.5m`, `4.5-6m`.

ii. 
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
...
'output_values': [all_stim_names, ['no_lick', 'lick'], ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ['Q1', 'Q2', 'Q3', 'Q4']],
```

iii. `CONVERSION_NOTES.md` Step 5 lists the 15 dm binning as a deliberate decision. Trajectory steps 64 and 67 discuss the agent's attempts to rebalance the distribution by covering the full 6 m run.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position bins are evaluated only at the retained running frames for each trial, so they are aligned one-for-one with the neural timepoints in `neural_trial`.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
...
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
output_trial[2] = pos_bins
```

iii. The trajectory's core alignment decision was to use the same kept frame indices for all time-varying streams. `CONVERSION_NOTES.md` Step 5 presents `position_bin` as a time-varying output on that axis.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['ft_RunSpeed']`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_RunSpeed` to `running_speed_bin`. The trajectory step 21 identifies running speed as one of the time-varying behavior outputs.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent pools running frames across sessions, optionally subsamples each session to at most 5000 speed samples, computes the 25th/50th/75th percentiles, and then bins each kept frame's speed by those global thresholds.

ii. 
```python
vr_move = beh['ft_move'] > 0
speeds = beh['ft_RunSpeed'][vr_move]
if len(speeds) > 5000:
    speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
...
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 says speed is discretized into global quartiles. Trajectory steps 67-69 explain the subsampling as a performance optimization during quartile estimation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are defined by `np.digitize` against the three quartile boundaries returned by `np.percentile(all_speeds, [25, 50, 75])`, then clipped into category IDs 0-3 and labeled `Q1`-`Q4`.

ii. 
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
...
'output_values': [all_stim_names, ['no_lick', 'lick'], ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ['Q1', 'Q2', 'Q3', 'Q4']],
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 9 both describe quartile-based speed bins. The trajectory shows the agent validating that they were at least approximately above chance in the decoder.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled only at the retained running frames and written onto the same per-trial time axis as the neural data.

ii. 
```python
valid_frame_indices = np.where(valid_mask)[0]
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
output_trial[3] = speed_bins
```

iii. This is part of the agent's consistent "all signals share `valid_frame_indices`" design, described in trajectory steps 24 and 28 and reflected throughout `CONVERSION_NOTES.md` Step 5.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing lick data are treated as no licks by creating an all-zero lick vector when `LickFr` is absent or empty. Missing `SoundFr` values produce an all-zero `time_to_sound` vector. Fractional `SoundFr` values are interpolated using adjacent frame times. Out-of-range lick indices are clipped away. Edge `SoundFr` values are clamped to the first or last frame time. Trials with fewer than two kept running frames are skipped, and sessions with fewer than two kept trials are dropped.

ii. 
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
if len(lick_fr) == 0:
    return lick_binary
...
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
...
elif s_fr_int >= len(ft) - 1:
    sound_time = ft[-1]
else:
    sound_time = ft[0]
...
if len(valid_frame_indices) < 2:
    continue
...
if valid_count < 2:
    return None
```

iii. The trajectory steps 49-50 and 67-69 discuss several of these edge-case decisions explicitly, especially the sound-cue timing and efficiency changes. `CONVERSION_NOTES.md` Step 10 also mentions handling duplicate-session `stimtype` cases by using the first occurrence.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading very large `spk` matrices from disk and then iterating trial-by-trial to slice variable-length frame lists. Computing global speed quartiles also adds overhead, though the agent optimized it to use only behavior data plus per-session subsampling.

ii. 
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
...
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

```python
for i, session in enumerate(sessions):
    ...
    speeds = beh['ft_RunSpeed'][vr_move]
```

iii. `CONVERSION_NOTES.md` Step 6 and Step 9 both emphasize memory pressure and long runtime, and the sample/full run logs show that "load" dominates the reported per-session timing. Trajectory steps 33, 67, and 75 explicitly discuss these bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` could be partially vectorized or restructured by pre-grouping frame indices per trial instead of recomputing a boolean mask for every trial. The region-index assignment loop over four region names is trivial but unnecessary. The speed-quartile loop could also avoid repeated dictionary lookups and list appends.

ii. 
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

```python
region_idx = np.full(n_neurons, 0, dtype=np.int32)
for r_idx, r_name in enumerate(['V1', 'mHV', 'lHV', 'aHV']):
    region_idx[area_map[r_name]] = r_idx
```

iii. The agent notes "vectorize loops" as a goal in the task workflow, but `CONVERSION_NOTES.md` Step 6 shows it mostly optimized via temporary files and lighter quartile estimation instead of deeper vectorization. The trajectory repeatedly focuses on runtime while preserving the same loop structure.

## 12-c. What processing does the code repeat multiple times?

i. It loads behavior dictionaries multiple times: once to enumerate sessions, again to collect all stimulus names, again to compute global speed quartiles, and again inside `process_session`. It also recomputes per-session caches separately in those passes and repeats full-trial mask generation on each trial.

ii. 
```python
all_sessions = get_all_sessions()
...
all_stim_names = get_all_stim_names(all_sessions)
...
speed_quartiles = compute_global_speed_quartiles(all_sessions)
...
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
```

iii. `CONVERSION_NOTES.md` Step 6 mentions efficiency work, but the final design still uses several separate dataset-wide passes. The trajectory shows the agent specifically revisiting the quartile computation because it was an obvious repeated-cost hotspot.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores large per-timepoint copies of trial-constant variables (`day_of_training`, `reward_availability`, and `visual_stimulus_category`) even though they could be stored once per trial. It also writes and rereads temporary per-session pickle files only to combine them later, and in sample mode it still computes global stimulus names and speed quartiles across all sessions. In `--show-processing` mode it additionally renders diagnostic plots that are not used downstream.

ii. 
```python
input_trial[1] = np.float32(day_val)
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
output_trial[0] = stim_idx
```

```python
with open(temp_fn, 'wb') as f:
    pickle.dump({...}, f, protocol=4)
...
for tf in temp_files:
    with open(tf, 'rb') as f:
        sd = pickle.load(f)
```

iii. `CONVERSION_NOTES.md` Step 6 frames temporary files as a memory optimization, but they are still extra I/O that is discarded after assembly. The trajectory also shows the agent knowingly computing global statistics even in sample mode and using plots only for visual inspection.
