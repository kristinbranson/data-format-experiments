# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `data/beh/Imaging_Exp_info.npy` as a master index, builds one unique `session_id` per recording (`mname_datexp_blk`), and keeps only the first occurrence when the same recording appears under multiple experiment types. It then processes sessions one by one: loading spikes from `data/spk`, retinotopy from `data/retinotopy`, and the matching behavior dictionary from the relevant `Beh_<exp_type>.npy`. It also makes separate full-dataset behavior passes to collect stimulus names and compute running-speed quartiles.

ii. ```python
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

iii. In `CONVERSION_NOTES.md`, the agent justified this as “All 89 sessions included: Each unique neural recording = one session,” and the trajectory shows it explicitly decided to use the first experiment-type entry for duplicated recordings.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The final `subjects` list is the order of first appearance while iterating processed sessions, and each session gets a corresponding `subject_idx`.

ii. ```python
sessions.append({'session_id': session_id, 'mname': entry['mname'],
                 'datexp': entry['datexp'], 'blk': entry['blk'],
                 'exp_type': exp_type, 'beh_key': beh_key, 'entry': entry})
```
```python
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The notes say there are 19 subjects and that each unique mouse name defines a subject; no more elaborate subject split was derived.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique recording keyed by `mname`, `datexp`, and `blk`. Duplicate appearances across experiment types are deduplicated by `session_id`.

ii. ```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. The trajectory states that “each unique neural recording (89 total) should be one session,” and the notes repeat that each unique neural recording is treated as a session.

## 1-d. How are the data split into trials?

i. The agent uses `ft_trInd` to assign frames to trials, but only keeps frames where `ft_move > 0`. Each trial is therefore the subset of moving frames with `ft_trInd == trial_idx`; it does not use `ft_CorrSpc`, `StartFr`, or a fixed-length trial window.

ii. ```python
ft_trInd = beh['ft_trInd'][:n_frames_neural]
ft_move = beh['ft_move'][:n_frames_neural]
vr_move = ft_move > 0
```
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    if len(valid_frame_indices) < 2:
        continue
```

iii. The notes justify this with “Only running timepoints used” and “Frame filter: Only VR-moving frames (`ft_move > 0`, matches reference),” reflecting the agent’s belief that the paper’s running-only analyses should define trial content here too.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than 2 kept frames after the moving-frame filter. Sessions are dropped if fewer than 2 trials survive.

ii. ```python
if len(valid_frame_indices) < 2:
    continue
```
```python
valid_count = len(neural_trials)
if valid_count < 2:
    return None
```

iii. The notes claim “All trials included, frames filtered for VR-moving,” but the code adds an extra trial-level filter of at least two retained frames and a session-level minimum of two surviving trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the concatenated `spks` arrays in each session’s `*_neural_data.npy`, with retinotopy `iarea` used to decide which neurons are kept and how regions are indexed.

ii. ```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes identify the source as “Suite2p deconvolved fluorescence traces” and describe retinotopy as the source of brain-region assignment.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, filters neurons by retinotopy, then slices per-trial frame subsets defined by `ft_trInd` and `ft_move > 0`. It stores each trial at its native retained length and casts to `float32`; there is no fixed 32-frame truncation/padding.

ii. ```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
```
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
neural_trials.append(neural_trial)
```

iii. The notes justify this as using “Raw deconvolved traces per frame” and preserving running-only frames, while also emphasizing memory-management choices elsewhere rather than any reference-style fixed-length window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using retinotopy: the code removes `iarea == -1` and `iarea == 7`, then maps the remaining neurons into `V1`, `mHV`, `lHV`, and `aHV`.

ii. ```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
iarea_filtered = iarea[neuron_mask]
```
```python
area_map = neu_area_ID(iarea_filtered)
region_idx = np.full(n_neurons, 0, dtype=np.int32)
for r_idx, r_name in enumerate(['V1', 'mHV', 'lHV', 'aHV']):
    region_idx[area_map[r_name]] = r_idx
```

iii. `CONVERSION_NOTES.md` explicitly says “Neuron filtering: Exclude `iarea==-1` and `iarea==7` (matches reference code)” and treats that as the main neural quality-control step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata and notes say the data are aligned to “Trial start (corridor entry),” but the implementation does not use `StartFr` for neural alignment. In practice, neural trials begin at the first frame in the retained moving-frame subset of each `ft_trInd` trial.

ii. ```python
'time_bin_size': TIME_BIN_MS, 'temporal_alignment_event': 'Trial start (corridor entry)',
```
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes repeatedly state “Temporally aligned based on trial start (corridor entry)” and “Frame filter: Only VR-moving frames,” but the code reflects only the latter directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one imaging frame at 3.17 Hz, stored as `1000 / 3.17` ms. No temporal rebinning or resampling is applied.

ii. ```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```
```python
'time_bin_size': TIME_BIN_MS
```

iii. The notes cite the 3.17 Hz imaging rate from the paper/notebook and treat the raw frame grid as the natural time base.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and frame timestamps `ft`.

ii. ```python
ft = beh['ft'][:n_frames_neural]
sound_fr = beh['SoundFr']
```
```python
s_fr_int = int(np.floor(s_fr))
s_fr_frac = s_fr - s_fr_int
sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
```

iii. The notes describe this input as coming from “SoundFr, ft” and computed using actual frame times in MATLAB datenum units converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound frame is linearly interpolated between neighboring frame times using its fractional frame index. The resulting signal is `ft_trial - sound_time`, so values are negative before the cue and positive after it. If `SoundFr` is `NaN`, the code fills the trial with zeros.

ii. ```python
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
else:
    s_fr_int = int(np.floor(s_fr))
    s_fr_frac = s_fr - s_fr_int
    ...
    time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The trajectory shows the agent deciding to use “actual frame times (MATLAB datenum)” for time variables, and the notes summarize this variable as “Time from frame to sound cue (seconds).”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same retained frame indices used for each neural trial.

ii. ```python
ft_trial = ft[valid_frame_indices]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
...
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The agent’s justification throughout the notes is that all time-varying streams should stay on the neural frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata, specifically `mname` and the date string `datexp`.

ii. ```python
mouse_sessions[s['mname']].append((i, s))
dates = [(idx, datetime(int(s['datexp'].split('_')[0]),
                        int(s['datexp'].split('_')[1]),
                        int(s['datexp'].split('_')[2]))) for idx, s in msessions]
```

iii. The notes map `datexp` to `day_of_training` and describe it as “Days since first session per mouse.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the agent sorts sessions by calendar date and computes the number of elapsed days since that mouse’s first session. That scalar is then broadcast across every time bin of each trial.

ii. ```python
dates.sort(key=lambda x: x[1])
first_date = dates[0][1]
for idx, date in dates:
    day_of_training[idx] = (date - first_date).days
```
```python
input_trial[1] = np.float32(day_val)
```

iii. The notes justify this as “Days since first session per mouse,” not as ordinal session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the implementation, it is derived only from `ft` after trial/frame selection; it does not use `StartFr`.

ii. ```python
ft = beh['ft'][:n_frames_neural]
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes originally mapped `ft` to this variable and described it as elapsed time from trial start, but the code operationalizes “trial start” as the first retained frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent subtracts the timestamp of the first retained frame of that trial from each retained frame timestamp and converts the result to seconds.

ii. ```python
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The trajectory shows the agent deciding to use actual frame timestamps for all time variables; here it applied that rule to the kept frame subset rather than to `StartFr`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same `valid_frame_indices` used for the neural trial matrix.

ii. ```python
ft_trial = ft[valid_frame_indices]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes emphasize frame-level alignment across streams, and the code follows that within its chosen trial window.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` flag.

ii. ```python
is_rew = beh['isRew']
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The notes map `isRew` directly to reward availability with no extra transformation beyond binary coding.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No complex processing is applied. The code converts the boolean/reward flag to `1.0` or `0.0` and broadcasts it across all time bins in the trial.

ii. ```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The notes describe this as a direct mapping: “1=rewarded, 0=not.”

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial wall identity, using `WallName`; the global label set is collected from `UniqWalls`.

ii. ```python
for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(wn))
```
```python
wall_name = beh['WallName']
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. The notes say the source is `WallName`, and later treat the problem as a 15-category texture-classification task.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent enumerates all unique wall names across the dataset, sorts them, and uses that 15-label set directly. Each trial’s `WallName` is mapped to an integer index and broadcast across the whole trial.

ii. ```python
all_stim_names = get_all_stim_names(all_sessions)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```
```python
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
output_trial[0] = stim_idx
```

iii. The notes explicitly evaluate the decoder as having “15 categories,” showing that the agent intentionally kept the fine-grained wall identities instead of collapsing them to four texture classes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. ```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```

iii. The notes identify `LickFr` as the source and describe it as frame indices for lick events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code constructs a binary per-frame lick vector. It rounds each lick frame to the nearest integer frame, discards indices outside the imaged range, and marks those frames with 1; retained trial frames then inherit those binary values.

ii. ```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
```
```python
lick_trial = lick_binary[valid_frame_indices]
output_trial[1] = lick_trial
```

iii. The trajectory shows the agent explicitly checking that `LickFr` existed and deciding to “construct licking as a binary time-varying” output from those frame indices.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by indexing the same retained trial frames used for the neural data.

ii. ```python
lick_trial = lick_binary[valid_frame_indices]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes frame this as keeping all behavioral streams on the frame grid of the neural recording.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. ```python
ft_Pos = beh['ft_Pos'][:n_frames_neural]
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The notes map `ft_Pos` directly to position bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent uses the full 6 m corridor and divides position by 15 decimeters, producing four 1.5 m bins over 0-6 m. It computes the bin only on retained moving frames.

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
output_trial[2] = pos_bins
```

iii. The notes justify this with “4 equal bins of 15dm over full 6m corridor.”

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are implicit at 0, 15, 30, and 45 decimeters via `floor(ft_Pos / 15.0)`, then clipped to category indices 0-3.

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```
```python
'output_values': [all_stim_names, ['no_lick', 'lick'],
                  ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'],
                  ['Q1', 'Q2', 'Q3', 'Q4']]
```

iii. The notes explicitly describe the same 15 dm binning choice.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by selecting the same `valid_frame_indices` used for the neural matrix.

ii. ```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes justify this as frame-wise alignment of behavioral and neural streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The notes map `ft_RunSpeed` to a 4-bin running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first computes global running-speed quartile thresholds across all sessions, using only frames where `ft_move > 0` and subsampling up to 5000 speeds per session. It then digitizes each retained trial frame using those shared thresholds.

ii. ```python
vr_move = beh['ft_move'] > 0
speeds = beh['ft_RunSpeed'][vr_move]
if len(speeds) > 5000:
    speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The notes justify this as “Global quartile-based (computed from all sessions),” and the trajectory shows the agent explicitly deciding that quartiles should be computed from all sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are set by three global percentile thresholds returned by `np.percentile(all_speeds, [25, 50, 75])`; `np.digitize` maps speeds into `Q1`-`Q4`.

ii. ```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```
```python
'output_values': [all_stim_names, ['no_lick', 'lick'],
                  ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'],
                  ['Q1', 'Q2', 'Q3', 'Q4']]
```

iii. The notes explicitly describe the thresholds as global quartiles rather than session-wise bins or rank-based quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by indexing the same retained frame subset used for each neural trial.

ii. ```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes justify this with the same frame-grid alignment used for other time-varying outputs.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates all frame-level behavior arrays to the number of neural frames, defaults missing `LickFr` to an empty array, drops out-of-range lick indices, substitutes zeros when `SoundFr` is `NaN`, clamps out-of-range sound-frame interpolation to the first or last frame, and skips trials with too few retained frames.

ii. ```python
ft = beh['ft'][:n_frames_neural]
ft_trInd = beh['ft_trInd'][:n_frames_neural]
ft_Pos = beh['ft_Pos'][:n_frames_neural]
ft_move = beh['ft_move'][:n_frames_neural]
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
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
```python
if len(valid_frame_indices) < 2:
    continue
```

iii. The notes mostly portray the dataset as clean, but the code includes several defensive fallbacks for missing or out-of-range event timestamps.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session-wise loading/filtering of the large spike arrays, the extra full-dataset pass to compute global speed quartiles, and the temporary-pickle write/read cycle used to stage sessions before final assembly.

ii. ```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```
```python
speed_quartiles = compute_global_speed_quartiles(all_sessions)
```
```python
with open(temp_fn, 'wb') as f:
    pickle.dump({...}, f, protocol=4)
...
with open(tf, 'rb') as f:
    sd = pickle.load(f)
```

iii. The notes mention large file sizes, long runtime, and temp-file caching as explicit implementation choices to manage memory while processing the full dataset.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly scans full frame-length arrays with `(ft_trInd == trial_idx) & vr_move` and `np.where`, which could have been replaced with a single pre-grouping of frame indices by trial. The per-session subject-index lookup with `list.index` is also avoidable.

ii. ```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```
```python
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The trajectory shows the agent was aware of memory pressure, but not of this repeated masking cost; the notes do not discuss vectorization here.

## 12-c. What processing does the code repeat multiple times?

i. It rereads behavior dictionaries multiple times for different passes: once to enumerate sessions, again to collect all stimulus names, again to compute global speed quartiles, and then once per session during actual conversion. It also writes every session to a temp file and immediately rereads it later.

ii. ```python
all_stim_names = get_all_stim_names(all_sessions)
speed_quartiles = compute_global_speed_quartiles(all_sessions)
...
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
```
```python
with open(temp_fn, 'wb') as f:
    pickle.dump({...}, f, protocol=4)
...
with open(tf, 'rb') as f:
    sd = pickle.load(f)
```

iii. The notes explicitly justify the temp-file cycle as a memory optimization, but they do not acknowledge the repeated behavior loading and preprocessing passes it also introduces.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores a 15-class wall-label vocabulary instead of the four texture categories used by the reference, creates temp cache files that are only intermediate transport, computes a `session_meta` list that is never used, and imports/modules (`sys`, `Counter`, `tempfile`) that are unused.

ii. ```python
all_stim_names = get_all_stim_names(all_sessions)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```
```python
session_meta = []
...
temp_fn = f'cache/session_{i:03d}.pkl'
...
for tf in temp_files:
    os.remove(tf)
```

iii. The notes defend the temp-file route as a practical memory workaround, but downstream analyses only consume the final assembled pickle, not these intermediates or the unused bookkeeping.
