# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, deduplicates recordings by the physical session key `mname_datexp_blk`, then processes each unique session separately. For each session it looks up behavior by scanning experiment types until it finds a matching behavior entry, then loads spikes from `spk/<session>_neural_data.npy` and retinotopy from `retinotopy/<mouse>_<date>_trans.npz`.

ii. 
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_sessions = get_all_unique_sessions(exp_info)
```
```python
def get_session_beh(session_key, exp_info):
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname and s['datexp'] == datexp and s['blk'] == blk:
                beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
                beh_all = np.load(beh_path, allow_pickle=True).item()
                ...
                return beh_all[key_fmt], exp_type, s
```
```python
spk = load_spk(mname, datexp, blk)
iarea = load_retino(mname, datexp)
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset has 89 unique physical recordings, that each recording should be included once, and that it will use the first available experiment type for the behavioral data.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The final `subjects` list is created in first-seen order while iterating sessions, and `subject_idx` stores the index of each session’s mouse in that list.

ii. 
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
...
'mname': s['mname'],
```
```python
subjects_seen = {}
...
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subj_idx = subjects_seen[mname]
...
subjects = list(subjects_seen.keys())
subject_idx = np.array(subject_idx_list, dtype=np.int64)
```

iii. The notes repeatedly identify the mouse id as the subject split and report 19 unique mice.

## 1-c. How are the data split into sessions?

i. A session is one unique `mname_datexp_blk` recording. The AI builds that key from the master index and keeps only the first occurrence of each key across experiment types.

ii. 
```python
seen = set()
...
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in seen:
    seen.add(key)
    sessions.append({
        'key': key,
        'mname': s['mname'],
        'datexp': s['datexp'],
        'blk': s['blk'],
        'exp_type': exp_type,
        'db': s,
    })
```

iii. The notes justify this as deduplicating repeated listings of the same physical recording while preserving one session per real recording, matching the paper’s 89 recordings.

## 1-d. How are the data split into trials?

i. Trials are split by `ft_trInd`, but the AI keeps only frames that are simultaneously in the corridor (`ft_CorrSpc`) and running (`ft_move > 0`). Each trial is therefore represented as the variable-length sequence of in-corridor running frames for one trial index.

ii. 
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
    frame_indices = np.where(trial_mask)[0]
```

iii. The notes explicitly say the conversion uses “running-only frames” and “corridor-only frames,” citing the paper’s analyses of running periods.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops any trial with fewer than 2 kept frames after its running-and-corridor mask. It also skips an entire session if fewer than 2 trials survive. It does not implement the reference solution’s long-trial outlier filter.

ii. 
```python
if len(frame_indices) < 2:
    continue
```
```python
if len(neural_trials) < 2:
    print(f"  WARNING: {key} has {len(neural_trials)} valid trials, skipping")
    return None
```

iii. The notes say “Trials with 0 valid frames after filtering should be excluded” and reflect a decoder-driven requirement that sessions must retain enough trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the deconvolved calcium traces in `spks` and from `iarea` in the retinotopy file for neuron inclusion and brain-region assignment.

ii. 
```python
dat = np.load(fn, allow_pickle=True).item()
spk = np.concatenate(dat['spks'], axis=0)
```
```python
dat = np.load(fn, allow_pickle=True)
return dat['iarea']
```

iii. The notes state that the source neural data are the per-plane `spks` arrays and that retinotopy is used to map neurons into visual regions.

## 2-b. How is the `neural` data processed?

i. The AI concatenates imaging planes, optionally truncates spikes/retinotopy to the smaller length if they disagree, filters neurons, selects the trial’s kept frames, and stores the result as `float32`. It does not compute dF/F or deconvolution.

ii. 
```python
spk = np.concatenate(dat['spks'], axis=0)
...
if len(iarea) != n_neurons_raw:
    min_n = min(n_neurons_raw, len(iarea))
    spk = spk[:min_n]
    iarea = iarea[:min_n]
```
```python
spk = spk[neuron_mask]
...
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes say the paper’s analyses use deconvolved traces directly and that no additional neural preprocessing is needed beyond region filtering and frame selection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by retinotopy/area assignment: neurons with `iarea == -1` or `iarea == 7` are excluded. Surviving neurons are mapped to one of `V1`, `mHV`, `lHV`, `aHV`.

ii. 
```python
EXCLUDED_IAREA = {-1, 7}
...
mask = np.ones(len(iarea), dtype=bool)
for exc in EXCLUDED_IAREA:
    mask &= (iarea != exc)
```
```python
region_name = BRAIN_REGION_MAP.get(int(ia), None)
if region_name is not None:
    region_idx[i] = BRAIN_REGIONS.index(region_name)
```

iii. The notes justify this from the reference code’s visual-cortex filter and explicitly say there is no extra firing-rate or quality cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. In the implemented code, neural data are aligned to the first kept running frame inside the texture corridor, not to all corridor-entry-aligned frames. Trial arrays begin at the first frame satisfying the running-and-corridor mask, and trial lengths remain variable.

ii. 
```python
trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
frame_indices = np.where(trial_mask)[0]
...
neural = spk[:, frame_indices].astype(np.float32)
```
```python
'temporal_alignment_event': 'Trial start (corridor entry, first running frame in texture corridor)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes and metadata frame this as following running-only analysis periods from the paper; the metadata explicitly labels the alignment as the first running frame in the texture corridor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging frame resolution and does not apply temporal rebinning. It stores metadata `time_bin_size` as 315.0 ms.

ii. 
```python
ft = beh['ft'][:nfr]
if frame_dt_s is None:
    frame_dt_s = float(np.median(np.diff(ft)) * 86400)
```
```python
'time_bin_size': 315.0,
'frame_rate_hz': 3.178,
```

iii. The notes say the native imaging rate is about 3.178 Hz and that no interpolation or rebinning is needed for the decoder format.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `ft`.

ii. 
```python
ft = beh['ft'][:nfr]
SoundFr = beh['SoundFr']
```
```python
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
```

iii. The notes map `SoundFr, ft` directly to `time_to_sound_cue`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI interpolates each trial’s `SoundFr` onto the session frame-time axis, converts MATLAB-day timestamps to seconds, and computes `sound_time_s - frame_times`, so the value is positive before the cue and negative after it.

ii. 
```python
frame_times = ft[frame_indices] * 86400
sound_fr = SoundFr[trial_idx]
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
time_to_sound = sound_time_s - frame_times
```

iii. The notes explicitly justify the use of actual timestamps and the positive-before-cue sign convention.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the same `frame_indices` array used to build the neural matrix for each trial, so it is aligned to the kept running frames rather than the full corridor-entry-to-exit frame sequence.

ii. 
```python
frame_indices = np.where(trial_mask)[0]
...
frame_times = ft[frame_indices] * 86400
...
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes treat all decoder variables as living on the same kept frame grid as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` together with the mouse name `mname`.

ii. 
```python
date = datetime.strptime(s['datexp'], '%Y_%m_%d')
mouse_dates[mname].append((date, s['key']))
```

iii. The notes state that this variable is computed from session dates for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the AI sorts sessions by calendar date and assigns `day_of_training` as the number of elapsed days since that mouse’s first recorded session.

ii. 
```python
for mname, dates in mouse_dates.items():
    dates.sort(key=lambda x: x[0])
    first_date = dates[0][0]
    for date, key in dates:
        day_map[key] = (date - first_date).days
```
```python
np.full((1, n_tp), day_of_training, dtype=np.float32)
```

iii. In the notes, the AI explicitly lists “Days since first session for each mouse” as the intended definition.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the implemented code it is derived from `ft` and the first kept frame in `frame_indices`; `StartFr` is not used.

ii. 
```python
ft = beh['ft'][:nfr]
frame_times = ft[frame_indices] * 86400
time_since_start = frame_times - frame_times[0]
```

iii. The notes describe this as seconds from corridor entry, but the actual code implements it from the first kept running frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI converts kept frame timestamps to seconds and subtracts the time of the trial’s first kept frame, making the trace start at 0 at that frame.

ii. 
```python
frame_times = ft[frame_indices] * 86400
time_since_start = frame_times - frame_times[0]
```

iii. The notes justify using actual timestamps, but the code does not interpolate `StartFr`; it anchors the variable to the first retained frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned to the same `frame_indices` as the neural data and therefore starts at the first kept running frame for that trial.

ii. 
```python
frame_indices = np.where(trial_mask)[0]
...
time_since_start = frame_times - frame_times[0]
...
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes treat the neural data and all input/output streams as sharing the same retained frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`.

ii. 
```python
isRew = beh['isRew']
```

iii. The notes map `isRew` directly to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is no real transformation beyond converting the per-trial value to float and broadcasting it across all time bins of the trial.

ii. 
```python
np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32)
```

iii. The notes describe reward availability as a binary per-trial flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. 
```python
WallName = beh['WallName']
wall_name = str(WallName[trial_idx])
```

iii. The notes explicitly map `WallName` to the visual stimulus category output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps detailed wall-texture labels to four broad categories (`circle`, `leaf`, `rock`, `wood`), converts the category to its index in `STIM_CATEGORIES`, and broadcasts that integer across the whole trial.

ii. 
```python
for s in ['circle1', 'circle2', 'circle3']:
    STIM_CATEGORY_MAP[s] = 'circle'
...
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'wood']
```
```python
stim_cat = STIM_CATEGORY_MAP.get(wall_name, None)
stim_cat_idx = STIM_CATEGORIES.index(stim_cat)
...
np.full((1, n_tp), stim_cat_idx, dtype=np.int64)
```

iii. The notes justify using broad category labels because the task asks for categories like “circle, leaf, etc.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii. 
```python
LickFr = beh['LickFr']
LickTrind = beh['LickTrind']
```

iii. The notes say licking should be a binary time series derived from lick events, and the code uses `LickTrind` to isolate a trial’s lick events before binning them.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI selects that trial’s lick frames, finds the nearest retained neural frame for each lick using `searchsorted`, and sets those bins to 1 in a binary vector. This is a nearest-neighbor snap to kept running frames, not a simple global frame flag.

ii. 
```python
trial_lick_mask = (LickTrind == trial_idx)
trial_lick_frs = LickFr[trial_lick_mask]
lick_binary = np.zeros(n_tp, dtype=np.float32)
```
```python
fi_sorted = frame_indices.astype(float)
lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
for li in range(len(trial_lick_frs)):
    idx = lick_bin_idx[li]
    if idx >= n_tp:
        idx = n_tp - 1
    elif idx > 0:
        if abs(trial_lick_frs[li] - fi_sorted[idx-1]) < abs(trial_lick_frs[li] - fi_sorted[idx]):
            idx = idx - 1
    lick_binary[idx] = 1.0
```

iii. The trajectory and notes indicate the AI wanted a binary time-varying licking output aligned to its retained frame grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to the same retained `frame_indices` used for neural data by snapping each lick to the nearest kept frame within the trial.

ii. 
```python
frame_indices = np.where(trial_mask)[0]
...
fi_sorted = frame_indices.astype(float)
...
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI’s stated goal was to align all outputs to the same running-frame grid as the neural arrays.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. 
```python
ft_Pos = beh['ft_Pos'][:nfr]
positions = ft_Pos[frame_indices]
```

iii. The notes map `ft_Pos` directly to the position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips positions into the 4 m texture corridor and bins them with edges at 10, 20, 30, and 40 decimeters.

ii. 
```python
positions = np.clip(positions, 0, CORRIDOR_LENGTH_DM - 0.001)
pos_bin = np.digitize(positions, POSITION_BIN_EDGES_DM[1:])
pos_bin = np.clip(pos_bin, 0, N_POSITION_BINS - 1)
```

iii. The notes justify this from the decoder specification requiring four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Positions are thresholded into four categories: `[0,10)`, `[10,20)`, `[20,30)`, `[30,40)` decimeters, labeled as `0-1m`, `1-2m`, `2-3m`, `3-4m`.

ii. 
```python
POSITION_BIN_EDGES_DM = np.array([0, 10, 20, 30, 40])
N_POSITION_BINS = 4
```
```python
'output_values': [
    STIM_CATEGORIES,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    speed_labels,
],
```

iii. The notes explicitly say the decoder task requires 4 bins of 1 m each.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained `frame_indices` used for the neural matrix, so it is aligned to the running-only trial representation.

ii. 
```python
positions = ft_Pos[frame_indices]
...
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes say all converted streams should be on the same kept frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
...
speeds = ft_RunSpeed[frame_indices]
```

iii. The notes map `ft_RunSpeed` directly to the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes global speed quartile edges across all sessions (or the sample subset), using only corridor-and-running frames and subsampling sessions with more than 10,000 speed values. Each trial’s retained frame speeds are then binned with those fixed global edges.

ii. 
```python
mask = ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
speeds = ft_RunSpeed[mask]
if len(speeds) > 10000:
    rng = np.random.default_rng(42)
    speeds = rng.choice(speeds, 10000, replace=False)
...
edges = np.percentile(all_speeds, [0, 25, 50, 75, 100])
```
```python
speeds = ft_RunSpeed[frame_indices]
speed_bin = np.digitize(speeds, speed_edges[1:-1])
```

iii. The notes justify this as enforcing consistent quartile thresholds across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded by `np.digitize` against the global percentile edges returned by `compute_speed_quartile_edges`, producing categories 0 through 3.

ii. 
```python
speed_edges = compute_speed_quartile_edges(speed_sessions, exp_info)
...
speed_bin = np.digitize(speeds, speed_edges[1:-1])
speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```

iii. The notes describe these bins as quartiles computed across all valid frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained `frame_indices` used for the neural data, so it shares the running-only trial grid.

ii. 
```python
speeds = ft_RunSpeed[frame_indices]
...
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes treat speed as another time-varying signal aligned to the same kept frame sequence.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several contingencies: it truncates behavior to `min(n_frames, nfr_beh)`, truncates spikes/retinotopy to the smaller length if neuron counts disagree, skips sessions with missing behavior, skips trials with unknown stimulus labels, skips trials with fewer than 2 kept frames, and skips sessions with fewer than 2 valid trials. For licking, licks past the last retained frame are snapped to the last retained bin rather than discarded.

ii. 
```python
nfr = min(n_frames, nfr_beh)
...
if len(iarea) != n_neurons_raw:
    min_n = min(n_neurons_raw, len(iarea))
    spk = spk[:min_n]
    iarea = iarea[:min_n]
```
```python
if beh is None:
    print(f"  WARNING: No behavioral data found for {key}, skipping")
    return None
...
if stim_cat is None:
    print(f"  WARNING: Unknown stimulus '{wall_name}' in trial {trial_idx}, skipping")
    continue
```
```python
if len(frame_indices) < 2:
    continue
...
if idx >= n_tp:
    idx = n_tp - 1
```

iii. The notes describe the dataset as “clean” and present these as defensive fallbacks rather than paper-driven processing rules.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are per-session spike-file loading/concatenation and the extra whole-dataset pass used to compute global running-speed quartile edges, which repeatedly scans behavior files before the main conversion pass. Optional plotting adds more repeated behavior loads.

ii. 
```python
spk = load_spk(mname, datexp, blk)
...
dat = np.load(fn, allow_pickle=True).item()
spk = np.concatenate(dat['spks'], axis=0)
```
```python
def compute_speed_quartile_edges(sessions, exp_info):
    for s in sessions:
        beh, _, _ = get_session_beh(s['key'], exp_info)
        ...
        all_speeds.append(speeds)
```

iii. The code prints timing around loading/filtering/processing, and the design itself implies extra I/O from the global speed-edge prepass.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the per-neuron region-mapping loop, the per-lick nearest-frame loop, and the repeated nested scan through `exp_info` in `get_session_beh`. Trial splitting also rebuilds a boolean mask for every trial.

ii. 
```python
for i, ia in enumerate(filtered_iarea):
    region_name = BRAIN_REGION_MAP.get(int(ia), None)
    ...
```
```python
for li in range(len(trial_lick_frs)):
    idx = lick_bin_idx[li]
    ...
    lick_binary[idx] = 1.0
```
```python
for exp_type, sessions in exp_info.items():
    for s in sessions:
        if s['mname'] == mname and s['datexp'] == datexp and s['blk'] == blk:
            ...
```

iii. The instructions explicitly asked for vectorization and reduced I/O, but the code keeps several Python-level loops.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly resolves behavior for a session by rescanning `exp_info` and reloading behavior files. That happens once in the global speed-edge pass, again during per-session processing, and again in `plot_processing` when plotting is enabled.

ii. 
```python
beh, _, _ = get_session_beh(s['key'], exp_info)
```
```python
beh, exp_type, db_entry = get_session_beh(key, exp_info)
```
```python
beh, _, _ = get_session_beh(key, exp_info)
```

iii. The notes say the script “follows reference code conventions,” but unlike the reference solution it does not group sessions by behavior file to avoid rereading them.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several values that are not used downstream: `frame_dt_s`, `reward_avail`, `valid_trial_indices`, `all_subjects`, `TOTAL_CORRIDOR_DM`, and some intermediate metadata-only display values. The optional plotting path also reloads behavior solely for visualization rather than for the saved dataset.

ii. 
```python
TOTAL_CORRIDOR_DM = 60
...
if frame_dt_s is None:
    frame_dt_s = float(np.median(np.diff(ft)) * 86400)
...
reward_avail = np.full(1, float(isRew[trial_idx]), dtype=np.float32)
...
valid_trial_indices.append(trial_idx)
```
```python
all_subjects = []
...
def plot_processing(...):
    beh, _, _ = get_session_beh(key, exp_info)
```

iii. None of these are described in the notes as analytically necessary; they are byproducts of implementation or optional diagnostics.
