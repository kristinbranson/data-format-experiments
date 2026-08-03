# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master index from `beh/Imaging_Exp_info.npy`, then eagerly loads each `Beh_<exp_type>.npy` while collecting sessions. It stores the selected behavior record directly in each session dict, and later loads the spike file and retinotopy file separately for each session inside `process_session()`.

Unlike the reference, it sorts experiment types with a custom priority (`test` before non-`test`, `sup_` before others, `after` before `before`) before deduplicating sessions. That means duplicate recordings are not resolved by the original file order.

ii.

```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
```

```python
exp_order = sorted(exp_info.keys(), key=lambda x: (
    0 if 'test' in x else 1,
    0 if 'sup_' in x else 1,
    0 if 'after' in x else 1,
    x
))
...
beh_data = np.load(beh_path, allow_pickle=True).item()
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
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
```

iii. `CONVERSION_NOTES.md` says the agent used the 23 experiment types in `Imaging_Exp_info.npy`, resolved overlapping recordings by a preferred ordering, and believed this would choose “more complete annotations”.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname`. The script gathers the sorted unique mouse names across all selected sessions and later records each session's subject index with `all_subjects.index(result['mname'])`.

ii.

```python
all_subjects = sorted(set(s['mname'] for s in sessions))
```

```python
all_subject_idx.append(all_subjects.index(result['mname']))
```

iii. The justification is implicit in the code. `CONVERSION_NOTES.md` also describes unique recordings as identified by `(mname, datexp, blk)`, so `mname` is the mouse identifier.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` tuple. During collection the agent tracks `seen_recordings` and keeps only the first record encountered under its custom experiment-type sort order.

ii.

```python
rec_key = (ndb['mname'], ndb['datexp'], ndb['blk'])
if rec_key in seen_recordings:
    continue
seen_recordings.add(rec_key)
```

iii. `CONVERSION_NOTES.md` explicitly says unique recordings are identified by `(mname, datexp, blk)` and that when a recording appears in multiple experiment types, the first match is used after applying the custom preference ordering.

## 1-d. How are the data split into trials?

i. The agent does not derive trial frame windows from `ft_trInd` and `ft_CorrSpc`. Instead, it trusts `beh['ntrials']`, position-interpolates neural activity across the full recording into `(n_trials, 60)` bins, then treats each trial as the first 40 texture bins of that interpolated array.

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
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

```python
for tr in range(n_trials):
    neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The agent's notes say it was following the paper's position-interpolated representation with 60 bins per corridor and keeping only the 40 texture bins.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial quality filter. The agent keeps all `n_trials` from behavior, and only applies a session-level minimum of at least 2 trials.

ii.

```python
n_trials = beh['ntrials']
...
for tr in range(n_trials):
    ...
```

```python
if result['n_trials'] < 2:
    print(f"  Skipping: only {result['n_trials']} trials")
    continue
```

iii. No explicit trial-filter justification is given beyond the general claim in `CONVERSION_NOTES.md` that all 89 recordings had usable data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from concatenating `spk_data['spks']` across imaging planes, with neuron selection based on retinotopy `iarea`.

ii.

```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

```python
ret_data = np.load(os.path.join(root, 'retinotopy', ret_fn), allow_pickle=True)
iarea = ret_data['iarea']
```

iii. `CONVERSION_NOTES.md` says the neural stream is based on deconvolved calcium traces and that neuron filtering uses the retinotopy visual-area assignments.

## 2-b. How is the `neural` data processed?

i. The agent filters to running frames (`ft_move > 0`), uses cumulative position `ft_PosCum` to linearly interpolate the neural traces into 60 evenly spaced corridor bins, then truncates to the first 40 bins (the 4 m textured segment). Trials are stored as `float32`, not padded frame windows.

ii.

```python
ft_move = beh['ft_move'][:n_frames]
vr_moving = ft_move > 0
ft_pos_cum = beh['ft_PosCum'][:n_frames]
```

```python
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
neural_trial = interp_spk[:, tr, :].astype(np.float32)
```

iii. The notes justify this as matching `spk_pos_interp()` / `get_interpPos_spk()` from the paper code, restricting to running frames and the texture area, and using `numpy.interp` for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps neurons with `iarea != -1` and `iarea != 7`, then maps retained neurons into four visual-region categories (`V1`, `mHV`, `lHV`, `aHV`). There is no additional neural quality filtering.

ii.

```python
valid_mask = (iarea != -1) & (iarea != 7)
region_idx = np.full(len(iarea), -1, dtype=int)
region_idx[iarea == 8] = 0
region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
region_idx[(iarea == 5) | (iarea == 6)] = 2
region_idx[(iarea == 3) | (iarea == 4)] = 3
```

iii. `CONVERSION_NOTES.md` says this matches `load_retino()` / `neu_area_ID()` and excludes only non-visual cortex (`-1`) and undefined region `7`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata says alignment is to trial start (corridor entry), but the actual neural representation is position-aligned: each trial is represented by 40 fixed 1-dm bins covering the textured corridor.

ii.

```python
N_TEXTURE_BINS = 40
VR_SPEED = 6.0
BIN_SIZE_SEC = 1.0 / VR_SPEED
```

```python
interp_spk = position_interpolate_spk(..., n_trials, n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

```python
'temporal_alignment_event': 'Trial start (corridor entry)',
```

iii. The notes justify this by claiming the bins correspond to the 4 m corridor after corridor entry, with constant-speed VR making each 1 dm spatial bin correspond to a fixed time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 166.67 ms bins, computed from a fixed VR speed of 6 dm/s and 1 dm position bins. Yes, temporal/spatial rebinning is applied: neural data are interpolated from frame times into corridor position bins.

ii.

```python
VR_SPEED = 6.0
BIN_SIZE_SEC = 1.0 / VR_SPEED
BIN_SIZE_MS = BIN_SIZE_SEC * 1000
```

```python
'time_bin_size': BIN_SIZE_MS,
```

```python
interp_spk = position_interpolate_spk(...)
```

iii. `CONVERSION_NOTES.md` explicitly states a 166.67 ms bin size and 40 timepoints per trial because it treats 1 dm of corridor as one bin at 60 cm/s.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent derives `time_to_sound_cue` from `beh['SoundPos']`, not from `SoundFr` and frame timestamps.

ii.

```python
sound_pos = beh['SoundPos']
```

iii. `CONVERSION_NOTES.md` says the sound cue is represented as its position in the corridor and converted into time using the fixed-speed assumption.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the agent creates a 40-bin vector and computes `(position_bin_index - sound_pos[trial]) * BIN_SIZE_SEC`. This makes the feature linear in corridor position rather than in recorded frame times.

ii.

```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The notes justify this as “negative before cue, positive after” under constant 60 cm/s VR speed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same 40 position bins as the position-interpolated neural data for each trial.

ii.

```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
...
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
```

iii. The notes state that the cue position and neural activity share the same corridor-bin representation.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` for each mouse.

ii.

```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
```

```python
d = compute_day_of_training(s['datexp'])
```

iii. `CONVERSION_NOTES.md` says day of training is “days since first recording for each mouse”.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script finds the earliest session date for each mouse, computes elapsed calendar days since that first recording, stores the result as a float, and then broadcasts it across all 40 bins of every trial in the session.

ii.

```python
mouse_first_date = {}
for s in sessions:
    d = compute_day_of_training(s['datexp'])
    if s['mname'] not in mouse_first_date or d < mouse_first_date[s['mname']]:
        mouse_first_date[s['mname']] = d
```

```python
days = (d - mouse_first_date[s['mname']]).days
session_days[key] = float(days)
```

```python
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. The notes explicitly justify this as “Days since first recording for each mouse.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from a raw timestamp variable like `StartFr` or `ft`. Instead it is synthesized from the fixed bin index `i` and the constant `BIN_SIZE_SEC`.

ii.

```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` describes it as a linear ramp from 0 to about 6.5 s across the 40 corridor bins.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent constructs a deterministic linear vector `[0, 1, ..., 39] * BIN_SIZE_SEC` for every trial, assuming constant VR speed and one fixed time per 1-dm spatial bin.

ii.

```python
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The justification in the notes is the constant-speed VR interpretation: 40 bins across 4 m gives a roughly 6.5 s trial segment.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction to the same 40 position bins used for the neural trial matrix.

ii.

```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
...
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. The notes present both neural data and this input as living on the same corridor-bin axis.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `beh['isRew']`.

ii.

```python
is_rew = beh['isRew'].astype(float)
```

iii. `CONVERSION_NOTES.md` states reward availability is taken from `beh['isRew']` and interpreted as whether the corridor is rewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is essentially no transformation besides casting to float and broadcasting the trial-level flag across all 40 bins.

ii.

```python
reward_avail = np.full(N_TEXTURE_BINS, is_rew[tr], dtype=np.float32)
```

iii. The notes justify this as a binary per-trial variable: 1 in rewarded corridors and 0 otherwise.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output is derived from `beh['WallName']`, with the list of possible stimulus labels collected globally from `beh['UniqWalls']` across sessions.

ii.

```python
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
all_stim_names = sorted(all_stim_set)
```

```python
wall_names = beh['WallName']
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
```

iii. The notes say the agent wanted consistent labeling across all sessions and kept all 15 distinct stimulus names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent keeps the 15 distinct wall names as 15 separate categories, maps each `WallName` to its index in the global sorted list `all_stim_names`, and broadcasts that category across all 40 bins of the trial.

ii.

```python
all_stim_names = sorted(all_stim_set)
```

```python
stim_indices = np.array([all_stim_names.index(wn) for wn in wall_names])
...
stim_cat = np.full(N_TEXTURE_BINS, stim_indices[tr], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly says visual stimulus is represented as 15 unique categories rather than collapsed texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The agent derives licking from `beh['LickPos']` and `beh['LickTrind']`, not from `LickFr`.

ii.

```python
lick_pos = beh['LickPos']
lick_trind = beh['LickTrind']
```

iii. The notes explicitly describe licking as position-binned lick events using `LickPos` and `LickTrind`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script initializes a `(n_trials, 40)` binary matrix and, for each lick event, sets the corresponding trial/bin entry to 1 using `int(pos)` as the bin index.

ii.

```python
def lick_to_position_bins(lick_pos, lick_trind, n_trials, n_bins=40):
    lick_binary = np.zeros((n_trials, n_bins), dtype=int)
    for i in range(len(lick_pos)):
        tr = int(lick_trind[i])
        pos = lick_pos[i]
        bin_idx = int(pos)
        if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
            lick_binary[tr, bin_idx] = 1
    return lick_binary
```

iii. The notes justify this as a binary “1 if any lick event occurred in that position bin, 0 otherwise.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned to the same 40 position bins and trial indices as the interpolated neural data.

ii.

```python
lick_binary = lick_to_position_bins(lick_pos, lick_trind, n_trials, N_TEXTURE_BINS)
...
lick_tr = lick_binary[tr, :].astype(np.int64)
```

iii. The notes say both are represented on the same position-binned corridor axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is not read from `ft_Pos`. Instead, the script constructs a fixed 40-bin corridor template and derives the 4 categories from the bin index itself.

ii.

```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The notes justify this by treating each trial as 40 one-decimeter bins spanning the 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent creates a deterministic 40-element vector with categories 0,1,2,3 repeated over 10 bins each, and uses the same vector for every trial.

ii.

```python
pos_bins = np.zeros(N_TEXTURE_BINS, dtype=int)
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
...
pos_tr = pos_bins.astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says the 40 bins cover the 4 m corridor and each 10-bin block corresponds to a 1 m category.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholding is `i // 10`, clipped to a maximum category of 3. So bins 0-9 map to 0, 10-19 to 1, 20-29 to 2, and 30-39 to 3.

ii.

```python
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

iii. The notes describe these as four equal 1-m spatial bins across the textured corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by using the same fixed 40 position bins as the neural representation; position is therefore identical across all trials at a given bin index.

ii.

```python
neural_trial = interp_spk[:, tr, :].astype(np.float32)
...
pos_tr = pos_bins.astype(np.int64)
```

iii. The notes present position as a direct readout of the same corridor-bin axis used for neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The agent uses `beh['run_pos']`, which is already organized as position-binned running speed, rather than `ft_RunSpeed`.

ii.

```python
run_pos = beh['run_pos']
run_speed = run_pos[:, :N_TEXTURE_BINS]
```

iii. The notes explicitly say the quartiles are computed from `run_pos` over the 40 texture bins.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script concatenates all `run_pos[:, :40]` values across every session, computes global 25th/50th/75th percentile edges, then digitizes each trial/bin speed value against those global thresholds.

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

iii. `CONVERSION_NOTES.md` justifies this as quartile binning across all 89 sessions and even records the resulting edge values.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global percentile thresholds are used. `np.digitize` converts each speed into categories 0-3, then the result is clipped to `[0, 3]`.

ii.

```python
speed_quantile_edges = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.digitize(run_speed, speed_quantile_edges)
speed_bins = np.clip(speed_bins, 0, 3)
```

iii. The notes justify this as quartile binning and report the global edge values.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by using the same 40 position bins as the position-interpolated neural data.

ii.

```python
run_speed = run_pos[:, :N_TEXTURE_BINS]
...
speed_tr = speed_bins[tr, :].astype(np.int64)
```

iii. The notes state that the running-speed signal lives on the same 40-bin corridor representation as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mainly handles missing data by skipping missing behavior files, skipping missing behavior keys, and skipping entire sessions that raise an exception during processing. It does not implement the reference behavior of trimming all streams to imaged frames, dropping post-imaging licks, or dropping trials with no surviving corridor frames.

ii.

```python
if not os.path.exists(beh_path):
    print(f"  Warning: behavior file not found for {exp_type}")
    continue
```

```python
if beh_key not in beh_data:
    print(f"  Warning: behavior key {beh_key} not found in {exp_type}")
    continue
```

```python
try:
    result = process_session(s, ROOT, all_stim_names, speed_quantile_edges)
except Exception as e:
    print(f"  ERROR processing session: {e}")
    ...
    continue
```

iii. There is no detailed justification beyond the warnings and exception handling in code. The notes instead emphasize that all 89 recordings had corresponding neural and retinotopy files.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive added processing is the per-neuron interpolation loop in `position_interpolate_spk()`, which runs `np.interp` once per retained neuron for every session. The script also makes a full-dataset pass over all `run_pos` arrays to compute global speed quantiles, in addition to loading the large spike files.

ii.

```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

```python
all_speeds = []
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
all_speeds = np.concatenate(all_speeds)
```

```python
spk_data = np.load(os.path.join(root, 'spk', spk_fn), allow_pickle=True).item()
spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```

iii. The explicit written justification is only partial: the notes say `numpy.interp` was chosen “for speed”.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization targets: the per-neuron interpolation loop, the per-lick event loop, the loop that builds `pos_bins`, and the list-comprehension-style loops that build `time_to_cue` and `time_since_start` inside every trial.

ii.

```python
for s in range(n_neurons):
    interp_spk[s, :] = np.interp(lin_pos, norm_pos, raw_spk[s, :])
```

```python
for i in range(len(lick_pos)):
    tr = int(lick_trind[i])
    pos = lick_pos[i]
    bin_idx = int(pos)
    if 0 <= bin_idx < n_bins and 0 <= tr < n_trials:
        lick_binary[tr, bin_idx] = 1
```

```python
for i in range(N_TEXTURE_BINS):
    pos_bins[i] = min(i // 10, 3)
```

```python
time_to_cue = np.array([(i - sound_pos[tr]) * BIN_SIZE_SEC
                         for i in range(N_TEXTURE_BINS)], dtype=np.float32)
time_since_start = np.array([i * BIN_SIZE_SEC for i in range(N_TEXTURE_BINS)],
                             dtype=np.float32)
```

iii. No explicit justification was written for leaving these loops in place.

## 12-c. What processing does the code repeat multiple times?

i. The code reparses session dates multiple times, scans all sessions once to collect stimulus names and again to collect all running speeds, and then iterates all sessions again for the actual conversion. It also fills `day_of_training` twice: first with a placeholder 0 vector and later by mutating each trial input in `main()`.

ii.

```python
def compute_day_of_training(date_str):
    return datetime.strptime(date_str, '%Y_%m_%d')
```

```python
for s in sessions:
    for wn in s['beh']['UniqWalls']:
        all_stim_set.add(str(wn))
...
for s in sessions:
    run_pos = s['beh']['run_pos'][:, :N_TEXTURE_BINS]
    all_speeds.append(run_pos.ravel())
```

```python
day_val = 0.0
day_of_training = np.full(N_TEXTURE_BINS, day_val, dtype=np.float32)
```

```python
for trial_input in result['input']:
    trial_input[1, :] = day_val
```

iii. There is no explicit justification for these repeated passes or the placeholder-fill pattern.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is interpolating each trial to 60 bins and then throwing away the last 20 bins. The script also computes and returns `session_date` from `process_session()` even though that field is never used downstream.

ii.

```python
interp_spk = position_interpolate_spk(..., n_bins=N_POS_BINS)
interp_spk = interp_spk[:, :, :N_TEXTURE_BINS]
```

```python
session_date = compute_day_of_training(datexp)
...
return {
    ...
    'session_date': session_date,
}
```

iii. No explicit justification is given for discarding bins 40-59 after interpolation or for carrying the unused `session_date` value.
