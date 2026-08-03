# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a master session index from `data/beh/Imaging_Exp_info.npy`, builds one unique session entry per neural recording (`mname_datexp_blk`), then loads retinotopy from `data/retinotopy`, spikes from `data/spk`, and behavior from the corresponding `Beh_<exp_type>.npy` file. Unlike the human reference, it reloads the behavior file separately for each session instead of grouping sessions by behavior file.

ii. 
```python
def build_session_map():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_map = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
            if spk_key not in session_map or 'stimtype' not in ndb:
                session_map[spk_key] = {
                    'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
                }
    return session_map
```
```python
def load_beh(session_info):
    beh_all = np.load(
        os.path.join(DATA_ROOT, 'beh', f"Beh_{session_info['exp_type']}.npy"),
        allow_pickle=True
    ).item()
    return beh_all[session_info['beh_key']]
```

iii. In `CONVERSION_NOTES.md`, the agent says the `Imaging_Exp_info.npy` file is the master index and that each physical recording should be used once. It also says sessions with different `stimtype` entries point to the same underlying trial data, so it is acceptable to choose any available behavior entry for that recording.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname`. During conversion the agent records the first-seen index for each mouse and writes `subject_idx` per session from that mapping.

ii. 
```python
mname = info['db']['mname']
...
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
...
subject_idx_list.append(subjects_seen[mname])
...
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The notes say subject count should be 19 and that mouse name should be used as the subject identifier.

## 1-c. How are the data split into sessions?

i. A session is defined by the concatenated key `mname_datexp_blk`. If the same recording appears multiple times in `Imaging_Exp_info.npy`, the agent keeps one entry, preferring entries without `stimtype`.

ii. 
```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {
        'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
    }
```

iii. In the notes, the agent says there are 142 index entries but only 89 unique physical recordings, and that sessions with `stimtype` variants share the same underlying behavior data.

## 1-d. How are the data split into trials?

i. Trials are split using `StartFr`. For trial `i`, the agent takes all frames from `StartFr[i]` up to `StartFr[i+1]`, or to the end of the session for the last trial. This includes gray-space and any pauses between corridor traversals.

ii. 
```python
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    start = max(0, start)
    end = min(nfr_use, end)
    n_frames = end - start
    if n_frames < 2:
        continue
```

iii. The notes explicitly justify this by saying: “Use frames from `StartFr` to start of next trial (or end of session). This captures corridor + gray space.”

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply a scientific quality filter to trials, but it drops trials with fewer than 2 frames under its `StartFr`-to-next-`StartFr` definition. It also drops whole sessions if fewer than 2 valid trials survive.

ii. 
```python
n_frames = end - start
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. No explicit justification beyond decoder compatibility was documented; the code implies the agent wanted at least two trials per session because the target format requires decodable sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spks` in the session spike file and `iarea` in the retinotopy file.

ii. 
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
```
```python
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes say the raw neural signal is already Suite2p deconvolved calcium activity, so no delta-F-over-F or deconvolution step is needed.

## 2-b. How is the `neural` data processed?

i. The agent filters neurons plane-by-plane using the retinotopy mask, concatenates the surviving neurons across planes, truncates to the number of imaged behavior frames, slices each trial by `StartFr` to the next `StartFr`, copies the slice, and stores it as `float16`. It does not pad or truncate trials to a common length.

ii. 
```python
for plane in planes:
    n = plane.shape[0]
    plane_mask = valid_mask[offset:offset+n]
    filtered.append(plane[plane_mask].astype(np.float16))
    offset += n
return np.concatenate(filtered, 0)
```
```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
...
trial_spk = spk[:, start:end].copy()
```

iii. The notes justify `float16` as a memory optimization and say the data are already deconvolved traces, so the main remaining processing is session/trial extraction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural quality filter is to exclude neurons with `iarea == -1` or `iarea == 7`, then assign the rest to one of four visual regions.

ii. 
```python
EXCLUDED_AREAS = {-1, 7}
...
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
region_idx = np.array([
    BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
    for ia in iarea[valid_mask]
], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says this matches the visual-cortex inclusion rule used in the reference code and paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry via `StartFr`, but each trial runs from `StartFr[i]` to `StartFr[i+1]` rather than to the textured-corridor end or to a fixed 32-frame window.

ii. 
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
...
trial_spk = spk[:, start:end].copy()
```

iii. The notes say the agent chose trial-start alignment with variable-length time series at the native frame rate, and explicitly chose to include corridor plus gray space.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses the native imaging frame as the time bin and does not apply temporal rebinning. The metadata reports `1000 / 3.17` ms per bin.

ii. 
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```
```python
'time_bin_size': TIME_BIN_MS,
```

iii. The notes explicitly say to use the native frame rate and avoid resampling.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent derives this input from `SoundFr`, together with the trial frame indices and a session sampling rate estimated from `ft`.

ii. 
```python
SoundFr = beh['SoundFr']
...
ft = beh['ft']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
fs = 1.0 / dt if dt > 0 else FRAME_RATE
...
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
```

iii. The notes say “Time to `SoundFr`” should be computed as `(SoundFr - current_frame) / fs`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the agent computes `(sound_fr - frame_idx) / fs`. If `SoundFr` is missing (`NaN`), it fills the entire trial with zeros instead of using interpolation or a missing-value marker.

ii. 
```python
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The notes justify this as a continuous time-varying variable aligned to trial frames. No separate written justification was found for the `NaN -> 0` fallback.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the exact same frame span, `start:end`, used for the neural trial slice.

ii. 
```python
trial_spk = spk[:, start:end].copy()
...
frame_idx = np.arange(start, end)
...
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The trajectory shows the agent repeatedly checked that inputs and neural data use the same `(n_features, T)` and `(n_neurons, T)` layout.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent derives day-of-training from session metadata fields in the experiment index, preferring `days` and then `sess#`.

ii. 
```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. The notes say the plan was to use `sess#` or `days` from `exp_info` if available, and only otherwise fall back to session order within subject.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The selected metadata value is cast to `float32` and broadcast across all time bins of each trial in the session.

ii. 
```python
day = np.float32(get_session_day(db))
...
np.full(n_frames, day, dtype=np.float32)
```

iii. The notes justify this as a per-trial scalar representing training context.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The agent derives this from `StartFr` plus the session frame rate estimate from `ft`.

ii. 
```python
StartFr = beh['StartFr'].astype(int)
...
ft = beh['ft']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. The notes say this variable should be computed as `(current_frame - StartFr) / fs`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent resets each trial’s clock to zero and fills the variable with `np.arange(n_frames) / fs`, so it is always 0 at the first bin and increases linearly by frame count.

ii. 
```python
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The notes justify it as a continuous variable that “starts at 0”.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned to the same `start:end` frame interval used for the neural trial slice, with time zero defined at `StartFr[i]`.

ii. 
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
...
trial_spk = spk[:, start:end].copy()
...
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. The trajectory shows the agent explicitly sanity-checked that `time_since_trial_start` begins at 0 for every trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The agent derives reward availability directly from `isRew`.

ii. 
```python
isRew = beh['isRew']
```

iii. The notes describe this as a binary per-trial scalar from the behavior table.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation is applied beyond converting the per-trial flag to float and broadcasting it across all bins of the trial.

ii. 
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. No additional justification was documented; the agent treated the stored `isRew` flag as already decoder-ready.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output is derived from the per-trial `WallName`.

ii. 
```python
WallName = beh['WallName']
...
stim = standardize_stim_name(str(WallName[i]))
```

iii. The notes say `WallName` should be used directly and then standardized into decoder categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent maps wall names through a custom equivalence table (`STIM_CATEGORY_MAP`) that preserves eight categories in the final full dataset, then encodes category indices in a second pass and broadcasts each trial’s stimulus identity across all time bins.

ii. 
```python
STIM_CATEGORY_MAP = {
    'circle1': 'circle1', 'circle2': 'circle2',
    'leaf1': 'leaf1', 'leaf2': 'leaf2', 'leaf3': 'leaf3',
    'leaf1_swap1': 'leaf1_swap1', 'leaf1_swap2': 'leaf1_swap2',
    'rock1': 'circle1', 'rock2': 'circle2',
    'wood1': 'leaf1', 'wood2': 'leaf2',
    'wood1_swap1': 'leaf1_swap1', 'wood1_swap2': 'leaf1_swap2',
    'brick1': 'circle1', 'brick2': 'circle2',
    'brick5': 'leaf3',
    'wood5': 'leaf3',
    'rock5': 'circle3',
}
```
```python
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. In the notes and trajectory, the agent argues that rock/wood/brick stimuli are equivalents of circle/leaf variants and that additional variants such as `wood5` and `rock5` should be folded into the same standardized family.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`.

ii. 
```python
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. The notes describe licking as a frame-level binary variable derived from whether any lick falls in a frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent makes a session-wide binary frame vector, truncating lick frame numbers to integers and setting a frame to 1 if at least one lick lands there.

ii. 
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. The notes explicitly justify this as “binary per frame”.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is sliced with the same `start:end` trial window used for the neural data.

ii. 
```python
trial_spk = spk[:, start:end].copy()
...
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. The notes and trajectory treat behavior arrays as already being indexed on the imaging frame grid, so the same frame slice is used for neural and licking outputs.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`, with `ft_CorrSpc` used to decide which frames are still in the textured corridor and which are gray-space frames.

ii. 
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. The notes say the plan was to use four 1 m bins across the texture and also track gray-space frames separately.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent creates a frame-level categorical vector with default value 4 (`gray`). It then assigns bins 0-3 for the four 1 m bins inside the textured corridor.

ii. 
```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly debates whether the task allows a gray-space category and decides to “use 4 bins and mark gray space as a 5th bin.”

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are at 0, 10, 20, 30, and 40 decimeters. Frames outside textured-corridor masks remain in category 4 (`gray`).

ii. 
```python
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3
```

iii. The notes say this corresponds to four 1 m texture bins plus an added gray-space category.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by slicing the session-wide `pos_bins` vector with the same `start:end` indices as the neural trial.

ii. 
```python
trial_spk = spk[:, start:end].copy()
...
pos_bins[start:end]
```

iii. The notes treat `ft_Pos` as already defined on the same frame grid as the neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. The notes identify `ft_RunSpeed` as the frame-level running-speed source variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first concatenates all sessions’ `ft_RunSpeed` arrays, keeps only positive speeds, computes global 25th/50th/75th percentile thresholds, and then thresholds each frame of each session into bins 0-3.

ii. 
```python
def collect_speed_quartiles(session_map, keys):
    all_speeds = []
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    valid = all_speeds > 0
    if valid.sum() == 0:
        return np.array([1.0, 2.0, 3.0])
    return np.percentile(all_speeds[valid], [25, 50, 75])
```
```python
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The notes say speed should be discretized into quartiles “computed across ALL running frames in the dataset.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Category 0 is below the first global threshold; categories 1, 2, and 3 are assigned by crossing the 25th, 50th, and 75th percentile thresholds of positive speeds.

ii. 
```python
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The notes justify global thresholds as a simple quartile discretization across the dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by slicing the frame-level `speed_bins` array with the same `start:end` indices as the neural trial.

ii. 
```python
trial_spk = spk[:, start:end].copy()
...
speed_bins[start:end]
```

iii. The notes treat running speed as already sampled on the imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent truncates all frame-level arrays to the number of jointly available neural/behavior frames, clips trial windows into valid bounds, drops out-of-range licks, fills missing `SoundFr` with zeros, and skips trials shorter than 2 frames.

ii. 
```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```
```python
start = max(0, start)
end = min(nfr_use, end)
...
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
...
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
```

iii. The notes justify the imaging/behavior truncation as necessary because behavior can run past imaging. No explicit written justification was found for replacing missing `SoundFr` with zeros.

## 12-a. What are the most time-consuming steps of the code?

i. The main cost is large-session neural loading/filtering and per-trial extraction. The agent also adds a dataset-wide pass over all behavior files to compute speed quartiles.

ii. 
```python
spk = load_spk_filtered(mname, datexp, blk, valid_mask)
...
for i in range(ntrials):
    ...
    trial_spk = spk[:, start:end].copy()
```
```python
speed_quartiles = collect_speed_quartiles(session_map, keys)
```

iii. The notes estimate neural loading/filtering at roughly 15-25 seconds per session and explicitly identify it as the runtime bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest vectorization opportunities are the Python loop over all trials in `process_session`, the loop over four position bins, and the nested loop that fills stimulus indices after session processing.

ii. 
```python
for i in range(ntrials):
    ...
    neural_trials.append(trial_spk)
    input_trials.append(inp)
    output_trials.append(out)
    stim_names.append(stim)
```
```python
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
```
```python
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. No explicit justification was documented for leaving these loops in Python; the notes instead emphasize I/O and memory concerns.

## 12-c. What processing does the code repeat multiple times?

i. The code reloads behavior once during `collect_speed_quartiles` and again during `process_session` for every session. It also performs stimulus handling in two passes: first storing placeholder zeros and `stim_names`, then filling indices later.

ii. 
```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
```
```python
def process_session(spk_key, session_info, speed_quartiles):
    ...
    beh = load_beh(session_info)
```
```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
...
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The notes justify the extra speed pass as a way to compute global quartiles “from behavior only” without loading neural data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds `stim_names` and writes a placeholder all-zero stimulus channel only to overwrite that channel later; it also tracks `nneu_total` only for logging. These steps do not affect the final scientific outputs beyond bookkeeping.

ii. 
```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),  # placeholder for stim idx
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
...
stim_names.append(stim)
```
```python
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```
```python
nneu_total = len(iarea)
...
print(f"{result['nneu']}({result['nneu_total']}) neurons, {result['ntrials']} trials, {time.time()-t_s:.1f}s")
```

iii. No explicit justification was documented for the placeholder stimulus pass; the trajectory instead shows the agent introduced it while trying to avoid a second neural-data load.
