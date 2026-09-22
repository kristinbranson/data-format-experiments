# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `beh/Imaging_Exp_info.npy` as the master index, builds a deduplicated list of sessions from `mname`, `datexp`, and `blk`, preloads all behavior files into one dictionary keyed by base session id, and then loads spikes and retinotopy per session during processing.

ii. ```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_beh = load_all_behavior(exp_info)
sessions = build_session_list(exp_info)
```
```python
spk = load_spk(mname, datexp, blk)
iarea = load_area_ids(mname, datexp)
beh = all_beh[key]
```

iii. In `CONVERSION_NOTES.md`, the agent says sessions repeat across experiment types and that behavior is identical across those duplicates, so it deduplicates to 89 physical sessions and strips `_swap1`/`_swap2` suffixes when loading behavior.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects by `mname`. The final `subjects` list is the sorted set of mouse names, and each session gets a `subject_idx` by lookup into that list.

ii. ```python
sessions.sort(key=lambda s: (s['mname'], s['datexp']))
subject_list = sorted(set(s['mname'] for s in sessions))
subject_idx_list.append(subject_list.index(sess['mname']))
```

iii. The notes describe 89 sessions across 19 mice and treat the mouse name as the subject identifier.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` combination. Repeated appearances of the same recording under multiple experiment types are deduplicated.

ii. ```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_dict:
    session_dict[key] = {
        'key': key,
        'mname': ndb['mname'],
        'datexp': ndb['datexp'],
        'blk': ndb['blk'],
        'exp_types': [],
    }
```

iii. The notes explicitly say `142 experiment entries → 89 unique sessions`.

## 1-d. How are the data split into trials?

i. For each session, a trial is the set of imaging frames where `ft_trInd == trial` and `ft_CorrSpc` is true. Trials are kept as variable-length frame windows, but windows with fewer than 2 corridor frames are marked invalid.

ii. ```python
for n in range(ntrials):
    frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
    if len(frames) >= 2:
        trials.append(frames)
    else:
        trials.append(None)
```

iii. In the notes, the agent says it wants time-based trials using all corridor frames rather than the paper’s position-interpolated representation.

## 1-e. How are trials filtered based on quality controls?

i. The only trial filter in the script is dropping trials with fewer than 2 corridor frames. The code does not remove extremely long stopped trials or apply a dataset-level trial-length cutoff.

ii. ```python
if len(frames) >= 2:
    trials.append(frames)
else:
    trials.append(None)
```
```python
valid_trials = [n for n in range(beh['ntrials']) if trial_frames_list[n] is not None]
```

iii. The notes say the agent chose to keep all corridor frames to preserve continuous behavior and licking, and later explicitly says it did not filter the 5607-frame outlier trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the concatenated `spks` arrays in each session’s spike file. Brain-region labels are derived from `iarea` in the retinotopy file.

ii. ```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in data['spks']], 0)
```
```python
trans = np.load(path, allow_pickle=True)
return trans['iarea']
```

iii. The notes identify the Suite2p deconvolved traces as the neural source and the retinotopy file as the source of area assignments.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, filters neurons by area, slices the kept frame window for each trial, and stores the result as `float32`. There is no dF/F computation, deconvolution, padding, or temporal rebinning.

ii. ```python
spk = load_spk(mname, datexp, blk)
neuron_mask, region_idx = get_neuron_mask_and_regions(iarea)
spk_filtered = spk[neuron_mask]
```
```python
neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The notes state that the traces are already deconvolved and that the agent wanted raw frame-rate, time-based trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by visual-area assignment: neurons with `iarea == -1` or `iarea == 7` are dropped, and the remaining codes are mapped into `V1`, `mHV`, `lHV`, and `aHV`.

ii. ```python
mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
```

iii. The notes say this matches the visual-cortex exclusion pattern seen in the reference code and that there is no additional quality filtering beyond earlier Suite2p processing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry by taking each trial’s corridor-frame window; the first kept frame of that window is effectively trial start for the saved arrays.

ii. ```python
frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The notes repeatedly describe the alignment event as corridor entry / trial start and justify using raw corridor frames to preserve continuous time structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one time bin. The script estimates frame duration from the median `ft` spacing and stores the median session value in metadata. No temporal rebinning is applied.

ii. ```python
dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
```
```python
'time_bin_size': median_dt * 1000,
```

iii. The notes state a frame rate of about 3.17 Hz, so each bin is about 315 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent derives it from `SoundFr` plus frame indices from each trial window, scaled by a session-level `dt_sec` estimated from `ft`.

ii. ```python
SoundFr = beh['SoundFr']
ft = beh['ft'][:nfr]
dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
```
```python
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. In the notes, the variable mapping is described as `SoundFr - frame_idx` converted to seconds; the trajectory also shows the agent planning to align to trial-start frames instead of interpolating event times on the true `ft` axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The code computes `(current_frame - SoundFr[trial]) * dt_sec` for every frame in the trial. This makes the value negative before the cue and positive after the cue.

ii. ```python
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. The notes explicitly describe this as a continuous, frame-based time variable built from frame offset times the imaging-frame duration.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same `frames` vector used to slice the neural data for the trial, so it has the same number of time bins as the neural array.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. The notes say all streams are aligned on the imaging frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session ordering information in the session index: mouse name plus experiment date, after deduplicating sessions.

ii. ```python
sessions.sort(key=lambda s: (s['mname'], s['datexp']))
```
```python
for m, sess_list in mouse_sessions.items():
    sess_list.sort(key=lambda s: s['datexp'])
    for i, s in enumerate(sess_list):
        s['day_index'] = i
```

iii. The notes describe `day_of_training` as chronological session index per mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted chronologically and assigned a zero-based index. That scalar is then broadcast across all time bins of every trial in the session.

ii. ```python
input_arr[1, :] = float(sess['day_index'])
```

iii. The notes justify this as the simplest way to capture experience progression from the available metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from each trial’s kept frame indices and the first kept frame in that trial, scaled by `dt_sec`. The code does not use `StartFr`.

ii. ```python
dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. The notes describe this as `frame_idx - start_frame` and say it is always nonnegative.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first saved frame of the trial is set to zero, and later frames increase by multiples of `dt_sec`. There is no interpolation of the fractional `StartFr` event time.

ii. ```python
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. The notes justify this as a simple continuous time axis aligned to the trial window.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed directly from the same frame window used to slice the neural data.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. The notes treat all trial-wise streams as frame-aligned.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, the per-trial rewarded-corridor flag in the behavior structure.

ii. ```python
isRew = beh['isRew']
input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. The notes map `isRew[trial]` directly to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code converts the per-trial reward flag to `1.0` or `0.0` and broadcasts that constant across all time bins of the trial.

ii. ```python
input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. The notes describe it as a discrete per-trial variable with no further transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, the per-trial corridor wall texture label.

ii. ```python
WallName = beh['WallName']
output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
```

iii. The notes identify `WallName` as the source field for the visual-stimulus output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent keeps all 15 distinct wall-texture names as separate categories by sorting them into `ALL_STIMULI` and indexing them with `STIM_TO_IDX`. The category is then broadcast across all frames of the trial.

ii. ```python
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
```
```python
output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
```

iii. The notes say the agent chose “15 stimulus categories” to preserve maximum information rather than collapse to broad texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, with `LickTrind` loaded but not used for the final signal.

ii. ```python
lick_fr = beh['LickFr'].astype(int) if len(beh['LickFr']) > 0 else np.array([], dtype=int)
lick_trind = beh['LickTrind'].astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
```

iii. The notes describe the output as binary lick presence from lick-frame events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code creates a frame-level binary indicator over the full session, sets valid lick frames to 1, and then slices that indicator by trial frame window.

ii. ```python
lick_indicator = np.zeros(nfr, dtype=np.int64)
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
if valid_mask.any():
    lick_indicator[lick_fr[valid_mask]] = 1
```
```python
lick_arrays.append(lick_indicator[frames].copy())
```

iii. The notes say this keeps licking as a time-varying binary decoder output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick trace is sampled on the same imaging-frame grid and sliced with the same per-trial `frames` window used for the neural data.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
output_arr[1, :] = lick_arrays[n]
```

iii. The notes treat lick timing as already aligned to frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the per-frame position trace.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr]
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. The notes say position is stored in decimeters and binned into meter-scale categories.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For the kept corridor frames, the code integer-divides position by 10 decimeters and clips the result to the range 0 to 3.

ii. ```python
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. The notes say this implements the requested four equal 1 m bins over the 4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position categories are `0-1m`, `1-2m`, `2-3m`, and `3-4m`, implemented by 10-decimeter bins.

ii. ```python
output_values = [
    ALL_STIMULI,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
]
```
```python
np.clip(ft_Pos[frames] // 10, 0, 3)
```

iii. The notes explicitly describe four 1 m position bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from the same kept imaging frames used for the neural trial matrix.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. The notes say all time-varying streams are frame-aligned.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, the per-frame running-speed trace.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
```
```python
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. The notes map `ft_RunSpeed` to the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first pools all corridor-frame speeds across all sessions, computes global 25th/50th/75th percentile thresholds, and then assigns each saved frame to a bin with `np.digitize`.

ii. ```python
corr_mask = ft_CorrSpc
if corr_mask.any():
    all_speeds.append(ft_RunSpeed[corr_mask])
```
```python
all_speeds_flat = np.concatenate(all_speeds)
speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
```
```python
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. The notes say “Speed quartiles computed globally” so that bin definitions are shared across the full dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are numeric percentile cutoffs computed once over the pooled corridor frames from all sessions. This is value-thresholding, not per-session rank-based equal-count binning.

ii. ```python
speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
```
```python
['Q1', 'Q2', 'Q3', 'Q4']
```

iii. The notes justify this as “global quartiles across all corridor frames in all sessions”.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is read on the imaging-frame grid and indexed by the same `frames` vector used for the neural data.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. The notes describe speed as a time-varying, frame-aligned output.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates all frame-level behavior arrays to the number of neural frames, ignores lick events outside that range, marks trials with fewer than 2 corridor frames invalid, and normalizes `_swap1`/`_swap2` behavior keys to a base session key.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr]
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
ft = beh['ft'][:nfr]
```
```python
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
```
```python
for suffix in ('_swap1', '_swap2'):
    if base_key.endswith(suffix):
        base_key = base_key[:-len(suffix)]
        break
```

iii. The notes mention the neural/behavior frame-count mismatch, the suffixed behavior keys, and the decision to treat very short corridor windows as invalid trials.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeatedly loading the large spike files and concatenating planes. The script does that in a first pass just to get frame counts and speeds, and again in the second pass to build the dataset.

ii. ```python
data = np.load(path, allow_pickle=True).item()
nfr = data['spks'][0].shape[1]
nneu_total = sum(s.shape[0] for s in data['spks'])
```
```python
spk = load_spk(mname, datexp, blk)
spk_filtered = spk[neuron_mask]
```

iii. The notes describe a two-pass conversion and report very large output sizes, implying that I/O and array materialization dominate runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the trial-extraction loop that runs `np.where` once per trial, the lick-array construction loop over trials, and the per-trial assembly loop in `process_session_full`.

ii. ```python
for n in range(ntrials):
    frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
```
```python
for n, frames in enumerate(trial_frames_list):
    ...
```
```python
for n in range(ntrials):
    frames = trial_frames_list[n]
    ...
```

iii. The script already labels one helper “vectorized”, but the main trial-wise work is still Python-loop driven.

## 12-c. What processing does the code repeat multiple times?

i. The script repeats spike-file loading across two passes, and recomputes per-session frame-related work rather than reusing it from pass 1. If `--show-processing` is used, it loads and processes the first sessions yet again for plotting.

ii. ```python
all_speeds, session_info = collect_speeds_from_behavior(sessions, all_beh)
```
```python
result = process_session_full(sess, all_beh, speed_quantiles)
```
```python
if args.show_processing and i < 2:
    plot_processing(sess, all_beh, speed_quantiles, ...)
```

iii. The repeated work follows from the agent’s two-pass design for global speed quantiles and from separate plotting code paths.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores some information only for logging or optional plots rather than for the converted dataset itself: `exp_types` in `session_dict`, `LickTrind`, `session_info` details such as total neuron counts, and plotting-specific reprocessing when enabled.

ii. ```python
'exp_types': [],
```
```python
lick_trind = beh['LickTrind'].astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
```
```python
session_info.append({
    'n_included': n_included,
    'nneu_total': nneu_total,
    'ntrials': ntrials,
    'ntrials_valid': n_valid,
    'nfr': nfr,
    'dt_sec': dt_sec,
})
```

iii. These choices reflect the agent’s emphasis on debugging, progress reporting, and optional visualization rather than minimal conversion-only work.
