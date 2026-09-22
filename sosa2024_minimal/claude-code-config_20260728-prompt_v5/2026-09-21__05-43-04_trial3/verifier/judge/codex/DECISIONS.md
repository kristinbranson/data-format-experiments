# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the 11 switch-task mice in `SESSIONS_DICT`, then iterates over every `.nwb` file in each matching `sub-<mouse>` directory under `/app/data`. Each session is opened with `h5py`, and neural and behavioral arrays are read directly from HDF5 paths.

ii.
```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, f'sub-{subj}')
    ...
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for nwb_file in nwb_files:
        ...
        neural, inp, out, n_neurons, info = process_session(
            subj, exp_day, nwb_path, scene)
```

```python
f = h5py.File(nwb_path, 'r')
dec_group = f['processing/ophys/Deconvolved']
fl_group = f['processing/ophys/Fluorescence']
neu_group = f['processing/ophys/Neuropil']
beh = f['processing/behavior/BehavioralTimeSeries']
```

iii. In the trajectory, the AI said the NWB dataset only contains the 11 switch-task mice and that fixed-condition mice are absent. It concluded that loading all `.nwb` files under those subject directories would cover the dataset, and chose direct `h5py` access after exploring the NWB structure.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded mouse IDs in `SESSIONS_DICT` and matched to subdirectories named `sub-<mouse>`.

ii.
```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))

for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, f'sub-{subj}')
```

iii. The trajectory summary states that the AI mapped the released `m<N>` IDs to the paper’s animals and intentionally limited processing to the 11 switch-task mice present in the NWB release.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session/day number is parsed from the filename (`_ses-XX_`) and matched to a scene entry in `SESSIONS_DICT`.

ii.
```python
for nwb_file in nwb_files:
    ses_str = nwb_file.split('_ses-')[1].split('_')[0]
    exp_day = int(ses_str)

    scene = None
    for entry in SESSIONS_DICT[subj]:
        if entry['exp_day'] == exp_day:
            scene = entry['scene']
            break
```

iii. In the trajectory, the AI explicitly concluded that the file names encode session day and that scene metadata from `sessions_dict.py` should be attached per session.

## 1-d. How are the data split into trials?

i. Trials are split by taking every positive sample in `trial_start` as a start index and every positive sample in `teleport` as an end index, then pairing them in order. Each trial uses the half-open slice `[t_start:t_end]`.

ii.
```python
trial_start_idx = np.where(trial_start_signal > 0)[0]
teleport_idx = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_start_idx), len(teleport_idx))
trial_start_idx = trial_start_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
t_start = trial_start_idx[t]
t_end = teleport_idx[t]
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The top-of-file decision comments say trial segmentation uses `trial_start` and `teleport`. The trajectory also says the AI planned to use those signals directly; there is no sign it adopted the reference solution’s rising-edge detection for teleport starts.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps almost all trials. It only skips trials whose end is not after the start and trials with fewer than 2 neural time bins. It also drops a session if fewer than 2 trials survive.

ii.
```python
if t_end <= t_start:
    continue
...
n_tp = neural.shape[1]
if n_tp < 2:
    continue
...
if neural is None or len(neural) < 2:
    print(f"  Skipping {subj} day {exp_day}: insufficient data")
    continue
```

iii. The trajectory does not show a separate discussion of trial-length QC. The implemented decision appears to be minimal decoder-format filtering rather than reproducing the reference’s `<50`-timepoint exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural signal is taken from the NWB `Deconvolved` arrays. Raw `Fluorescence` and `Neuropil` are loaded only to recompute dF/F for interneuron detection.

ii.
```python
dec_group = f['processing/ophys/Deconvolved']
fl_group = f['processing/ophys/Fluorescence']
neu_group = f['processing/ophys/Neuropil']
...
deconvolved = np.concatenate(dec_planes, axis=1)
fluorescence = np.concatenate(fl_planes, axis=1)
neuropil_data = np.concatenate(neu_planes, axis=1)
...
deconv_filtered = deconvolved[:, final_cell_indices]
```

iii. The AI’s docstring and final trajectory summary explicitly say it believed the NWB `Deconvolved` field already captured the full paper pipeline, so it used that for `neural`.

## 2-b. How is the `neural` data processed?

i. The saved neural data are not recomputed from fluorescence. The AI concatenates planes, filters cells, and then slices the stored deconvolved traces into trials. Separate dF/F preprocessing is only used for interneuron detection.

ii.
```python
deconvolved = np.concatenate(dec_planes, axis=1)
...
final_cell_indices = cell_indices[~is_int]
deconv_filtered = deconvolved[:, final_cell_indices]
...
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. In the trajectory and in the file header, the AI justified this by saying the NWB deconvolved traces were already the output of the full paper pipeline and therefore could be used directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two cell-level filters: `iscell` manual curation and exclusion of putative interneurons defined by correlation of dF/F with speed greater than 0.5.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
cell_mask = iscell[:, 0].astype(bool)
...
dff = compute_dff_for_interneuron_detection(
    F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)
is_int = detect_interneurons(dff, speed, threshold=0.5)
...
final_cell_indices = cell_indices[~is_int]
```

iii. The trajectory shows the AI checked the paper notebooks for `int_thresh = 0.5`, read the `is_putative_interneuron` function, and decided to match those paper filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials start at the first `trial_start` sample; no extra offset is applied. Alignment is therefore to the start of each extracted trial.

ii.
```python
t_start = trial_start_idx[t]
t_end = teleport_idx[t]
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The header comments and final summary both state that temporal alignment is to trial start, so the AI treated trial extraction itself as the alignment step.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI intended to keep the native imaging sampling and apply no temporal rebinning. In code, trial time is advanced by `1 / frame_rate`, and metadata uses `1000 / median_frame_rate` in ms.

ii.
```python
frame_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
dt = 1.0 / frame_rate
...
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
...
time_bin_ms = 1000.0 / median_frame_rate
```

iii. The top comments and trajectory summary say the AI believed the native frame rate was about 15.5 Hz and that no resampling was required.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the number of neural frames in the trial and the session frame rate, not from the behavioral timestamps array.

ii.
```python
frame_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
n_tp = neural.shape[1]
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. The AI’s justification in the header is that native imaging bins should be preserved; the trajectory does not show a separate argument for preferring frame count over the stored behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI computes a regularly spaced vector `0, dt, 2*dt, ...` where `dt = 1 / frame_rate`.

ii.
```python
dt = 1.0 / frame_rate  # time per frame in seconds
...
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. The decision follows the AI’s assumption that trial time should be reconstructed from the imaging sample rate rather than copied from the behavior timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector is created with the same number of bins as the neural slice for that trial and starts at 0 for the first neural sample.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
n_tp = neural.shape[1]
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
...
input_data = np.stack([
    time_from_start,
    ...
], axis=0)
```

iii. The trajectory repeatedly frames the dataset as already frame-aligned across modalities, and the code reflects that by building time from trial length rather than matching separate timestamps stream-by-stream.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The implemented code derives environment from the per-session scene metadata in `SESSIONS_DICT` via `parse_scene()` and `get_env_for_trial()`, not from the loaded `environment` behavior series.

ii.
```python
env_signal = beh['environment/data'][:]
...
def get_env_for_trial(scene_info, trial_idx):
    env_before, _, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return env_after if env_after is not None else env_before
    return env_before
...
env = get_env_for_trial(scene, t)
env_binary = 0 if env == 1 else 1
```

iii. The file header claims environment comes from the NWB environment time series, but the trajectory shows the AI also trusted scene names to encode session structure. The actual implementation follows the scene-based approach.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses the scene string, switches environment at trial 30 on switch sessions, and then maps `Env1 -> 0` and `Env2 -> 1`. It broadcasts that constant value across all time bins in the trial.

ii.
```python
SWITCH_TRIAL = 30
...
env = get_env_for_trial(scene, t)
env_binary = 0 if env == 1 else 1  # ENV1=0, ENV2=1
...
np.full(n_tp, env_binary, dtype=np.float32)
```

iii. In the trajectory, the AI said scene names encode the reward-zone/environment schedule and that day-8 transitions looked correct, which is the justification for this metadata-based reconstruction.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index over extracted trials. The underlying trial definitions come from `trial_start` and `teleport`.

ii.
```python
for t in range(n_trials):
    ...
    input_data = np.stack([
        time_from_start,
        np.full(n_tp, env_binary, dtype=np.float32),
        np.full(n_tp, t, dtype=np.float32),
        np.full(n_tp, prev_rewarded, dtype=np.float32),
    ], axis=0)
```

iii. The trajectory does not show any attempt to use the stored `trial number` series. The implemented decision is the standard sequential-within-session index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing beyond filling the entire trial with the loop counter `t`.

ii.
```python
np.full(n_tp, t, dtype=np.float32)
```

iii. The AI treated trial number as a per-trial covariate and simply broadcast the integer index across the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event timestamps. The current trial’s reward outcome is computed from whether any reward timestamp falls inside that trial’s time window; the next trial then inherits that as its “previous reward” label.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
...
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
...
np.full(n_tp, prev_rewarded, dtype=np.float32)
```

iii. The header comments explicitly say reward outcome is determined from reward-event timestamps mapped to trial windows. The same mechanism drives previous-trial outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned 0. After each trial, the AI sets `prev_rewarded = rewarded`, and the following trial is filled with that binary value across all time bins.

ii.
```python
prev_rewarded = 0
...
input_data = np.stack([
    time_from_start,
    np.full(n_tp, env_binary, dtype=np.float32),
    np.full(n_tp, t, dtype=np.float32),
    np.full(n_tp, prev_rewarded, dtype=np.float32),
], axis=0)
...
prev_rewarded = rewarded
```

iii. The trajectory does not show extra discussion here; this is the direct sequential implementation of “previous trial outcome.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives distance from the animal’s `position` time series plus reward-zone boundaries inferred from session metadata (`scene` in `SESSIONS_DICT`), not from the `reward_zone` behavior series.

ii.
```python
position = beh['position/data'][:]
...
zone_letter = get_reward_zone_for_trial(scene, t)
zone_start, zone_end = REWARD_ZONES[zone_letter]
...
dist = compute_distance_to_reward_zone(pos_trial, zone_start, zone_end)
```

iii. In the trajectory, the AI explored `reward_zone`, decided the nonzero values were noisy but localized to the active zone, and concluded the scene names encoded the schedule more directly. It therefore switched to scene-based reward-zone assignment.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside, positive after.

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    dist[before] = position[before] - zone_start
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end
    return dist
```

iii. The trajectory summary says the AI believed reward zone A/B/C locations from the paper should be turned into signed distances relative to the mouse’s position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 instruction-specified bins with hard-coded comparisons.

ii.
```python
out[dist < -50] = 0
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. The trajectory and file header both state that the AI was following the decoder-task bins here.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from the same per-trial position slice indexed by the neural trial boundaries, so it shares the neural trial’s bin count and sample order.

ii.
```python
pos_trial = position[t_start:t_end]
...
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
dist = compute_distance_to_reward_zone(pos_trial, zone_start, zone_end)
```

iii. The AI treated all within-trial behavioral streams as already aligned to the imaging frames once the common trial slice was chosen.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior series.

ii.
```python
position = beh['position/data'][:]
...
pos_trial = position[t_start:t_end]
```

iii. The trajectory shows the AI inspected position extensively while reasoning about reward zones, and there is no additional source for absolute position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the per-trial position trace, clamps it to `[0, 450]`, and then discretizes it.

ii.
```python
pos_trial = position[t_start:t_end]
pos_trial = np.clip(pos_trial, 0, 450)
...
pos_disc = discretize_position(pos_trial)
```

iii. The code comments show the AI wanted positions constrained to the 450 cm corridor before applying bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is discretized into five 90 cm bins covering the 450 cm track.

ii.
```python
out[position < 90] = 0
out[(position >= 90) & (position < 180)] = 1
out[(position >= 180) & (position < 270)] = 2
out[(position >= 270) & (position < 360)] = 3
out[position >= 360] = 4
```

iii. The AI was following the decoder-task discretization here.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same `[t_start:t_end]` slice as the neural data for each trial.

ii.
```python
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
pos_trial = position[t_start:t_end]
```

iii. The trajectory indicates the AI assumed the behavior and neural arrays were frame-aligned once cropped to a common session length.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior series.

ii.
```python
lick = beh['lick/data'][:]
...
lick_trial = lick[t_start:t_end]
```

iii. The file header explicitly says lick is taken from the NWB lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized: any positive value becomes 1, otherwise 0.

ii.
```python
lick_binary = (lick_trial > 0).astype(np.int64)
```

iii. The header comments state this decision directly: “any lick > 0 → 1.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same trial slice as the neural data and is stored as a time-varying vector of the same length.

ii.
```python
lick_trial = lick[t_start:t_end]
...
output = np.stack([
    dist_disc,
    pos_disc,
    speed_disc,
    lick_binary,
    ...
], axis=0)
```

iii. As with position, the AI assumes trial slicing is sufficient for alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from session metadata (`scene`) in `SESSIONS_DICT`, parsed into environment/zone components and switched at trial 30 when appropriate.

ii.
```python
def parse_scene(scene):
    ...

def get_reward_zone_for_trial(scene_info, trial_idx):
    env_before, zone_before, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return zone_after
    return zone_before
...
zone_letter = get_reward_zone_for_trial(scene, t)
zone_idx = {'A': 0, 'B': 1, 'C': 2}[zone_letter]
```

iii. The trajectory explicitly says the AI concluded that scene names encode the reward-zone schedule and that switch days change at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the session scene string, determines the active letter for the current trial, maps `A/B/C` to `0/1/2`, and broadcasts that categorical value across the trial.

ii.
```python
zone_letter = get_reward_zone_for_trial(scene, t)
zone_idx = {'A': 0, 'B': 1, 'C': 2}[zone_letter]
...
np.full(n_tp, zone_idx, dtype=np.int64)
```

iii. The trajectory justification is the same as for 7-a: the AI distrusted the raw `reward_zone` values as a direct label source and preferred the paper’s session metadata.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event timestamps and the trial start/end timestamps.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
...
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
```

iii. The file header says reward outcome is determined by mapping reward-event timestamps to trial windows, and the trajectory repeats that summary.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward outcome is 1 if any reward timestamp falls between that trial’s first and last timestamps, else 0. The value is then broadcast across the trial.

ii.
```python
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
...
np.full(n_tp, rewarded, dtype=np.int64)
```

iii. The AI’s explicit justification was that reward events are naturally per-event timestamps, so per-trial inclusion in the trial window is the relevant binary summary.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code does limited defensive handling. It crops neural and behavioral arrays to the shorter session length, truncates starts/ends to the smaller count of `trial_start` and `teleport` positives, skips invalid or 1-bin trials, and skips sessions with zero neurons or fewer than 2 trials. It does not warn on length mismatches and does not explicitly handle missing reward-zone labels because those are reconstructed from metadata.

ii.
```python
n_frames = min(n_frames_neural, n_frames_behav)

deconvolved = deconvolved[:n_frames]
fluorescence = fluorescence[:n_frames]
...
trial_start_idx = trial_start_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
...
if t_end <= t_start:
    continue
...
if n_tp < 2:
    continue
...
if n_neurons == 0:
    return None, None, None, 0, {}
```

iii. The trajectory does not show a dedicated missing-data policy beyond practical fixes needed to get the conversion and decoder verification to run.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading full NWB arrays (`Deconvolved`, `Fluorescence`, `Neuropil`, behavior), recomputing dF/F for interneuron detection with per-trial and per-cell loops, and iterating through every trial to build output arrays.

ii.
```python
dec_planes = [dec_group[p]['data'][:] for p in planes]
fl_planes = [fl_group[p]['data'][:] for p in planes]
neu_planes = [neu_group[p]['data'][:] for p in planes]
...
for start, stop in zip(trial_starts, trial_ends):
    ...
    for c in range(n_cells):
        bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
...
for t in range(n_trials):
    ...
```

iii. The trajectory shows repeated long conversion runs and subsequent decoder validation; the implementation itself makes the main hotspots clear.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-cell Gaussian-smoothing loops inside `compute_dff_for_interneuron_detection`, the per-cell correlation loop in `detect_interneurons`, and parts of the per-trial trial-building loop.

ii.
```python
for c in range(n_cells):
    bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
...
for c in range(n_cells):
    r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
...
for t in range(n_trials):
    ...
```

iii. The AI did not discuss vectorization explicitly in the trajectory; this is an implicit consequence of the implemented structure.

## 13-c. What processing does the code repeat multiple times?

i. The code reads both stored deconvolved data and raw fluorescence/neuropil for every session, then recomputes dF/F solely for interneuron filtering before discarding it. It also repeatedly reparses scene information per trial via helper calls.

ii.
```python
deconvolved = np.concatenate(dec_planes, axis=1)
fluorescence = np.concatenate(fl_planes, axis=1)
neuropil_data = np.concatenate(neu_planes, axis=1)
...
dff = compute_dff_for_interneuron_detection(
    F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)
...
zone_letter = get_reward_zone_for_trial(scene, t)
env = get_env_for_trial(scene, t)
```

iii. The trajectory makes clear that the AI wanted the stored deconvolved signal for `neural` but still wanted a separate dF/F reconstruction to match the paper’s interneuron filter.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `env_signal` but never uses it; it computes full-session dF/F traces that are only used for a binary interneuron mask and then discarded; and it reads raw fluorescence/neuropil despite ultimately saving only the precomputed deconvolved signal.

ii.
```python
env_signal = beh['environment/data'][:]
...
dff = compute_dff_for_interneuron_detection(
    F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)
...
deconv_filtered = deconvolved[:, final_cell_indices]
```

iii. This follows directly from the AI’s mixed design choice in the header and trajectory: trust NWB `Deconvolved` for the final neural signal, but separately reconstruct dF/F just to reproduce the interneuron exclusion step.
