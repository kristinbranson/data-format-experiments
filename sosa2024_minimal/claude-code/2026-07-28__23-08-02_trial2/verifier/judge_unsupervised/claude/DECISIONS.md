# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from a data directory organized by subject subdirectories (e.g., `data/sub-m11/`). Each `.nwb` file is opened with `h5py`, and behavioral and neural data are extracted from the HDF5 structure. The function `load_nwb_session()` reads position, speed, lick, trial_start, teleport, trial number, environment, reward_zone, timestamps, autoreward, reward events, iscell, plane index, deconvolved activity, and fluorescence. The main loop in `convert_all_data()` iterates over sorted subject directories, then sorted NWB files within each.

ii.
```python
def load_nwb_session(filepath):
    with h5py.File(filepath, 'r') as f:
        bts = f['processing/behavior/BehavioralTimeSeries']
        ophys = f['processing/ophys']
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick = bts['lick/data'][:]
        # ... (reads all behavioral and neural fields)
```

```python
def convert_all_data(data_dir, sample_only=False):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    for subj_dir_name in subjects:
        session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
        for sess_file in session_files:
            nwb_data = load_nwb_session(filepath)
            result = process_session(nwb_data, scene, exp_day)
```

iii. The AI noted that the data was in NWB format from DANDI archive and used h5py to directly read the HDF5 structure, extracting all relevant behavioral and neural data streams.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by subdirectory names (e.g., `sub-m3`, `sub-m11`). A mapping `SUBJECT_MAP` converts these to GCAMP identifiers. The subject list is built dynamically as sessions are processed. Only the 11 switch-task mice present in the NWB data are included (GCAMP2, GCAMP6, GCAMP10 fixed-condition mice are absent).

ii.
```python
SUBJECT_MAP = {
    'sub-m3': 'GCAMP3',
    'sub-m4': 'GCAMP4',
    # ...
}
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
# ...
mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'
if mouse_name not in subject_names:
    subject_names.append(mouse_name)
subj_idx = subject_names.index(mouse_name)
```

iii. From CONVERSION_NOTES: "Subjects: 11 mice (switch task), each with 12-14 sessions." The 3 fixed-condition mice are not in the NWB dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by filename (e.g., `sub-m11_ses-03_behavior+ophys.nwb`). The experimental day is extracted from the filename. Session metadata (scene name, reward zone info) is looked up from the hardcoded `SESSIONS_INFO` dictionary, which mirrors the reference code's `sessions_dict.py`.

ii.
```python
session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])
for sess_file in session_files:
    ses_part = sess_file.split('_')[1]  # 'ses-03'
    exp_day = int(ses_part.split('-')[1])
    scene = SESSIONS_INFO[gcamp_name][exp_day]
```

iii. The AI transcribed session info from the reference `sessions_dict.py` into a Python dict `SESSIONS_INFO` mapping (subject, day) to scene name.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` and `teleport` event markers in the NWB behavioral timeseries. The indices where `trial_start > 0` mark trial starts, and indices where `teleport > 0` mark trial ends. Each trial spans from start to teleport index.

ii.
```python
tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
teleport_idx = np.where(nwb_data['teleport'] > 0)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))
# ...
for i in range(n_trials):
    start = tstart_idx[i]
    end = teleport_idx[i]
```

iii. From CONVERSION_NOTES: "Trials defined by `trial_start` and `teleport` events in the NWB behavioral timeseries. Each trial spans from trial_start to teleport (the on-track portion only, excluding teleport zone)."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if: (1) `end <= start` (invalid boundaries), (2) fewer than 2 timepoints, or (3) lick sensor errors are detected (though error trials are not excluded -- their lick data is zeroed out). Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
if end <= start:
    continue
if n_timepoints < 2:
    continue
# Lick error correction sets lick to NaN (then 0), but trial is NOT excluded
lck = np.nan_to_num(lck, nan=0.0)
```

iii. The AI decided not to exclude lick-error trials entirely but instead zeroed their lick data. The CONVERSION_NOTES state: "Lick data for error trials is set to 0 (not NaN) after correction."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the deconvolved calcium activity stored in `processing/ophys/Deconvolved/` in the NWB files. For multi-plane sessions (m17, m18), data from multiple planes are concatenated.

ii.
```python
deconv_keys = list(ophys['Deconvolved'].keys())
deconv_list = []
for pk in sorted(deconv_keys):
    deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
deconvolved = np.concatenate(deconv_list, axis=1)
```

iii. From CONVERSION_NOTES: "Used deconvolved calcium activity from NWB files (processing/ophys/Deconvolved). This corresponds to the OASIS-deconvolved signal described in the paper."

## 2-b. How is the `neural` data processed?

i. The deconvolved activity is used directly from the NWB files without additional processing (no dF/F computation, no smoothing, no re-deconvolution). The NWB files already contain pre-processed deconvolved signals. For multi-plane animals, planes are pooled by concatenation along the neuron axis.

ii.
```python
deconvolved = np.concatenate(deconv_list, axis=1)
# ...
neural = deconvolved[:, neuron_mask]  # (timepoints, n_neurons)
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. The AI noted: "For multi-plane animals (m17, m18), planes are pooled per the paper: 'ROIs were identified separately per plane, but planes were pooled for all analyses'."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps: (1) Only ROIs marked as cells by Suite2P curation (`iscell[:, 0] == 1`) are kept. (2) Putative interneurons are excluded based on Pearson correlation > 0.5 between fluorescence and running speed.

ii.
```python
cell_mask = nwb_data['iscell'][:, 0] == 1
deconvolved = nwb_data['deconvolved'][:, cell_mask]
fluorescence = nwb_data['fluorescence'][:, cell_mask]

is_interneuron = identify_interneurons(
    fluorescence, nwb_data['speed'], valid_mask,
    threshold=SPEED_CORR_THRESHOLD  # 0.5
)
neuron_mask = ~is_interneuron
neural = deconvolved[:, neuron_mask]
```

iii. From CONVERSION_NOTES: "Suite2P curation: Only ROIs with iscell[:, 0] == 1 are included. Interneuron exclusion: ROIs with Pearson correlation > 0.5 between fluorescence (dF/F) and running speed are excluded."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's data is sliced from `tstart_idx[i]` to `teleport_idx[i]`, giving the on-track portion starting at the trial start event.

ii.
```python
start = tstart_idx[i]
end = teleport_idx[i]
trial_n = neural[start:end, :].T.astype(np.float32)
```

iii. From CONVERSION_NOTES: "Aligned to start of trial (trial_start event). off_start = 0.0 (alignment event is the trial start itself)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5 Hz (~64.5 ms per frame). No rebinning is applied.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
'imaging_rate_hz': 15.5078125,
```

iii. From CONVERSION_NOTES: "Time bin size: ~64.48 ms (1/15.5078125 Hz imaging rate)."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is computed from the trial index and the imaging rate, not directly from a raw variable. The number of timepoints within the trial and the frame time (1/imaging_rate) are used.

ii.
```python
frame_time = 1.0 / nwb_data['imaging_rate']
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. The AI computed time as frame index * frame period, which assumes evenly spaced frames at the imaging rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple linear ramp from 0 to `(n_timepoints-1) * frame_time` is created using `np.arange`. No additional processing.

ii.
```python
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
```

iii. No special processing noted; it's a straightforward computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time array has the same number of timepoints as the neural data (both span from `start` to `end`), so they are inherently aligned.

ii.
```python
n_timepoints = end - start
time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. Alignment is implicit since both use the same start/end indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the scene name in `SESSIONS_INFO`, not from the raw NWB `environment` timeseries. The scene name encodes `Env1` or `Env2`.

ii.
```python
env_before, env_after = parse_scene_environment(scene)
env_per_trial = np.full(n_trials, env_before, dtype=int)
if env_after is not None:
    ct = min(SWITCH_TRIAL, n_trials)
    env_per_trial[ct:] = env_after
```

iii. The AI parsed the scene name to determine environment (0 for Env1, 1 for Env2), handling day 8 environment switches.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene names are parsed: if only "Env1" appears, environment=0; if only "Env2", environment=1. For day 8 (both envs in scene name), first 30 trials get one env, remaining get the other.

ii.
```python
def parse_scene_environment(scene):
    if 'Env1' in scene and 'Env2' not in scene:
        return 0, None
    elif 'Env2' in scene and 'Env1' not in scene:
        return 1, None
    elif 'Env1' in scene and 'Env2' in scene:
        parts = scene.split('_to_')
        env_before = 0 if 'Env1' in parts[0] else 1
        env_after = 0 if 'Env1' in parts[1] else 1
        return env_before, env_after
```

iii. The approach follows the reference code's `env_morph_dict` mapping.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `i` in the trial processing loop, which is a 0-indexed counter within the session.

ii.
```python
input_arr[2, :] = float(i)  # trial number
```

iii. The AI used 0-indexed trial number within the session rather than the `trial_num` field from the NWB file.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop index. The trial number is the 0-based index of the trial within the session.

ii.
```python
input_arr[2, :] = float(i)
```

iii. This is a straightforward assignment of the trial index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `reward_per_trial`, which is itself computed by checking whether any reward timestamps fall within each trial's time boundaries.

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1

prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. From CONVERSION_NOTES: "A trial is 'rewarded' if any reward timestamp falls within the trial boundaries. Previous trial outcome for the first trial of each session is set to 0."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The reward outcome array is shifted by one position: `prev_outcome[1:] = reward_per_trial[:-1]`. The first trial always gets 0 (no previous trial).

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = reward_per_trial[:-1]
```

iii. Simple one-trial shift of the reward outcome array.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from: (1) the `position` timeseries from the NWB behavioral data, and (2) the reward zone coordinates determined from the `SESSIONS_INFO` scene name and the `REWARD_ZONES` dictionary.

ii.
```python
pos = nwb_data['position'][start:end]
rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
```

iii. The AI combined the animal's position with the known reward zone boundaries to compute signed distance.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if before the zone (position < rz_start), zero if inside, positive if after (position > rz_end).

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end
    return dist
```

iii. This computes distance to the nearest edge of the reward zone with appropriate sign.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins as specified in the instructions.

ii.
```python
def discretize_distance_to_reward(distance):
    out = np.zeros_like(distance, dtype=int)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. The bins match the specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial boundaries (start:end indices) are used for both position and neural data, so alignment is implicit.

ii.
```python
pos = nwb_data['position'][start:end]
# same start:end as neural[start:end, :]
```

iii. Both behavioral and neural data share the same timebase in the NWB file.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` timeseries in the NWB behavioral data.

ii.
```python
pos = nwb_data['position'][start:end]
pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
```

iii. Position values are clipped to [0, 450] before discretization.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, TRACK_LENGTH] then discretized into 5 equal bins (each 90 cm).

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out
```

iii. Uses floor division for binning, clamped to valid bin range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal bins: 0-90 cm (bin 0), 90-180 (bin 1), 180-270 (bin 2), 270-360 (bin 3), 360-450 (bin 4).

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 450 / 5 = 90
out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. Matches the specification for 5 equal-sized bins on a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial boundaries as neural data (start:end indices).

ii.
```python
pos = nwb_data['position'][start:end]  # same indices as neural
```

iii. Implicit alignment through shared indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` timeseries in the NWB behavioral data (`processing/behavior/BehavioralTimeSeries/lick/data`).

ii.
```python
lick = bts['lick/data'][:]
# ...
lck = licks_corrected[start:end]
```

iii. Lick data is read directly from NWB, then corrected for sensor errors.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Three steps: (1) Lick sensor error correction: trials where >30% of samples have cumulative lick count >2 have their lick data set to NaN. (2) NaN values are replaced with 0. (3) Lick values are binarized: >0 becomes 1, else 0.

ii.
```python
licks_corrected, error_trials = correct_lick_sensor_errors(
    nwb_data['lick'], tstart_idx, teleport_idx,
    threshold=LICK_ERROR_THRESHOLD  # 0.3
)
# ...
lck = np.nan_to_num(lck, nan=0.0)
lck_binary = (lck > 0).astype(int)
```

iii. From CONVERSION_NOTES: "Per the paper: trials where >30% of imaging frame samples have cumulative lick count >2 are flagged. Lick data for error trials is set to 0 (not NaN) after correction. Remaining lick counts binarized: >0 = lick, 0 = no lick."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial boundaries (start:end) as neural data.

ii.
```python
lck = licks_corrected[start:end]  # same as neural[start:end, :]
```

iii. Implicit alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `SESSIONS_INFO` dictionary (which encodes the scene name per session per subject, transcribed from the reference code's `sessions_dict.py`). The scene name determines which reward zone (A, B, or C) is active per trial, including switch logic.

ii.
```python
rz_labels = parse_scene_reward_zones(scene, n_trials)
# ...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]
```

iii. The AI used the known session metadata rather than trying to infer the reward zone from behavioral data.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine zone(s). For single-zone sessions, all trials get the same label. For switch sessions, trials before trial 30 get the pre-switch zone and trials 30+ get the post-switch zone. Labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    if scene.endswith('_LocationA'):
        rz_labels[:] = 'A'
    # ... etc for B, C
    else:
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after
```

iii. Follows the reference code's `get_reward_zones()` logic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` event data and timestamps in the NWB file (`processing/behavior/BehavioralTimeSeries/Reward/data` and `Reward/timestamps`).

ii.
```python
reward_data = bts['Reward/data'][:]
reward_ts = bts['Reward/timestamps'][:]
# ...
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. Reward is determined by whether any reward timestamp falls within the trial boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks if any reward event timestamp falls within the trial's start and end times. If yes, the trial is marked as rewarded (1), otherwise not rewarded (0).

ii.
```python
reward_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    start_t = timestamps[tstart_idx[i]]
    end_t = timestamps[teleport_idx[i]]
    if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
        reward_per_trial[i] = 1
```

iii. The reference code checks `np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)` which additionally requires the mouse to be in the reward zone. The AI's approach only checks for reward timestamps, which could differ slightly.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several handling approaches: (1) If behavioral and neural data lengths differ, both are truncated to the minimum length. (2) NaN lick values (from error correction) are replaced with 0. (3) Sessions with fewer than 2 valid trials are skipped. (4) Trials with invalid boundaries (end <= start or < 2 timepoints) are skipped. (5) Sessions with no valid neurons are skipped.

ii.
```python
# Length mismatch handling
if n_behavior != n_neural:
    min_len = min(n_behavior, n_neural)
    position = position[:min_len]
    # ... truncate all arrays

# NaN handling
lck = np.nan_to_num(lck, nan=0.0)

# Invalid trial handling
if end <= start:
    continue
if n_timepoints < 2:
    continue
```

iii. From CONVERSION_NOTES: "One session (m18 ses-01) had a 1-sample mismatch between behavioral and neural data; truncated to minimum length."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB files with h5py (reading large neural datasets into memory). (2) The interneuron identification loop, which computes per-neuron correlations. (3) The per-trial processing loop which slices and constructs arrays for each trial.

ii.
```python
# Loading all data from NWB
deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
# Interneuron loop
for c in range(n_neurons):
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

iii. NWB loading dominates because each file contains large neural arrays that must be read into memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The interneuron identification loop (`for c in range(n_neurons)`) computes correlations one neuron at a time; this could be vectorized using matrix operations. (2) The reward determination loop (`for i in range(n_trials)`) checks reward timestamps per trial; this could use searchsorted for batch processing. (3) The lick error correction loop iterates trial by trial.

ii.
```python
# Could be vectorized:
for c in range(n_neurons):
    neural_ts = neural_data[valid_mask, c]
    speed_ts = speed_data[valid_mask]
    r = np.corrcoef(neural_ts, speed_ts)[0, 1]
```

iii. The per-neuron correlation loop is the most obvious candidate for vectorization.

## 13-c. What processing does the code repeat multiple times?

i. (1) The code reads both fluorescence and deconvolved data from NWB even though fluorescence is only used for interneuron detection. (2) Position data is accessed multiple times (once for valid_mask in interneuron detection, once per trial). (3) The reward zone parsing is called once per session but could be precomputed in batch.

ii.
```python
# Both loaded, but fluorescence only used for interneuron check:
deconvolved = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(fluor_list, axis=1)
```

iii. Loading fluorescence is necessary for the interneuron check, but the dual load of large arrays increases memory usage.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `autoreward` field is loaded from NWB but never used. (2) The `scanning` field is accessed in exploratory code but not in the final conversion. (3) The `reward_zone_ts` (rzone timeseries) is loaded but not used for reward zone determination (scene metadata is used instead). (4) The `trial_num` field from NWB is loaded but the trial number is computed from loop index instead. (5) The `plane_idx` is loaded but not used (all neurons get brain_region_idx = 0 for CA1).

ii.
```python
autoreward = bts['autoreward/data'][:]  # loaded but unused
reward_zone = bts['reward_zone/data'][:]  # loaded but unused
trial_num = bts['trial number/data'][:]  # loaded but loop index used instead
plane_idx = seg['planeIdx'][:]  # loaded but unused
```

iii. These fields are loaded for completeness/exploration but have no impact on the output.
