# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed every `*.nwb` file under `/app/data/sub-*`, opened each file with `h5py`, and processed every file as one session. Within each session it read behavioral time series, reward timestamps, and ophys groups, then later split them into trials.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

with h5py.File(nwb_path, 'r') as f:
    bts = f['processing/behavior/BehavioralTimeSeries']
    position = bts['position/data'][:]
    speed = bts['speed/data'][:]
    lick_raw = bts['lick/data'][:]
    reward_timestamps = bts['Reward/timestamps'][:]
    ophys = f['processing/ophys']
```

iii. In `CONVERSION_NOTES.md`, the AI said the dataset consists of 11 subject directories and ~152 sessions and that all NWB files should be loaded. The trajectory shows it intentionally switched to direct NWB/HDF5 access rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file’s `general/subject/subject_id`, collected into a set, and sorted at the end. Session-to-subject assignment is stored by looking up each session’s subject in that sorted subject list.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
...
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The notes say subjects correspond to the 11 mice in the NWB tree. The code uses the embedded subject metadata instead of directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `process_session()` returns one session object, and the outer loop appends one entry per NWB file to `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
...
for sess in all_sessions:
    neural_list.append(sess['neural_trials'])
    input_list.append(sess['input_trials'])
    output_list.append(sess['output_trials'])
```

iii. The notes explicitly describe “152 sessions” as the 152 NWB files and repeatedly refer to one-file-per-session processing.

## 1-d. How are the data split into trials?

i. Trial boundaries are taken directly from indices where `trial_start_flag > 0` and `teleport_flag > 0`. The code truncates to the shorter of the two arrays and slices trial data as `[start:end]`, excluding the teleport frame.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]

n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
...
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. The notes and trajectory say the AI inspected `trial_start` and `teleport`, concluded that `position[s:e]` covers the full track and excludes the teleport jump, and believed this matched the reference’s effective alignment.

## 1-e. How are trials filtered based on quality controls?

i. The code only drops obviously invalid or extremely short trials: trials with `e <= s` or with fewer than 2 frames. It does not apply the 50-timepoint minimum described in the human reference.

ii.
```python
if e <= s or (e - s) < 2:
    ...
    continue
...
if valid_trial_count < 2:
    print(f"  Skipping {session_label}: only {valid_trial_count} valid trials")
    return None
```

iii. The notes justify keeping trials within `trial_start` to `teleport` and focus on decoder validity. They do not document the reference solution’s `<50`-frame trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural matrices come from `processing/ophys/Deconvolved/.../data` after ROI filtering. The code also loads raw fluorescence and neuropil traces to compute dF/F used only for interneuron detection.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
...
deconv_cells = deconv_all[:, cell_mask].T
neural_data = deconv_cells[final_cell_mask]
```

iii. `CONVERSION_NOTES.md` says the decoder should use deconvolved events, but that dF/F from `F` and `Fneu` is still needed to reproduce interneuron exclusion.

## 2-b. How is the `neural` data processed?

i. The AI concatenates all planes, crops neural/behavior streams to the same total length if needed, filters ROIs by `iscell`, computes per-trial dF/F from `F` and `Fneu`, excludes putative interneurons using dF/F-speed correlation, and then saves deconvolved events for the remaining cells without temporal rebinning.

ii.
```python
deconv_all = np.concatenate(deconv_list, axis=1)
F_all = np.concatenate(F_list, axis=1)
Fneu_all = np.concatenate(Fneu_list, axis=1)
...
cell_mask = iscell[:, 0].astype(bool)
...
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
...
neural_data = deconv_cells[final_cell_mask]
```

iii. The notes frame this as “same as reference code/paper”: use deconvolved events for the decoder, but reproduce dF/F-based interneuron detection from the original analysis pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered by the NWB `iscell` flag, then a second filter removes putative interneurons whose dF/F is correlated with speed above `0.5`.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
...
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

iii. The notes repeatedly justify this with the paper’s interneuron-exclusion description, and the trajectory shows the AI spent time optimizing this step because it considered it essential.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial from `trial_start_inds[i]` to `teleport_inds[i]`. Within each trial, time 0 corresponds to the first frame after the trial-start flag.

ii.
```python
s = trial_start_inds[i]
e = teleport_inds[i]
trial_neural = neural_data[:, s:e]
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes say the alignment event is “trial start” and that no extra offset correction is needed because the behavior and imaging streams are already synchronized.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code keeps the native imaging-frame resolution. For two-plane recordings it divides the stored imaging rate by the number of planes to recover the effective per-plane sampling rate, then uses that as the common time bin. No rebinning is applied.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
...
'time_bin_size': time_bin_ms,
```

iii. The notes state the intended bin size is about 64.5 ms at ~15.5 Hz and explicitly say “Time bin = imaging frame”.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived indirectly from trial length plus the effective imaging rate. The code does not use behavioral timestamps to form the input; it uses `np.arange(n_t) / effective_rate`.

ii.
```python
trial_timestamps = pos_timestamps[s:e]
...
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. In the notes the AI initially planned `(t - trial_start) / imaging_rate`; in code it implemented that by assuming uniform frame spacing rather than subtracting recorded timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, it creates a frame index `0, 1, ..., n_t-1` and divides by the effective imaging rate to produce elapsed seconds since trial start.

ii.
```python
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The trajectory and notes indicate the AI assumed the behavior and imaging clocks are regular enough that frame count divided by rate is equivalent to subtracting timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned one-to-one with neural frames because the time vector is generated with the same `n_t` as `trial_neural = neural_data[:, s:e]`.

ii.
```python
trial_neural = neural_data[:, s:e]
n_t = e - s
input_arr = np.zeros((4, n_t), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. The notes say all streams are already synchronized at imaging-frame resolution, so matching trial slices is sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
```

iii. The notes map this variable directly to the paper’s morph/environment identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code takes the median nonnegative environment value within a trial, casts it to float, and broadcasts that per-trial scalar across all time bins in the input array.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
input_arr[1, :] = env_type
```

iii. The notes justify this as a per-trial variable that should be constant within trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is not taken from the stored `trial number` stream. The AI uses the loop index `i` over extracted trials as the within-session trial number.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. The notes and trajectory say the stored `trial number` was considered unreliable relative to the chosen `trial_start`/`teleport` trial segmentation, so the sequential extracted-trial index was used instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The only processing is assigning the per-trial loop index and broadcasting it across time bins for that trial.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. The AI’s justification is that the decoder needs the within-session trial count, and the extraction loop already defines that ordering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, combined with each trial’s start and end times from `position/timestamps`.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The notes say previous trial outcome should be based on whether reward delivery occurred in the preceding trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial gets `0`. For every later trial, the code carries forward `prev_trial_rewarded`, where reward on the previous trial is determined by asking whether any reward timestamp falls between that trial’s start and end times.

ii.
```python
prev_trial_rewarded = 0
...
prev_outcome = np.float32(prev_trial_rewarded)
input_arr[3, :] = prev_outcome
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
prev_trial_rewarded = int(was_rewarded)
```

iii. The notes justify this as the binary omitted/rewarded label requested by the instructions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial position plus reward-zone coordinates inferred from session metadata in `identifier` via `scene`, not from the behavioral `reward_zone` signal. The reward zone for each trial is assigned by `get_reward_zones(scene, n_trials, change_trial=30)`.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
...
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
trial_pos = position[s:e]
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say the AI believed the scene string plus the reference code’s switch-trial logic was enough to reconstruct reward zone location, including cross-environment switches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance is computed to the current trial’s reward zone: negative before the zone, zero inside it, positive after it.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end
    return dist
```

iii. The notes explicitly describe this as the intended reward-relative distance variable.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code uses manual thresholding into the requested 7 categories: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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

iii. The notes say this was chosen to match the decoder specification exactly, especially the special “0 cm” bin.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned framewise because both come from the same per-trial slice `[s:e]`.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say behavior and imaging are already synchronized at the frame level.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavioral `position` time series.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
```

iii. The notes describe this as the direct VR corridor position signal.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code slices the per-trial position trace and discretizes it; there is no interpolation or smoothing.

ii.
```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The notes say absolute position should be a straightforward discretization of the corridor coordinate.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses 5 equal-width bins spanning `0` to `450` cm, implemented with `np.linspace(0, TRACK_LENGTH, 6)` and clipping into bin indices `0..4`.

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The notes explicitly defend this as matching the instruction “5 equal-sized bins”.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by using the same per-trial frame indices as neural activity.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. The notes say the behavior and imaging streams are already synchronized, so shared slicing is enough.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The notes identify `lick` as the raw lick sensor stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code applies a lick-sensor error rule: if more than 35% of frames in a trial have `lick > 2`, the whole trial is zeroed. Otherwise values above 1 are clipped to 1, and the result is binarized as `trial_lick > 0`.

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The notes claim this comes from the reference code’s lick-cleaning heuristic, although the notes sometimes describe the bad-trial action as setting licks to `NaN` while the implemented code sets them to zero.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same `[s:e]` slice used for neural data.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
```

iii. The notes say no additional temporal alignment was needed.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` string, parsed into `scene`, then converted to trialwise reward-zone labels by `get_reward_zones()`. The behavioral `reward_zone` stream is loaded but not used for this output.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_loc = rz_label_to_idx(rz_labels[i])
```

iii. The notes justify this by saying the scene string encodes fixed and switched reward-zone configurations closely enough to reproduce the reference logic.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code maps each trial to a label `A/B/C` using scene parsing plus a fixed switch trial of 30, then converts that label to categorical indices `0/1/2` and broadcasts it across the trial.

ii.
```python
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
def rz_label_to_idx(label):
    mapping = {'A': 0, 'B': 1, 'C': 2}
    return mapping.get(label, -1)
...
output_arr[4, :] = rz_loc
```

iii. The trajectory shows the AI noticed missing cases for cross-environment scene names and patched `get_reward_zones()` to cover them.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, together with each trial’s start and end times from `position/timestamps`.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. The notes describe reward outcome as whether any reward event occurred in the current trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code checks whether any reward timestamp lies between the current trial’s first and last timestamps, converts that Boolean to `0/1`, and broadcasts it across the trial.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
...
reward_outcome = int(was_rewarded)
output_arr[5, :] = reward_outcome
```

iii. The notes justify this as the per-trial rewarded-versus-omitted label requested by the decoder task.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code crops neural and behavioral streams to their common minimum length, skips sessions with too few cells or too few valid trials, skips malformed trials with `e <= s` or `<2` frames, defaults unrecognized scenes to reward zone A, and converts suspected lick-sensor failures to zero-lick trials. It does not implement the reference solution’s Viterbi-style handling of missing reward-zone observations.

ii.
```python
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    ...
if n_cells_iscell < 2:
    return None
...
if e <= s or (e - s) < 2:
    continue
...
print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
...
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
```

iii. The notes frame these as pragmatic fixes to preserve a decodable dataset and highlight the multi-plane length mismatch and cross-environment scene parsing as edge cases the AI found and resolved.

## 13-a. What are the most time-consuming steps of the code?

i. The notes and trajectory identify dF/F computation plus speed-correlation interneuron detection as the main bottleneck, with full NWB I/O across all sessions also being substantial.

ii.
```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
...
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
```

iii. `CONVERSION_NOTES.md` explicitly reports runtime improvements after vectorizing interneuron detection and estimates full conversion time from that bottleneck.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining vectorization opportunities are the per-trial dF/F loop, the per-trial reward detection and array construction loop, and the repeated scalar broadcasting when building `input_arr` and `output_arr`.

ii.
```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
...
for i in range(n_trials):
    ...
    was_rewarded = detect_reward_in_trial(...)
    input_arr[1, :] = env_type
    output_arr[4, :] = rz_loc
```

iii. The AI only vectorized the correlation step. The notes call out that change specifically, implying the remaining trial loop still dominates some work.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly scans trials to build dF/F, then scans trials again to build outputs; it recomputes reward inclusion for every trial separately; and it repeatedly broadcasts per-trial scalars over all frames.

ii.
```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(...)
...
for i in range(n_trials):
    was_rewarded = detect_reward_in_trial(...)
    input_arr[1, :] = env_type
    input_arr[2, :] = trial_number
    input_arr[3, :] = prev_outcome
    output_arr[4, :] = rz_loc
    output_arr[5, :] = reward_outcome
```

iii. This repetition is implicit in the implementation and consistent with the notes’ focus on trialwise processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded work is loading `Fluorescence` and `Neuropil`, computing dF/F, and running interneuron detection even though the saved neural data are deconvolved events. The script also loads several variables it never uses downstream (`rzone_cumul`, `trial_num`, `scanning`, `reward_data`, `plane_idx`, `plane_assignment`) and broadcasts per-trial labels across all time bins instead of storing them once.

ii.
```python
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
...
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
...
rzone_cumul = bts['reward_zone/data'][:]
trial_num = bts['trial number/data'][:]
scanning = bts['scanning/data'][:]
reward_data = bts['Reward/data'][:]
```

iii. The notes justify the dF/F work as reference fidelity for interneuron exclusion, but in the final dataset the computed dF/F itself is thrown away and the extra loaded behavioral fields are unused.
