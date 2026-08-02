# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all sessions by globbing every NWB file under `data/sub-*/*.nwb`, then processes each file as one session with `process_session()`. Within each NWB file it reads behavioral time series from `processing/behavior/BehavioralTimeSeries` and imaging data from `processing/ophys`, including deconvolved events, fluorescence, neuropil, subject/session metadata, and timestamps.

ii. 
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
```

```python
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    bts = f['processing/behavior/BehavioralTimeSeries']
    position = bts['position/data'][:]
    ...
    ophys = f['processing/ophys']
    iscell = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
```

iii. `CONVERSION_NOTES.md` says the NWB files are the primary source and lists the specific NWB paths and datasets explored in Steps 1-2. The trajectory shows the agent decided to treat each NWB as a session and use the precomputed NWB contents directly.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file’s `general/subject/subject_id` field. The final dataset stores a sorted unique subject list and a `subject_idx` per processed session.

ii. 
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. In the notes, Step 2 records 11 subjects from the NWB directory structure, and Step 9 checks that the converted data still contain 11 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. The outer loop over `nwb_files` creates one session entry in `neural`, `input`, and `output` for each valid processed NWB file.

ii.
```python
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. The notes repeatedly refer to “152 sessions” and describe each file `sub-*_ses-*_behavior+ophys.nwb` as a session file.

## 1-d. How are the data split into trials?

i. Trials are defined by the binary `trial_start` and `teleport` behavioral flags. The agent finds all indices where those flags are positive, pairs them in order, and slices each trial as `[trial_start_ind : teleport_ind)`.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
...
s = trial_start_inds[i]
e = teleport_inds[i]
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. In trajectory steps 132-136, the agent explicitly justified this slicing as matching the effective reference behavior because the NWB flags are offset relative to the original code’s `start-1:stop-1` indexing.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only dropped if their bounds are invalid or too short: `e <= s` or length `< 2`. Lick sensor corruption is not used to drop trials; instead the lick signal is zeroed within affected trials. Sessions with fewer than two valid trials are skipped.

ii.
```python
if e <= s or (e - s) < 2:
    ...
    continue
...
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
...
if valid_trial_count < 2:
    return None
```

iii. The notes describe trial curation as bounded to trial windows plus lick-error handling, and the code follows that directly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The stored `neural` data come from the NWB deconvolved calcium events in `processing/ophys/Deconvolved/plane*/data`, after ROI filtering. Raw fluorescence and neuropil are only used to compute dF/F for interneuron detection, not as the final neural signal.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
...
neural_data = deconv_cells[final_cell_mask]
```

iii. Step 5 of the notes states the key decision explicitly: “Neural data = deconvolved events,” because the paper’s decoder used deconvolved events and the NWB already contains them.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, truncates neural/behavioral streams to a shared minimum length, filters ROIs with `iscell`, computes per-trial dF/F from fluorescence and neuropil to identify interneurons, removes those interneurons, and then stores the remaining deconvolved event traces trial-by-trial without further rebinning.

ii.
```python
deconv_all = np.concatenate(deconv_list, axis=1)
...
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
deconv_all = deconv_all[:n_timepoints_total]
...
cell_mask = iscell[:, 0].astype(bool)
...
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
neural_data = deconv_cells[final_cell_mask]
```

iii. The notes justify using deconvolved events to match the paper’s decoder, and computing dF/F only for interneuron exclusion. They also justify keeping all frames rather than applying a speed threshold because the task asks the decoder to predict speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROI filtering is two-stage: keep only `iscell[:, 0] == 1`, then exclude putative interneurons whose dF/F-speed correlation exceeds `0.5`. Sessions with fewer than two remaining cells are discarded.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
...
if n_final_cells < 2:
    return None
```

iii. `CONVERSION_NOTES.md` lists these as the neuron curation rules taken from the reference code and paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural traces are aligned to trial start. For each trial, the agent slices neural frames from the trial-start index to the teleport index, and `input[0]` is then defined as time since that trial start.

ii.
```python
s = trial_start_inds[i]
e = teleport_inds[i]
trial_neural = neural_data[:, s:e]
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. Step 5 says “Trial alignment = trial start,” and trajectory steps 132-136 document the explicit alignment check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging frame as the time bin. For single-plane sessions this is the NWB imaging rate; for two-plane sessions the code divides the reported imaging rate by the number of planes. No temporal rebinning is applied.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. The notes say “Time bin = imaging frame” and report about 64.5 ms bins at about 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the trial duration implied by the trial-start and teleport boundaries plus the inferred effective frame rate. The code does not use the raw timestamps directly for this variable.

ii.
```python
s = trial_start_inds[i]
e = teleport_inds[i]
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The notes describe this as “(t - trial_start) / imaging_rate” and treat the imaging frame as the native bin size.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The agent computes a regularly spaced ramp starting at zero with one value per neural frame: `0, 1/rate, 2/rate, ...`. It assumes constant sampling within each session.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The justification in the notes is that all streams are already synchronized at the imaging frame rate, so a frame-count-based time axis is sufficient.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created with exactly `n_t = e - s` samples for the same trial slice used for `trial_neural`, so it is frame-by-frame aligned and starts at the first neural sample of the trial.

ii.
```python
trial_neural = neural_data[:, s:e]
n_t = e - s
input_arr = np.zeros((4, n_t), dtype=np.float32)
input_arr[0, :] = time_from_start
```

iii. This is the same alignment choice the agent documented when validating trial slicing in the trajectory.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment/data` time series within each trial.

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
```

iii. The mapping table in Step 5 says environment comes from the environment behavioral time series and should be represented as a per-trial scalar.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent takes the median nonnegative environment value within a trial, casts it to float, and broadcasts it across all time bins in that trial.

ii.
```python
env_type = np.float32(env_val)
input_arr[1, :] = env_type
```

iii. The notes justify this as a per-trial contextual variable rather than a time-varying signal.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although `trial number/data` is loaded, the final `trial_number` input is actually derived from the loop index `i`, not from the raw NWB trial-number variable.

ii.
```python
trial_num = bts['trial number/data'][:]
...
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. The notes originally planned to use the trial-number variable, but the implemented code instead uses the enumerated trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation of the raw time series is performed. The code simply uses the current loop counter as a 0-indexed per-trial scalar and broadcasts it across the trial.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. There is no separate justification in the notes; this appears to be an implementation simplification.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the NWB reward timestamps, by checking whether the previous trial contained any reward delivery event.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
prev_outcome = np.float32(prev_trial_rewarded)
```

iii. Step 5 says previous trial outcome should be computed from reward detection in the previous trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is initialized as 0. For each subsequent trial, the code carries forward the prior trial’s reward outcome as a binary scalar and broadcasts it across the trial.

ii.
```python
prev_trial_rewarded = 0
...
prev_outcome = np.float32(prev_trial_rewarded)
...
prev_trial_rewarded = int(was_rewarded)
```

iii. The notes call this out explicitly, including the first-trial default of 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the trial position samples plus reward-zone start/end coordinates inferred from the session scene name and switch rule, not from the stored `reward_zone/data` time series.

ii.
```python
position = bts['position/data'][:]
...
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes say this variable should be computed from position and reward-zone coordinates using the reference `get_reward_zones` logic.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code computes signed distance to the nearest reward-zone boundary: negative before the zone, zero inside the zone, positive after the zone.

ii.
```python
before = position < rz_start
inside = (position >= rz_start) & (position <= rz_end)
after = position > rz_end
dist[before] = position[before] - rz_start
dist[inside] = 0.0
dist[after] = position[after] - rz_end
```

iii. The notes explicitly describe the same signed-distance convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into 7 categories using the task-specified bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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

iii. Step 5’s discretization table matches these thresholds exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `trial_pos = position[s:e]` for the same trial slice used for `trial_neural`, so it is time-locked frame-by-frame to the neural data.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
output_arr[0, :] = dist_to_rz_binned
```

iii. The agent consistently used the same `[s:e]` slice for all time-varying trial signals.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position/data` time series within each trial.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The notes map absolute position directly from the position behavioral time series.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent slices position to the trial window and then discretizes those raw centimeter positions. There is no smoothing or interpolation.

ii.
```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The notes describe this output as direct from position with only discretization added to satisfy the decoder format.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is binned into 5 equal-width bins spanning the 450 cm track, using `np.digitize` on linearly spaced edges.

ii.
```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. This follows the decoder task requirement for 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is generated from the same per-trial frame slice `position[s:e]` as the neural trial, so each position bin corresponds to one neural frame.

ii.
```python
trial_neural = neural_data[:, s:e]
trial_pos = position[s:e]
output_arr[1, :] = pos_binned
```

iii. The notes’ processing pipeline emphasizes common trial-window alignment across all streams.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick/data` time series, which the agent interprets as a cumulative lick count signal.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The notes identify `lick` as coming from the behavioral time series and mention the cumulative-count artifact rule from the paper code.

## 9-b. What processing is involved in computing `output` *Lick*?

i. For each trial, if more than 35% of samples have `lick > 2`, the entire trial’s lick vector is set to zero. Otherwise values greater than 1 are clipped to 1, then all positive values are binarized.

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The notes justify this as adapting the reference lick-cleaning rule to a decoder output that cannot contain NaNs. They explicitly note that the original code used NaN-like invalidation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced over the same `[s:e]` trial window as the neural data, so the binary lick output has one sample per neural frame.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
output_arr[3, :] = lick_binned
```

iii. This is consistent with the agent’s general trial-window alignment choice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session `identifier` scene string, mapped through `get_reward_zones()` with a hard-coded switch trial of 30 to assign zone labels A/B/C per trial.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
rz_loc = rz_label_to_idx(rz_labels[i])
```

iii. The notes repeatedly justify this by citing the reference `behavior.get_reward_zones()` logic and the standard switch trial.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses fixed-location and switch scenes, including cross-environment switch names, assigns a zone label per trial, converts A/B/C to 0/1/2, and broadcasts that scalar across the trial.

ii.
```python
elif 'A_to' in scene and scene[-1] == 'B':
    rz_labels[:change_trial] = 'A'
    rz_labels[change_trial:] = 'B'
...
mapping = {'A': 0, 'B': 1, 'C': 2}
output_arr[4, :] = rz_loc
```

iii. Trajectory steps 107-115 show the agent discovering and fixing cross-environment scene parsing specifically to match the reference code behavior.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward/timestamps` event series in the behavioral data, using the trial timestamps to decide whether any reward occurred during that trial.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
reward_outcome = int(was_rewarded)
```

iii. The notes’ variable mapping explicitly says reward outcome comes from reward detection within the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code treats reward outcome as binary: 1 if any reward timestamp falls within the trial window, otherwise 0. That value is then broadcast across all time bins of the trial.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
...
output_arr[5, :] = reward_outcome
```

iii. The notes justify this as the per-trial rewarded-versus-omitted decoder label.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles data irregularities with simple heuristics: truncate neural and behavioral streams to the shorter length, skip malformed/short trials, avoid division by zero in dF/F with an epsilon floor, default missing environment values to 0, zero out lick-corrupted trials, and default unrecognized scenes to reward zone A with a warning.

ii.
```python
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
...
if e <= s or (e - s) < 2:
    continue
...
abs_baseline[abs_baseline < 1e-10] = 1e-10
...
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
...
print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
```

iii. The notes and trajectory mention the multi-plane length mismatch fix and the scene-parser fallback explicitly. The rest is implicit from the code.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading every NWB file, computing per-trial dF/F across all cells, and the session-level interneuron-detection pass. The agent specifically identified interneuron detection as the bottleneck.

ii.
```python
for i in range(n_trials):
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
```

iii. In trajectory steps 79 and the Step 7 notes, the agent says the bottleneck was interneuron detection and then vectorized that part.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining non-vectorized loop is the per-trial dF/F computation loop. The per-trial extraction/output-construction loop also repeats scalar broadcasting and reward checks that could be partially vectorized or precomputed.

ii.
```python
for i in range(n_trials):
    ...
    dff_full[:, s:e] = compute_dff_trial(...)
...
for i in range(n_trials):
    ...
    input_arr[1, :] = env_type
    output_arr[5, :] = reward_outcome
```

iii. The agent already vectorized the correlation step, so the remaining obvious opportunities are the two trial loops.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly slices trial windows across many arrays, repeatedly checks reward timestamps for each trial, repeatedly builds broadcast trial arrays, and computes dF/F for all cells even though dF/F is only used once for interneuron filtering.

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    was_rewarded = detect_reward_in_trial(...)
    ...
    input_arr = np.zeros((4, n_t), dtype=np.float32)
    output_arr = np.zeros((6, n_t), dtype=np.int64)
```

iii. This follows directly from the implementation structure; the notes do not dispute it.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several loaded or created objects are unused downstream: `rzone_cumul`, `trial_num`, `scanning`, `reward_data`, `plane_idx`, `plane_assignment`, `input_data`, `trial_scalars`, and the predeclared bin-edge constants. The computed dF/F arrays are also discarded after interneuron detection. Optional plotting also does extra work unrelated to the final pickle.

ii.
```python
rzone_cumul = bts['reward_zone/data'][:]
trial_num = bts['trial number/data'][:]
scanning = bts['scanning/data'][:]
reward_data = bts['Reward/data'][:]
plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]
plane_assignment = []
...
input_data = np.array([time_from_start], dtype=np.float32)
trial_scalars = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
```

iii. The trajectory and notes emphasize accuracy and validation rather than minimizing redundant work, so these leftovers appear to be incidental implementation artifacts.
