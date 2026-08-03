# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads available NWB experiments by listing files in the data directory, extracting `ophys_experiment_id`s from filenames, joining those IDs against `ophys_experiment_table.csv`, keeping only `active_behavior` rows, and then dropping `MESO.1` experiments. Each retained experiment is then loaded with `BehaviorOphysExperiment.from_nwb_path(...)`.

ii. ```python
def get_experiment_table():
    nwb_files = os.listdir(NWB_DIR)
    all_exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files if f.endswith('.nwb')]
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(all_exp_ids)]
    exp_table = exp_table[exp_table['behavior_type'] == 'active_behavior']
    exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
    return exp_table

def load_experiment(exp_id):
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{exp_id}.nwb')
    return BehaviorOphysExperiment.from_nwb_path(nwb_path)
```

iii. The justification in `CONVERSION_NOTES.md` and trajectory step 68 is that passive sessions should be excluded, and Multiscope (`MESO.1`) should also be excluded because those sessions are 11 Hz while Scientifica sessions are 31 Hz, and the agent wanted one fixed bin size across sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from `dataset.metadata`. The script builds a unique `subjects` list and a per-session `subject_idx`.

ii. ```python
meta = session_data['metadata']
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The notes explicitly say subjects correspond to mice, and the trajectory repeatedly reports dataset size in unique mice.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as one session. It does not aggregate multiple experiments that share an `ophys_session_id`; instead, it stores the true `ophys_session_id` only as metadata.

ii. ```python
exp_ids = exp_table['ophys_experiment_id'].values

for i, exp_id in enumerate(exp_ids):
    ...
    all_neural.append(neural_trials)
    ...
    session_info.append({
        'exp_id': exp_id,
        'session_id': int(meta['ophys_session_id']),
        ...
    })
```

iii. `CONVERSION_NOTES.md` lists this as a key decision: "Each experiment = one session: Each imaging plane treated as independent session."

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. For each retained trial row, the script uses `start_time` and `stop_time` and keeps all ophys frames with `start_time <= t < stop_time`. Trials with fewer than 5 frames are discarded.

ii. ```python
trials = dataset.trials
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]

for _, trial_row in valid_trials.iterrows():
    start_time = trial_row['start_time']
    stop_time = trial_row['stop_time']

    frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
    frame_indices = np.where(frame_mask)[0]

    if len(frame_indices) < 5:
        continue
```

iii. In trajectory step 57 the agent says the temporal alignment is based on ophys timestamps and that each trial should extract frames within `[start_time, stop_time]`.

## 1-e. How are trials filtered based on quality controls?

i. The main trial filter is semantic: keep `go` and `catch`, exclude `aborted` and `auto_rewarded`. The script also skips whole experiments with fewer than 2 valid trials and skips individual segmented trials with fewer than 5 ophys frames.

ii. ```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
if len(valid_trials) < 2:
    return None

...
if len(frame_indices) < 5:
    continue
```

iii. The notes say the trial curation rules are exactly "Include Go and Catch trials, exclude Aborted and Auto-rewarded trials." The `>= 2` trials rule came from the decoder format requirement, and the `>= 5` frames rule is an extra implementation choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `dataset.events`, specifically the `events` column.

ii. ```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The notes and trajectory say the agent intentionally used detected calcium events rather than `dff_traces`, citing the methods text.

## 2-b. How is the `neural` data processed?

i. The script stacks all per-neuron event vectors into a `(n_neurons, n_timepoints)` float32 array for the session, then slices that array into per-trial matrices using the ophys frame indices. No temporal smoothing, normalization, or rebinnig is applied in the script.

ii. ```python
events_array = np.vstack(events['events'].values).astype(np.float32)
...
neural_trial = events_array[:, frame_indices]
```

iii. The recovered justification is that the Allen pipeline already produced the event traces and the paper/methods used detected calcium events directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit per-neuron quality filter in the script. The agent assumes Allen's ROI/cell filtering has already been applied upstream. The only neural QC in code is skipping experiments with fewer than 2 neurons.

ii. ```python
n_neurons = events_array.shape[0]

if n_neurons < 2:
    return None
```

iii. `CONVERSION_NOTES.md` says "No neuron filtering: ROIs already filtered by Allen pipeline."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The master time base is `dataset.ophys_timestamps`. Neural, running, pupil, image identity, and change are all represented on that time base first, then the trial uses the shared subset of frames inside the trial window. The metadata names the alignment event as `Trial start time`.

ii. ```python
ophys_ts = dataset.ophys_timestamps
...
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
...
'temporal_alignment_event': 'Trial start time',
```

iii. Trajectory step 57 says the agent chose ophys timestamps as the common clock and extracted frames inside each trial window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the native Scientifica ophys frame rate, about 31 Hz (`~32.3 ms` bins). No rebinning is applied. Instead, the agent excludes 11 Hz `MESO.1` experiments.

ii. ```python
dt = np.median(np.diff(ophys_ts))
...
exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
...
'time_bin_size': 1000.0 / 31.0,
```

iii. The notes justify this by saying the output format requires consistent time bins, so Multiscope sessions were dropped instead of being rebinned.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `dataset.stimulus_presentations`, using `stimulus_block_name`, `image_name`, `start_time`, and `end_time`.

ii. ```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
...
starts = sp_active['start_time'].values
ends = sp_active['end_time'].values
names = sp_active['image_name'].values
```

iii. The agent's notes say the image identity output should come from the change-detection stimulus block and ignore omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent collects all non-omitted image names across the dataset, sorts them into a global index, marks the matching ophys frames during each presentation, then forward-fills the last seen image across the gray-screen intervals. Any remaining `-1` frames at the start of a trial are replaced with that trial's first valid image label.

ii. ```python
for name in sp_active['image_name'].unique():
    if isinstance(name, str) and name != 'omitted':
        all_image_names.add(name)
...
for j in range(len(starts)):
    img_name = names[j]
    if not isinstance(img_name, str) or img_name == 'omitted':
        continue
    ...
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

last_img = -1
for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img

neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The notes say gray-screen periods were intentionally forward-filled with the last shown image. Trajectory steps 101-102 show the agent discovered four `-1` frames and justified replacing them with the next/first valid image label to avoid invalid classifier targets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The identity trace is built on full-session ophys timestamps and then sliced with the exact same `frame_indices` used for `neural_trial`.

ii. ```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
...
img_trial = image_idx_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The justification is the same common-clock decision: all outputs live on ophys timestamps before trial segmentation.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `dataset.stimulus_presentations['is_change']` together with each presentation's `start_time` and `end_time`.

ii. ```python
change_sp = sp_active[sp_active['is_change'] == True]
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

iii. The notes map `stimulus_presentations.is_change` directly to the `image_change` output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script creates a session-length zero vector and sets it to `1` during any change-image presentation interval in the change-detection block.

ii. ```python
def get_change_at_ophys(sp_active, ophys_ts):
    n_tp = len(ophys_ts)
    change = np.zeros(n_tp, dtype=np.int32)

    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The notes say this output should be binary at ophys timestamps, with `1` during change presentation.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary; the categories are simply `0` and `1`, named `no_change` and `change`.

ii. ```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The agent did not describe additional thresholding because the source variable is already boolean.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like the other outputs, the full-session change vector is defined on `ophys_timestamps` and then indexed by each trial's `frame_indices`.

ii. ```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
...
change_trial = change_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The common-clock justification is stated throughout the notes and trajectory.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed['timestamps']` and `dataset.running_speed['speed']`.

ii. ```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. The notes identify running speed as a ~60 Hz stream that must be resampled to the ophys clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script linearly interpolates running speed onto the ophys timestamps, pools all session values from pass 1, computes global percentile cut points, and digitizes each session's ophys-aligned trace in pass 2.

ii. ```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)

all_running_values.append(session_data['running_full'])
...
all_running_concat = np.concatenate(all_running_values)
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
...
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. Trajectory step 68 says the agent explicitly chose dataset-level percentile bins for consistency across experiments.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins are formed using the 0th, 20th, 40th, 60th, 80th, and 100th percentiles over all included running-speed samples. `np.digitize` with the interior four edges produces categories `0..4`.

ii. ```python
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` calls this "Global percentile binning."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running-speed trace is first resampled to ophys timestamps and is then cut into trials with the same `frame_indices` as the neural data.

ii. ```python
running_full = resample_to_ophys(...)
...
running_trial = running_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The justification is the same ophys-timestamp alignment decision.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It is derived from `dataset.eye_tracking['pupil_width']` and `dataset.eye_tracking['timestamps']`.

ii. ```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. The notes and trajectory identify `pupil_width` as the pupil-size variable.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script drops `NaN` pupil samples, linearly interpolates the remaining values onto the ophys timestamps when there are more than 10 valid samples, and otherwise fills the whole session with `NaN`. Any error while reading eye tracking is swallowed and also becomes all-`NaN`.

ii. ```python
try:
    eye = dataset.eye_tracking
    pupil_raw = eye['pupil_width'].values
    pupil_ts = eye['timestamps'].values
    valid_mask = ~np.isnan(pupil_raw)
    if valid_mask.sum() > 10:
        pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
    else:
        pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
except:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
```

iii. The justification in the notes is only that pupil is a ~60 Hz signal that needs resampling; the trajectory also notes that pupil has NaNs from blinks and missing samples.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global percentile edges are computed from all non-NaN pupil values seen in pass 1. In pass 2, only valid samples are digitized; missing samples remain at the array's default value `0`, which doubles as the lowest bin.

ii. ```python
all_pupil_concat = np.concatenate(all_pupil_values)
pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))

valid_pupil_mask = ~np.isnan(pupil_full)
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
if valid_pupil_mask.any() and pupil_edges is not None:
    pupil_binned[valid_pupil_mask] = np.clip(
        np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
    ).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` again describes this as global percentile binning, but it does not separately justify the choice to encode missing pupil values as `0`.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil trace is aligned by interpolation onto `ophys_timestamps`, then indexed per trial with the same `frame_indices` as the neural data.

ii. ```python
pupil_full = resample_to_ophys(..., ophys_ts)
...
pupil_trial = pupil_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The justification is the same shared-ophys-clock design used for all outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `dataset.trials`.

ii. ```python
def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. The notes explicitly map `trials.hit/miss/FA/CR` to `output[4]`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script converts the four booleans into integer labels `0..3` and repeats the chosen label across every time point in the trial.

ii. ```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
...
output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
```

iii. The notes describe trial outcome as "Static per trial (repeated)" because the decoder format expects the trial output to be time-aligned with the rest of the trial arrays.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases ad hoc: omitted images are skipped; eye-tracking failures or too-few valid pupil samples become all-`NaN`; missing pupil bins later become category `0`; any image-identity `-1` frames at trial start are replaced with the first valid image; very short trials are dropped; and sessions with too few neurons or valid trials are skipped.

ii. ```python
if not isinstance(img_name, str) or img_name == 'omitted':
    continue

if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(...)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
except:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)

pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
...
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid

if len(frame_indices) < 5:
    continue
```

iii. Trajectory steps 101-102 show the agent found four invalid image labels in the generated pickle and justified replacing them with the next/first valid image. The notes also present missing pupil data and frame-rate mismatches as issues to resolve pragmatically rather than preserve explicitly.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading each NWB experiment, materializing full-session arrays for events/running/pupil/stimulus labels in pass 1, concatenating global running and pupil values to compute percentiles, re-looping through all trials in pass 2, and serializing the very large pickle.

ii. ```python
for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    ...
    session_data = extract_session_data(dataset, temp_image_to_idx)
    ...
all_running_concat = np.concatenate(all_running_values)
all_pupil_concat = np.concatenate(all_pupil_values)
...
for i, exp_id in enumerate(exp_ids):
    ...
    neural_trials, output_trials = segment_trials(session_data, running_edges, pupil_edges)
...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The trajectory explicitly discusses runtime, estimates full conversion time, and motivates the two-pass restructuring around performance.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-stimulus loop in `get_image_at_ophys`, the per-timepoint forward-fill loop in that same function, the `iterrows()` loop in `get_change_at_ophys`, and the `iterrows()` loop over trials in `segment_trials`.

ii. ```python
for j in range(len(starts)):
    ...
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img

for _, row in change_sp.iterrows():
    ...

for _, trial_row in valid_trials.iterrows():
    ...
```

iii. The notes mention vectorization as a concern, and the trajectory shows the agent optimizing some parts but leaving these loops in place.

## 9-c. What processing does the code repeat multiple times?

i. The script computes a temporary image mapping during pass 1 and then recomputes `image_idx_full` with the final global mapping in pass 2. It also duplicates digitization logic in both `segment_trials` and `plot_processing`.

ii. ```python
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)
...
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
...
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

iii. Trajectory step 68 explains the two-pass design and acknowledges that some work is intentionally repeated to get dataset-wide percentile bins and a stable global image mapping.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes processing that is only for debugging or intermediate bookkeeping: `plot_processing(...)` and the saved PNGs, the temporary image mapping from pass 1 that is overwritten in pass 2, and full-session continuous running/pupil/image arrays that are only used to derive final trial-level discrete outputs.

ii. ```python
if args.show_processing and len(all_neural) <= 2:
    plot_processing(...)

temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
...
session_data['image_idx_full'] = get_image_at_ophys(...)
```

iii. The notes describe the plots as visual sanity checks, and the trajectory shows the temporary mapping and stored full-session intermediates were accepted as a pragmatic way to simplify the implementation.
