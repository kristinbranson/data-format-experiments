# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent did not use the Allen SDK project cache. It enumerated local NWB files under `/app/data`, read `ophys_experiment_table.csv`, filtered to `active_behavior`, excluded `equipment_name == 'MESO.1'`, and then loaded each remaining `ophys_experiment_id` directly with `BehaviorOphysExperiment.from_nwb_path()`. It used a two-pass workflow: pass 1 loaded every experiment and cached session-level arrays plus global statistics; pass 2 segmented trials and assembled the output.

ii. 
```python
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

```python
for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    ...
    session_data = extract_session_data(dataset, temp_image_to_idx)
    all_session_data.append(session_data)
```

iii. In `CONVERSION_NOTES.md`, the agent justified local NWB loading by citing the tutorials and AllenSDK loader (`BehaviorOphysExperiment.from_nwb_path`) and said the available data were a subset of the full release. In Step 10 and the trajectory, it justified excluding `MESO.1` experiments because the output format required a single time-bin size across sessions, and it preferred dropping the 11 Hz Multiscope sessions instead of downsampling the 31 Hz sessions.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values taken from each loaded experiment’s metadata. The script builds a `subject_set` map while iterating over experiments and stores subject indices per output session.

ii. 
```python
meta = session_data['metadata']
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The justification is implicit in the code and notes: `mouse_id` is treated as the subject identifier supplied by Allen metadata. The notes also report counts of unique mice from the metadata tables.

## 1-c. How are the data split into sessions?

i. The agent treated each `ophys_experiment_id` as one output session. It did not group multiple experiments with the same `ophys_session_id`; instead it iterated experiment-by-experiment, kept one session record per experiment, and only stored the `ophys_session_id` as metadata.

ii. 
```python
exp_ids = exp_table['ophys_experiment_id'].values
...
for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    ...
    all_neural.append(neural_trials)
```

```python
session_info.append({
    'exp_id': exp_id,
    'session_id': int(meta['ophys_session_id']),
    ...
})
```

iii. This decision is explicit in `CONVERSION_NOTES.md`: “Each experiment = one session.” The notes justify it by saying each imaging plane is treated as an independent session; after the later `MESO.1` exclusion, the retained Scientifica data are single-plane 31 Hz experiments.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. The agent kept go or catch trials that were not aborted and not auto-rewarded, then used each trial’s `start_time` and `stop_time` to select ophys frames. Trials remain variable-length.

ii. 
```python
trials = dataset.trials
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
```

```python
for _, trial_row in valid_trials.iterrows():
    start_time = trial_row['start_time']
    stop_time = trial_row['stop_time']
    frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
    frame_indices = np.where(frame_mask)[0]
```

iii. The notes justify this as matching the task definition of valid trials: include Go and Catch, exclude Aborted and Auto-rewarded, and align everything to ophys timestamps. The trajectory also shows the agent explored the Allen trial table and concluded that these trial fields were the right segmentation source.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring `(go | catch)`, excluding `aborted` and `auto_rewarded`, requiring at least two valid trials per experiment, and later dropping trial segments with fewer than 5 ophys frames. The code does not explicitly require non-null `change_time`.

ii. 
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
if len(valid_trials) < 2:
    return None
```

```python
frame_indices = np.where(frame_mask)[0]
if len(frame_indices) < 5:
    continue
```

iii. The notes justify the main trial curation rule directly from the task and the paper methods. The extra session-level and minimum-frame filters are not strongly justified in the notes beyond satisfying decoder format constraints. The trajectory later adds the broader “at least two trials per session” requirement from the target data format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derived neural data from `dataset.events['events']`, not from `dataset.dff_traces`.

ii. 
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The justification is explicit in the notes: the agent read `methods.txt`, concluded the paper used detected calcium events, and wrote “Use `events` (calcium events from FastLZeroSpikeInference) NOT `dff_traces`.”

## 2-b. How is the `neural` data processed?

i. The agent stacks the per-cell event traces from one experiment into a neuron-by-time matrix, casts to `float32`, and then slices trial windows from that matrix. It does not merge multiple experiments from the same `ophys_session_id`.

ii. 
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
...
neural_trial = events_array[:, frame_indices]
```

iii. The notes justify this by saying the NWB files already contain processed event traces and ROI filtering has already been applied by the Allen pipeline, so no further neural preprocessing is needed before trial extraction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no per-neuron quality filter beyond whatever Allen already applied upstream. The only explicit checks are dropping whole experiments with fewer than 2 neurons and relying on the existing Allen ROI filtering.

ii. 
```python
n_neurons = events_array.shape[0]
if n_neurons < 2:
    return None
```

iii. `CONVERSION_NOTES.md` says ROI filtering was already applied by the Allen pipeline and that no additional neuron filtering was needed. The `n_neurons < 2` rule is a decoder-format safeguard rather than a biological QC rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys clock and then segmented by trial `start_time` and `stop_time`. Each trial’s neural matrix contains all ophys frames whose timestamps satisfy `start_time <= t < stop_time`.

ii. 
```python
ophys_ts = dataset.ophys_timestamps
...
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. The notes repeatedly justify temporal alignment as “based on ophys timestamps,” matching the task instruction. The agent chose trial-start/stop segmentation rather than a fixed window around `change_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent kept the native Scientifica frame rate and hard-coded the metadata time bin as `1000/31` ms, after excluding the 11 Hz Multiscope sessions. It did not rebin neural data within those retained sessions.

ii. 
```python
exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
```

```python
'metadata': {
    ...
    'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
    ...
}
```

iii. The justification is explicit in the notes and trajectory: the agent found mixed 31 Hz and 11 Hz sessions, concluded the required format needed one common bin size, and decided it was better to exclude the 11 Hz sessions than to downsample the 31 Hz sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations` restricted to `stimulus_block_name == 'change_detection_behavior'`, using the `image_name`, `start_time`, and `end_time` columns.

ii. 
```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
...
starts = sp_active['start_time'].values
ends = sp_active['end_time'].values
names = sp_active['image_name'].values
```

iii. The notes justify this as the direct stimulus stream for the behavior block. The agent preferred framewise image labels from stimulus presentations rather than reconstructing them from the trial table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent collected all non-`omitted` image names globally, assigned integer codes, labeled ophys frames that fell inside each stimulus presentation, then forward-filled the last image through the gray periods. During trial segmentation it also replaced any remaining `-1` values at the front of a trial with the first valid image in that trial.

ii. 
```python
for name in sp_active['image_name'].unique():
    if isinstance(name, str) and name != 'omitted':
        all_image_names.add(name)
...
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
```

```python
for j in range(len(starts)):
    ...
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

last_img = -1
for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img
```

```python
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The justification is explicit in the notes: “Image identity during gray screen: Forward-filled with last shown image.” The trajectory also records that the agent saw some trials begin before the first stimulus frame and chose to backfill those trial-leading `-1` values with the next valid image code.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The agent first builds a full-session image-identity vector on the ophys timestamp grid, then slices that vector with the same per-trial ophys frame indices used for neural data.

ii. 
```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
...
img_trial = image_idx_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The justification is that both neural and image outputs are represented on the same ophys time base before trial segmentation, so using the same `frame_indices` guarantees alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `dataset.stimulus_presentations` in the active behavior block, using the boolean `is_change` flag together with each presentation’s `start_time` and `end_time`.

ii. 
```python
change_sp = sp_active[sp_active['is_change'] == True]
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

iii. The notes justify this by treating the stimulus table as the authoritative description of when a changed image was actually on screen. The sanity checks say the signal was “correct at change presentations.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent creates a full-session binary vector initialized to 0 and sets it to 1 during each stimulus presentation whose `is_change` flag is true. This marks only the changed image presentation itself, not the following gray interval.

ii. 
```python
def get_change_at_ophys(sp_active, ophys_ts):
    change = np.zeros(n_tp, dtype=np.int32)
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The notes justify this only briefly: “Image change signal correct at change presentations.” The agent’s stated interpretation was that the binary signal should indicate the actual changed-image flash.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous value is thresholded. The agent directly represents image change as a binary categorical variable with values `0` and `1`.

ii. 
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The justification is implicit in the task itself, which asked for a binary image-change output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The full-session change vector is built on ophys timestamps and then sliced with the same `frame_indices` used for the trial neural matrices.

ii. 
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
...
change_trial = change_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The justification is the same as for image identity: shared ophys timestamps are used as the common alignment grid before trial segmentation.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed`, specifically its timestamp and speed columns.

ii. 
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. The notes describe `dataset.running_speed` as the AllenSDK locomotion stream and say it needs resampling to ophys timestamps.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent linearly interpolates running speed onto the ophys timestamp grid with `np.interp`, pools the full-session interpolated values across all retained experiments, computes global quintile edges with `np.percentile`, and bins each timepoint with `np.digitize`.

ii. 
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)
```

```python
all_running_values.append(session_data['running_full'])
...
all_running_concat = np.concatenate(all_running_values)
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
...
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The notes explicitly justify “Global percentile binning” across experiments and say running speed should be resampled to ophys timestamps via linear interpolation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is converted into 5 percentile bins shared across all retained experiments. The labels are `bin_0` through `bin_4`.

ii. 
```python
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

```python
[f'bin_{i}' for i in range(5)]
```

iii. The notes explicitly say the agent chose five global percentile bins to match the decoder specification.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to the full ophys timestamp grid, then trial slices are taken with the same per-trial frame indices used for neural data.

ii. 
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
...
running_trial = running_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes justify this as standard resampling of a higher-rate behavioral stream onto the imaging time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using `pupil_width` and `timestamps`. The code filters only `NaN` pupil widths; it does not use `likely_blink`.

ii. 
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
valid_mask = ~np.isnan(pupil_raw)
```

iii. The notes say the source variable is `dataset.eye_tracking.pupil_width` and treat it analogously to running speed, but they do not mention any blink-based exclusion in the implemented code.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent interpolates non-NaN `pupil_width` values to the ophys timestamp grid with `np.interp`, pools all non-NaN full-session values across experiments, computes global quintile edges with `np.percentile`, and bins valid timepoints with `np.digitize`. Missing pupil values remain in bin 0 because `pupil_binned` is initialized to zeros.

ii. 
```python
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
```

```python
valid_pupil = session_data['pupil_full'][~np.isnan(session_data['pupil_full'])]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil)
...
pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
```

```python
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
if valid_pupil_mask.any() and pupil_edges is not None:
    pupil_binned[valid_pupil_mask] = np.clip(
        np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
    ).astype(np.int32)
```

iii. The notes justify global percentile binning and ophys-grid interpolation, but the specific implementation choice to ignore `likely_blink` is not justified in the notes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 global percentile bins with labels `bin_0` through `bin_4`. Missing values fall into the default zero bin.

ii. 
```python
pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
...
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
```

iii. The justification is the same as for running speed: match the decoder requirement for five percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil width is interpolated onto the ophys timestamp grid before trial segmentation, and then the same trial `frame_indices` are used for both pupil bins and neural traces.

ii. 
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
...
pupil_trial = pupil_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes justify this by the same logic used for running speed: resample behavioral signals to the ophys time base, then slice trials.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
TRIAL_OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']

def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. The trajectory shows the agent inspected the trial table and noted these were the mutually exclusive valid-trial outcomes. The notes map them directly to the decoder output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent converts each trial outcome to an integer code 0 to 3 and repeats that code across every frame in the trial, producing a time-varying row that is constant within trial.

ii. 
```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
```

iii. The notes explicitly list “Trial outcome as time-varying: Repeated constant value across trial timepoints” as a key decision.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases: failed experiment loads are skipped; missing or insufficient pupil data produce all-`NaN` pupil arrays; experiments with too few neurons or trials are skipped; very short trial windows are skipped; omitted or pre-first-stimulus image labels are repaired by filling with a valid image code; missing pupil bins default to 0 because the binned array starts at zero.

ii. 
```python
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    print(f'  ERROR loading experiment {exp_id}: {e}')
    all_session_data.append(None)
    continue
```

```python
if n_neurons < 2:
    return None
...
if len(valid_trials) < 2:
    return None
...
if len(frame_indices) < 5:
    continue
```

```python
except:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
...
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The notes and trajectory justify these as pragmatic safeguards: keep the conversion running, satisfy decoder-format constraints, and repair a small number of trials that began before the first labeled stimulus frame.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is pass 1, where every experiment is loaded from NWB and full-session arrays are extracted. The notes report roughly 24 minutes for the full conversion and the trajectory explicitly says pass 1 dominated runtime while pass 2 was much faster.

ii. 
```python
for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    ...
    session_data = extract_session_data(dataset, temp_image_to_idx)
```

```python
print('\n=== Pass 1: Loading data and collecting global statistics ===')
...
print(f'Pass 1 completed in {time.time() - total_start:.1f}s')
```

iii. The agent’s trajectory explicitly identifies loading NWB experiments and extracting full-session data as the bottleneck. It later restructured the code so each NWB file was loaded only once.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python-level loops remain: iterating over all stimulus presentations to assign image identity, a second loop to forward-fill image codes frame-by-frame, iterating over change presentations to make the change signal, and iterating over valid trials during segmentation. These are the main obvious vectorization opportunities.

ii. 
```python
for j in range(len(starts)):
    ...
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img
```

```python
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

```python
for _, trial_row in valid_trials.iterrows():
    ...
    neural_trials.append(neural_trial)
    output_trials.append(output_trial)
```

iii. The agent did not explicitly enumerate these loops in the notes, but the trajectory shows it was concerned about speed and rewrote the script to avoid reloading NWB files. These remaining loops appear to have been accepted as good enough once I/O dominated runtime.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats image-identity work. In pass 1 it builds a temporary image map and computes `image_idx_full`; in pass 2 it recomputes `image_idx_full` again using the final global image map. The plotting code also recomputes binned running and pupil signals even though the segmented outputs already exist.

ii. 
```python
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)
```

```python
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

```python
running_binned = np.clip(np.digitize(session_data['running_full'], running_edges[1:-1]), 0, 4)
...
pupil_binned[valid_p] = np.clip(np.digitize(session_data['pupil_full'][valid_p], pupil_edges[1:-1]), 0, 4)
```

iii. The notes justify the two-pass design as necessary for global image and percentile mappings, but they do not explicitly discuss the repeated image-label computation as an inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded work is full-session bookkeeping kept only to enable pass 2: `sp_active`, `dt`, and the initial `image_idx_full` computed with a temporary image map. Optional plotting also recomputes continuous and binned traces purely for visualization. None of that extra work is part of the final saved dataset.

ii. 
```python
return {
    'ophys_ts': ophys_ts,
    ...
    'image_idx_full': image_idx_full,
    ...
    'dt': dt,
    'sp_active': sp_active,
}
```

```python
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

```python
if args.show_processing and len(all_neural) <= 2:
    plot_processing(exp_id, session_data, running_edges, pupil_edges, neural_trials, output_trials)
```

iii. The agent’s justification is indirect: it wanted a two-pass conversion with global mappings and visual QC plots. The extra intermediate products are byproducts of that design rather than outputs needed downstream.
