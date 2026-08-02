# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from local NWB files in `/app/data`, not from the Allen SDK cache used by the reference. It enumerates NWB filenames, filters the local `ophys_experiment_table.csv` to those experiment IDs, keeps `active_behavior` experiments, and further excludes `MESO.1` experiments before loading each experiment with `BehaviorOphysExperiment.from_nwb_path()`.

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

iii. In `CONVERSION_NOTES.md` Step 5 the AI explicitly decided that "Each experiment = one session," and in trajectory step 124 it justified excluding `MESO.1` because mixed 11 Hz and 31 Hz sessions would violate the format requirement that time bins be consistent across sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id`. During assembly, the AI reads `mouse_id` from each experiment's metadata, interns unique IDs into `subject_set`, and writes `subject_idx` per kept experiment/session.

ii.
```python
subject_set = {}  # mouse_id -> index
...
meta = session_data['metadata']
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. `CONVERSION_NOTES.md` Step 2 reports dataset size in terms of unique mice, and Step 9 compares converted mouse counts against the subset available after the AI's filtering.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one output session. It does not group experiments by shared `ophys_session_id`; instead each imaging plane is processed independently and appended as its own session entry.

ii.
```python
exp_ids = exp_table['ophys_experiment_id'].values
...
for i, exp_id in enumerate(exp_ids):
    ...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. `CONVERSION_NOTES.md` Step 5 states this directly: "Each experiment = one session: Each imaging plane treated as independent session." Trajectory step 56 repeats the same decision.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. The AI keeps rows marked `go` or `catch` and not `aborted` or `auto_rewarded`, then slices the full-session ophys timeline between `start_time` and `stop_time` for each remaining row. Trials are therefore variable-length and defined by the built-in trial table boundaries.

ii.
```python
trials = dataset.trials
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
...
for _, trial_row in valid_trials.iterrows():
    start_time = trial_row['start_time']
    stop_time = trial_row['stop_time']
    frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
    frame_indices = np.where(frame_mask)[0]
```

iii. In Step 3 and Step 5 notes, the AI documented the task rule "Include Go and Catch trials, exclude Aborted and Auto-rewarded trials." Trajectory steps 17, 22, and 57 show that it interpreted a trial as the full `start_time` to `stop_time` window on ophys timestamps.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes aborted and auto-rewarded trials, requires `go` or `catch`, skips trials with fewer than 5 ophys frames, skips experiments with fewer than 2 neurons or fewer than 2 valid trials, and globally removes `MESO.1` experiments. It does not require `change_time` to be present.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
if len(valid_trials) < 2:
    return None
...
if len(frame_indices) < 5:
    continue
...
if n_neurons < 2:
    return None
...
exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
```

iii. The notes justify the trial filter from the task instructions and justify excluding `MESO.1` in Step 10 / trajectory step 124 as a way to force a single frame rate. There is no separate documented justification for the `len(frame_indices) < 5` cutoff beyond it being an internal safeguard.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `dataset.events['events']`, not from `dff_traces`. It interprets this as detected calcium events produced by the Allen processing pipeline.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 1 says "Use `events` ... NOT `dff_traces`" and cites `methods.txt`: "For all analysis of neural data we used the detected calcium events." Trajectory steps 55-57 repeat that justification.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: the AI vertically stacks per-cell event traces from the single experiment into a `(neurons, time)` array and later slices that array into per-trial windows. Because each experiment is treated as its own session, there is no cross-plane merge step.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
...
neural_trial = events_array[:, frame_indices]
```

iii. The notes frame this as using already processed Allen outputs and avoiding further transforms. Step 6 also describes the script as a two-pass loader plus trial slicer rather than a neural pre-processing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no per-neuron quality filter beyond whatever Allen already stored in the NWB files. The only explicit AI-side filters are dropping whole experiments with fewer than 2 neurons and discarding experiments skipped earlier by the rig/session filters.

ii.
```python
n_neurons = events_array.shape[0]
if n_neurons < 2:
    return None
```

iii. `CONVERSION_NOTES.md` Steps 1 and 3 say ROI filtering was already applied by the Allen pipeline and that no additional neuron filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys timestamp clock and segmented by trial start/stop. The AI does not re-center trials on change time or another within-trial event; it simply takes the ophys frames whose timestamps fall in `[start_time, stop_time)`.

ii.
```python
start_time = trial_row['start_time']
stop_time = trial_row['stop_time']
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. The notes repeatedly describe "Temporal alignment: Based on ophys timestamps" and trajectory step 57 says: "For each trial, extract the ophys frames within `[start_time, stop_time]`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native Scientifica sampling without temporal rebinning and hardcodes metadata `time_bin_size` to `1000.0 / 31.0` ms after excluding `MESO.1` sessions. Running and pupil are resampled to that ophys grid, but neural activity itself is not temporally rebinned.

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
...
exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
```

iii. The AI first noticed mixed 11 Hz and 31 Hz sessions, then in trajectory steps 123-124 justified dropping the 11 Hz subset to satisfy the requirement that all trials share one bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, specifically `start_time`, `end_time`, and `image_name` within the `change_detection_behavior` stimulus block.

ii.
```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
...
starts = sp_active['start_time'].values
ends = sp_active['end_time'].values
names = sp_active['image_name'].values
```

iii. In Step 5 notes, the AI mapped `stimulus_presentations.image_name` to the output field and decided to use the change-detection block only.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI constructs a full-session per-frame image code array. For each stimulus presentation it writes the image code over the non-gray presentation interval, skips omitted stimuli, and then forward-fills the last image through gray gaps. Later it patches any remaining `-1` values at the start of a trial with the first valid image code in that trial, or 0 if none exists.

ii.
```python
image_idx = np.full(n_tp, -1, dtype=np.int32)
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
...
neg_mask = img_trial < 0
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. `CONVERSION_NOTES.md` Step 5 says "Image identity during gray screen: Forward-filled with last shown image." Trajectory steps 101-102 explain the later `-1` repair as a fix for frames before the first stimulus presentation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The AI first computes image identity on the full-session ophys timeline and then slices the same per-trial `frame_indices` used for neural data. Alignment is therefore by shared ophys frame index.

ii.
```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
...
frame_indices = np.where(frame_mask)[0]
...
img_trial = image_idx_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes describe this as "Resample/alignment to ophys timestamps" and treat all outputs as derived on the ophys grid before trial slicing.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `dataset.stimulus_presentations['is_change']` and the corresponding stimulus `start_time` / `end_time` values within the change-detection block.

ii.
```python
change_sp = sp_active[sp_active['is_change'] == True]
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

iii. `CONVERSION_NOTES.md` Step 5 maps `stimulus_presentations.is_change` directly to the image-change output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI makes a binary full-session vector on ophys timestamps and sets it to 1 only during stimulus presentations whose `is_change` flag is true. It does not extend the label through the following gray period, and it does not derive the change signal from trial-table `change_time`.

ii.
```python
def get_change_at_ophys(sp_active, ophys_ts):
    n_tp = len(ophys_ts)
    change = np.zeros(n_tp, dtype=np.int32)
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    return change
```

iii. The notes justify this at a high level as "1 during change presentation." Trajectory step 117 lists "Image change: correct (1 at change, 0 before)" as one of the AI's sanity checks.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding beyond binary labeling is applied. The output is directly coded as 0 for no-change frames and 1 for frames belonging to a change stimulus presentation.

ii.
```python
change = np.zeros(n_tp, dtype=np.int32)
...
change[mask] = 1
...
output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
```

iii. The notes describe image change as a binary variable rather than a discretized continuous one, so the AI treated exact binary assignment as sufficient.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is computed on the session-wide ophys timeline and then sliced with the same trial frame indices as the neural data.

ii.
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
...
change_trial = change_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The alignment rationale is the same one used throughout the script: compute on `ophys_ts`, then extract the shared per-trial window.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically its `timestamps` and `speed` columns.

ii.
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `dataset.running_speed` directly to the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto the ophys timestamps with `np.interp`, then computes five global percentile edges across the concatenated full-session running traces from all kept experiments, and finally bins each frame with `np.digitize` and clips the result to 0-4.

ii.
```python
def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)
...
all_running_concat = np.concatenate(all_running_values)
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
...
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 states "Global percentile binning" and Step 6 notes the two-pass design used to collect global statistics before final binning.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-percentile bins using percentile edges computed globally across all retained experiments.

ii.
```python
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The notes justify this as matching the decoder requirement for "five equal percentile bins" while keeping category definitions consistent across experiments.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The AI resamples running speed to the ophys timestamps before trial segmentation, then uses the same per-trial frame indices as the neural data.

ii.
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
...
running_trial = running_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes repeatedly say alignment is on the ophys clock for all output streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking['pupil_width']` and `dataset.eye_tracking['timestamps']`.

ii.
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. `CONVERSION_NOTES.md` Step 5 maps `dataset.eye_tracking.pupil_width` to the pupil-diameter output.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI drops only NaN pupil samples, not blink-flagged samples, interpolates the remaining samples onto the ophys timestamps with `np.interp`, computes five global percentile edges from all valid interpolated pupil values, and bins each ophys frame into 0-4. If there are too few valid samples or eye tracking fails, the session gets all-NaN pupil values which later map to bin 0.

ii.
```python
valid_mask = ~np.isnan(pupil_raw)
if valid_mask.sum() > 10:
    pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
...
all_pupil_concat = np.concatenate(all_pupil_values)
pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
...
pupil_binned[valid_pupil_mask] = np.clip(
    np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
).astype(np.int32)
```

iii. The notes justify interpolation to ophys timestamps and global percentile binning, but they do not document the divergence from blink filtering; that choice is only evident from the code.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five global percentile bins. Frames with missing pupil data remain at the default category 0 because `pupil_binned` is initialized to zeros and only valid frames are overwritten.

ii.
```python
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
if valid_pupil_mask.any() and pupil_edges is not None:
    pupil_binned[valid_pupil_mask] = np.clip(
        np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
    ).astype(np.int32)
```

iii. Step 5 notes say pupil diameter should use "5 percentile bins." The implicit missing-data-to-bin-0 behavior comes from the implementation rather than an explicit written justification.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The AI resamples pupil width onto the ophys timestamps before trial segmentation and then slices it with the same frame indices as the neural data.

ii.
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
...
pupil_trial = pupil_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The same ophys-timebase alignment rationale used for running speed is applied here.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. `CONVERSION_NOTES.md` Step 5 maps `trials.hit/miss/FA/CR` to a 4-category trial-outcome output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the four booleans to integer labels 0-3 with a fixed priority order, then repeats the resulting label across every time bin in the trial rather than storing a scalar once per trial.

ii.
```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
...
output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
```

iii. Step 5 notes explicitly say "Trial outcome as time-varying: Repeated constant value across trial timepoints."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several defensive fallbacks: failed experiment loads are skipped; experiments with too few neurons or trials are skipped; very short trials are dropped; missing or sparse pupil data become all-NaN and later category 0; pre-stimulus `-1` image labels are replaced with the first valid label in that trial or 0; and if no pupil data exist globally, `pupil_edges` is left as `None`.

ii.
```python
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    ...
    all_session_data.append(None)
    continue
...
if n_neurons < 2:
    return None
...
if len(valid_trials) < 2:
    return None
...
if len(frame_indices) < 5:
    continue
...
else:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
...
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The notes and trajectory justify the `-1` image fix and the rig exclusion explicitly. The rest of the fallback behavior is mostly implicit in the code rather than separately documented.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive step is loading every NWB experiment and retaining full-session arrays during Pass 1 while collecting global image names and percentile statistics. The notes report about 24 minutes for the full run.

ii.
```python
print('\n=== Pass 1: Loading data and collecting global statistics ===')
all_session_data = []
all_running_values = []
all_pupil_values = []
all_image_names = set()

for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    ...
    session_data = extract_session_data(dataset, temp_image_to_idx)
```

iii. `CONVERSION_NOTES.md` Step 6 says the full dataset took about 24 minutes and Step 9 reports the same order of runtime for the full conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python-level loops could have been vectorized further: the loop over stimulus presentations in `get_image_at_ophys`, the forward-fill loop over every ophys frame, the `iterrows()` loop in `get_change_at_ophys`, and the `iterrows()` trial loop in `segment_trials`.

ii.
```python
for j in range(len(starts)):
    ...
for i in range(n_tp):
    ...
for _, row in change_sp.iterrows():
    ...
for _, trial_row in valid_trials.iterrows():
    ...
```

iii. The AI did not explicitly document these inefficiencies, but they are visible in the final code and are the main loop-heavy parts beyond raw file I/O.

## 9-c. What processing does the code repeat multiple times?

i. The biggest repeated processing is image-identity construction: Pass 1 computes `image_idx_full` with a temporary image mapping, and Pass 2 recomputes it with the final global mapping. The code also repeats running/pupil binning work inside `plot_processing()` when visualization is enabled.

ii.
```python
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)
...
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
...
running_binned = np.clip(np.digitize(session_data['running_full'], running_edges[1:-1]), 0, 4)
...
pupil_binned[valid_p] = np.clip(np.digitize(session_data['pupil_full'][valid_p], pupil_edges[1:-1]), 0, 4)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly describes a two-pass approach. The repeated image-index computation is a direct consequence of that design.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The temporary Pass-1 image coding is discarded and replaced in Pass 2. The script also keeps `sp_active` only so it can rebuild image labels later, and the optional plotting pathway recomputes derived signals solely for inspection rather than for the saved dataset.

ii.
```python
# Build image_to_idx (temporary, will rebuild after collecting all names)
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
session_data = extract_session_data(dataset, temp_image_to_idx)
...
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
...
if args.show_processing and len(all_neural) <= 2:
    plot_processing(exp_id, session_data, running_edges, pupil_edges, neural_trials, output_trials)
```

iii. The notes justify the two-pass design for global statistics and the plotting mode for sanity checks, but those computations do not directly survive into the final saved arrays.
