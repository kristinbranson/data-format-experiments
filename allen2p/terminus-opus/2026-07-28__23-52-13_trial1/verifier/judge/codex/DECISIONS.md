# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates local NWB files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, filters `ophys_experiment_table.csv` down to those experiment IDs, keeps only `active_behavior`, excludes `MESO.1` equipment, and then loads each remaining experiment directly from disk with `BehaviorOphysExperiment.from_nwb_path()`. It does not use the Allen SDK project cache or reconstruct sessions from multiple experiments.

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

iii. In `CONVERSION_NOTES.md`, the agent justified this as working from the local subset of available NWB files, focusing on active behavior only, and excluding Multiscope because it believed the target format required a single shared time bin size.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by unique `mouse_id` values taken from each loaded experiment's metadata. A running dictionary maps each new mouse ID to a subject index.

ii.
```python
subject_set = {}
...
meta = session_data['metadata']
mouse_id = str(meta['mouse_id'])
if mouse_id not in subject_set:
    subject_set[mouse_id] = len(subject_set)
all_subject_idx.append(subject_set[mouse_id])
```

iii. The notes say subjects correspond to mice in the experiment metadata, and the trajectory repeatedly summarizes the dataset in terms of unique mice discovered from the metadata tables.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as an independent session. It does not group multiple experiments from the same `ophys_session_id` together.

ii.
```python
exp_table = get_experiment_table()
exp_ids = exp_table['ophys_experiment_id'].values
...
for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    session_data = extract_session_data(dataset, temp_image_to_idx)
```

iii. `CONVERSION_NOTES.md` explicitly records the decision: "Each experiment = one session: Each imaging plane treated as independent session." The stated reason was to keep a uniform per-session frame rate and avoid multi-plane grouping complexity.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. For each valid trial row, the agent takes all ophys frames with timestamps in `[start_time, stop_time)`, producing variable-length trials.

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

iii. In the trajectory, the agent reasoned that trial windows are variable length and should be aligned to ophys timestamps, using the full start-to-stop interval rather than a fixed window.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only `go` or `catch` trials, excludes `aborted` and `auto_rewarded` trials, skips sessions with fewer than 2 valid trials, and drops segmented trials with fewer than 5 ophys frames. It does not explicitly require non-null `change_time`.

ii.
```python
valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
if len(valid_trials) < 2:
    return None
...
if len(frame_indices) < 5:
    continue
```

iii. The notes justify the first part directly from the task: include Go and Catch, exclude Aborted and Auto-rewarded. The extra trial-length and session-size thresholds appear to be pragmatic guardrails rather than reference-derived curation rules.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives neural data from `dataset.events`, specifically the per-neuron `events` arrays.

ii.
```python
events = dataset.events
events_array = np.vstack(events['events'].values).astype(np.float32)
```

iii. The agent's notes repeatedly justify this by quoting the methods text: "For all analysis of neural data we used the detected calcium events," and by stating that events were preferred over `dff_traces`.

## 2-b. How is the `neural` data processed?

i. The agent vertically stacks the event vectors into a neuron-by-time matrix for each experiment and then slices that matrix into trials. It performs no additional normalization, denoising, or merging across experiments in the same ophys session.

ii.
```python
events_array = np.vstack(events['events'].values).astype(np.float32)
...
neural_trial = events_array[:, frame_indices]
```

iii. The notes say the events already reflect Allen pipeline processing and that no further neuron filtering or transformation was necessary beyond extracting per-trial windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter. The only neural-related filter is dropping whole experiments with fewer than 2 neurons.

ii.
```python
n_neurons = events_array.shape[0]
if n_neurons < 2:
    return None
```

iii. `CONVERSION_NOTES.md` states that ROI filtering was already applied by the Allen pipeline and therefore "No neuron filtering" was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys time base and then segmented by trial start and stop times. Per trial, the extracted window begins at `start_time`, not at `change_time`.

ii.
```python
frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
frame_indices = np.where(frame_mask)[0]
neural_trial = events_array[:, frame_indices]
```

iii. The notes and trajectory describe the alignment as "based on ophys timestamps" with trial windows taken directly from the SDK trial definitions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native ophys sampling for the retained Scientifica experiments and does not rebin in time. It hard-codes metadata `time_bin_size` to `1000/31` ms and excludes Multiscope sessions to enforce that rate.

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
```

iii. The notes explicitly justify excluding Multiscope sessions because the agent believed "The format requires consistent time bins across all sessions," so it retained only 31 Hz experiments.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, restricted to `stimulus_block_name == 'change_detection_behavior'`, using the `image_name`, `start_time`, and `end_time` columns.

ii.
```python
sp = dataset.stimulus_presentations
sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
...
starts = sp_active['start_time'].values
ends = sp_active['end_time'].values
names = sp_active['image_name'].values
```

iii. The notes say image identity should come from actual stimulus presentations in the active change-detection block, with omissions skipped and gray periods filled using the last image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent builds a global mapping from image names to integer IDs, marks frames during each non-omitted stimulus presentation with that ID, forward-fills through gray periods, and replaces any leading `-1` values inside a trial with the first valid image ID.

ii.
```python
image_idx = np.full(n_tp, -1, dtype=np.int32)
for j in range(len(starts)):
    img_name = names[j]
    if not isinstance(img_name, str) or img_name == 'omitted':
        continue
    mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
    image_idx[mask] = image_to_idx[img_name]

last_img = -1
for i in range(n_tp):
    if image_idx[i] >= 0:
        last_img = image_idx[i]
    elif last_img >= 0:
        image_idx[i] = last_img
...
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The trajectory documents the rationale twice: first, to "forward-fill" image identity during gray periods, and later to patch a small number of `-1` values at the start of trials by using the next valid image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The agent computes a full-session image ID time series on the ophys timestamps and then indexes that series with the same frame indices used for neural trial extraction.

ii.
```python
image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
...
frame_indices = np.where(frame_mask)[0]
img_trial = image_idx_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes call out temporal alignment to the ophys timestamps as the main synchronization strategy for neural and behavioral outputs.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `dataset.stimulus_presentations` in the active block, specifically the `is_change`, `start_time`, and `end_time` columns.

ii.
```python
change_sp = sp_active[sp_active['is_change'] == True]
for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1
```

iii. The notes describe this as using the actual change presentations from stimulus timing rather than trial metadata.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent constructs a full-session binary time series that is `1` only during frames whose ophys timestamps fall inside a stimulus presentation row marked `is_change == True`; it is `0` elsewhere.

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

iii. The agent’s notes and sanity checks focus on "1 at change, 0 before" and do not mention extending the label through the post-flash gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No extra thresholding is applied beyond binary categorization: `0` for no change and `1` for change.

ii.
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    TRIAL_OUTCOME_NAMES,
]
```

iii. The choice is implicit in the code and the notes, which consistently describe image change as a binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The binary change series is first created on the full ophys time base and then sliced with the same trial frame indices used for neural data.

ii.
```python
change_full = get_change_at_ophys(sp_active, ophys_ts)
...
frame_indices = np.where(frame_mask)[0]
change_trial = change_full[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes state that all outputs were resampled or defined on ophys timestamps so they could share the neural indexing directly.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
running = dataset.running_speed
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
```

iii. The notes identify running speed as a behavioral stream sampled separately from ophys and therefore needing resampling to the ophys clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent linearly interpolates running speed onto ophys timestamps with `np.interp`, pools all full-session values across experiments to compute five percentile edges, and digitizes each frame into bins 0-4.

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

iii. `CONVERSION_NOTES.md` explicitly records the decision "Global percentile binning" for running speed, and the trajectory describes resampling running from its native timestamps to the ophys time base.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five percentile bins using globally computed edges and converted to integer labels 0 through 4.

ii.
```python
running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
```

iii. The notes justify global binning so the same category semantics apply across all experiments.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto the full-session ophys timestamps and then indexed by the same per-trial frame indices as the neural data.

ii.
```python
running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
...
running_trial = running_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes repeatedly say that all behavioral outputs were aligned to the ophys timestamp stream before trial segmentation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically `pupil_width` and `timestamps`.

ii.
```python
eye = dataset.eye_tracking
pupil_raw = eye['pupil_width'].values
pupil_ts = eye['timestamps'].values
```

iii. The notes identify eye tracking as the pupil source but do not mention using `likely_blink`; the implementation instead relies on NaN filtering and exception handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent removes NaNs from `pupil_width`, linearly interpolates the remaining samples to the ophys timestamps with `np.interp`, falls back to an all-NaN vector if eye tracking is unavailable or too sparse, pools valid pupil values globally to compute percentile edges, and digitizes valid frames into bins 0-4.

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

iii. The agent’s justification in the notes is mainly that pupil needed the same resampling and percentile discretization as running speed; handling of missing eye data was treated as an edge case.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five percentile bins computed from the pooled valid pupil samples, with missing samples left at the default category `0`.

ii.
```python
pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
if valid_pupil_mask.any() and pupil_edges is not None:
    pupil_binned[valid_pupil_mask] = np.clip(
        np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
    ).astype(np.int32)
```

iii. The notes state that pupil diameter should use the same global percentile scheme as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is interpolated to the full-session ophys timestamps and then sliced with the same frame indices used for the neural trial matrix.

ii.
```python
pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
...
pupil_trial = pupil_binned[frame_indices]
neural_trial = events_array[:, frame_indices]
```

iii. The notes frame this as standard synchronization to the ophys clock before trial extraction.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def get_trial_outcome(trial_row):
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1
```

iii. The notes and tutorials both identify these four booleans as the canonical non-aborted trial outcomes for this task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent maps the four outcome booleans to integer classes 0-3 and repeats the chosen class across every timepoint in that trial.

ii.
```python
outcome = get_trial_outcome(trial_row)
outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
...
output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
```

iii. `CONVERSION_NOTES.md` explicitly lists "Trial outcome as time-varying: Repeated constant value across trial timepoints" as a key decision.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases pragmatically: failed experiment loads are skipped; experiments with too few neurons or trials are dropped; very short segmented trials are skipped; missing or sparse eye tracking becomes all-NaN pupil; missing pupil bins default to 0; and image IDs of `-1` at the start of a trial are replaced with the first valid image in that trial.

ii.
```python
try:
    dataset = load_experiment(exp_id)
except Exception as e:
    ...
    continue
...
if n_neurons < 2:
    return None
...
if len(frame_indices) < 5:
    continue
...
except:
    pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
...
if neg_mask.any():
    first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
    img_trial[neg_mask] = first_valid
```

iii. The trajectory explicitly mentions fixing four trials with `-1` image identity values by backfilling from the next valid image, and the notes describe missing eye data and zero-event trials as expected edge cases rather than reasons to halt processing.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each NWB experiment from disk in Pass 1 and holding full-session arrays long enough to compute global percentile edges and image mappings.

ii.
```python
for i, exp_id in enumerate(exp_ids):
    dataset = load_experiment(exp_id)
    ...
    session_data = extract_session_data(dataset, temp_image_to_idx)
...
all_running_concat = np.concatenate(all_running_values)
all_pupil_concat = np.concatenate(all_pupil_values)
```

iii. The notes report about 24 minutes for the full run and describe the script as a two-pass pipeline, which implies experiment loading and full-session extraction dominate the runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the loops over stimulus presentations in `get_image_at_ophys()` and `get_change_at_ophys()`, the manual forward-fill loop over all ophys frames, and the per-trial `iterrows()` loop in `segment_trials()`.

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

for _, row in change_sp.iterrows():
    mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
    change[mask] = 1

for _, trial_row in valid_trials.iterrows():
    ...
```

iii. The agent does not explicitly justify leaving these loops scalar in the notes; this is a property of the implementation rather than a documented design choice.

## 9-c. What processing does the code repeat multiple times?

i. The code uses two passes over the experiments. In Pass 1 it loads every experiment and computes preliminary image indices with a temporary mapping; in Pass 2 it recomputes `image_idx_full` with the final global mapping and then segments trials. The plotting helper also recomputes running and pupil bins for visualization.

ii.
```python
# Pass 1
session_data = extract_session_data(dataset, temp_image_to_idx)
...
# Pass 2
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
neural_trials, output_trials = segment_trials(session_data, running_edges, pupil_edges)
...
running_binned = np.clip(np.digitize(session_data['running_full'], running_edges[1:-1]), 0, 4)
```

iii. `CONVERSION_NOTES.md` explicitly describes the script as a "two-pass approach," with the first pass collecting global statistics and the second pass re-extracting image indices and segmenting trials.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several full-session intermediates that are only needed transiently: full-session `image_idx_full`, `change_full`, `running_full`, and `pupil_full` arrays; `sp_active`; `dt`; and, when enabled, visualization-specific recomputations. The temporary image mapping from Pass 1 is also discarded once the final global mapping is built.

ii.
```python
session_data = {
    'image_idx_full': image_idx_full,
    'change_full': change_full,
    'running_full': running_full,
    'pupil_full': pupil_full,
    'dt': dt,
    'sp_active': sp_active,
}
...
temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
...
session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
```

iii. The notes frame these choices as implementation conveniences for global binning, later verification, and optional plotting rather than downstream-required outputs.
