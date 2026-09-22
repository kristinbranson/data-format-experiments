# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads the local metadata CSV `ophys_experiment_table.csv`, finds downloaded NWB files by globbing `/app/data/.../behavior_ophys_experiments/*.nwb`, filters the experiment table to those experiment IDs, removes passive sessions, and then opens each NWB file directly with `h5py`.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
    return exp_table
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
```

iii. The justification in `CONVERSION_NOTES.md` is that direct NWB loading via `h5py` is "fast, no AllenSDK overhead," and that only the downloaded subset should be used. The notes also state that passive sessions should be excluded.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the metadata table. The script creates subject indices lazily as experiments are processed.

ii.
```python
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)

all_subject_idx.append(subject_map[mouse_id])
```

iii. The notes explicitly map `mouse_id` to `subjects` and `subject_idx`, and the trajectory summary reports the final dataset in terms of unique mice.

## 1-c. How are the data split into sessions?

i. Each **experiment** is treated as one output session. The code iterates over rows of the experiment table, processes one NWB per row, and appends that result as one session. `ophys_session_id` is stored only as metadata; experiments sharing the same `ophys_session_id` are not grouped together.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

```python
session_metadata.append({
    'exp_id': result['exp_id'],
    'ophys_session_id': result['ophys_session_id'],
    ...
})
```

iii. The planning notes say: "Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." The consistency notes also discuss handling Multiscope experiments separately.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each valid trial, the code uses `start_time` and `stop_time` and includes all ophys frames satisfying `start_time <= t < stop_time`.

ii.
```python
trial_data = nwb_data['trials']
valid_trial_idx = get_valid_trials(trial_data)

for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
```

iii. The notes say: "Trial definition: Use `start_time` and `stop_time` from trials table for Go and Catch trials only." The trajectory also emphasizes that short sessions with few valid trials were considered legitimate rather than a bug.

## 1-e. How are trials filtered based on quality controls?

i. The script keeps only Go or Catch trials, excludes `aborted` and `auto_rewarded`, skips trials with fewer than 2 ophys frames, and skips experiments with fewer than 2 surviving trials. It does **not** explicitly require non-null `change_time`.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]
```

```python
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
    return None
```

iii. The notes state the trial curation rule as "Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials." The trajectory later defends sessions with very few valid trials as acceptable if they still pass the `>=2` threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB dataset `processing/ophys/dff/traces/data` plus its timestamps `processing/ophys/dff/traces/timestamps`.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. The notes say the AI chose dF/F traces rather than events because dF/F is the standard calcium-imaging signal and is already pre-computed in the NWB files.

## 2-b. How is the `neural` data processed?

i. The AI applies minimal processing: transpose the NWB dF/F matrix from `(n_frames, n_cells)` to `(n_cells, n_frames)`, slice it by each trial’s frame mask, and cast each trial matrix to `float32`. It does not merge experiments that share a session.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
neural_trials.append(neural)
```

iii. The notes justify this by saying dF/F is already pre-computed and that each experiment should remain separate in the output rather than merged across planes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter in the script. The only neural QC is to skip experiments with zero cells. The script reads `cell_roi_ids` from image segmentation metadata but never uses them to filter cells.

ii.
```python
if 'image_segmentation' in f['processing']['ophys']:
    seg = f['processing']['ophys']['image_segmentation']
    for key in seg.keys():
        if 'id' in seg[key]:
            data['cell_roi_ids'] = seg[key]['id'][:]
            break
```

```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. In `CONVERSION_NOTES.md`, the AI acknowledges Allen SDK ROI filtering but concludes that no extra explicit filter is needed, writing that the downloaded NWB files already appear usable. The trajectory also mentions considering a `valid_roi` filter "for safety" but leaving the code unchanged.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys timestamps and segmented by trial `start_time`/`stop_time`. Within a trial, all arrays use the same boolean frame mask on the ophys clock.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes explicitly say "Temporal alignment: Align to ophys timestamps" and "For each trial, extract the ophys frames between trial start_time and stop_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment stays on its native ophys sampling grid. The per-experiment `dt` is the median inter-frame interval, and the saved metadata reports the median `dt` across experiments.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
```

iii. The notes say to use native ophys timestamps and explicitly mention both ~31 Hz and ~11 Hz data. The trajectory later comments that Multiscope experiments are shorter in frames because of the lower sampling rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table, specifically `start_time`, `stop_time`, and `image_name`. It is not derived from the trials table’s `initial_image_name` and `change_image_name`.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
```

```python
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The notes say image identity should come "from stimulus presentations" and should include a gray-screen category during the inter-stimulus interval.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code first scans all experiments to collect all unique stimulus `image_name` values, prepends a synthetic `"gray"` label, initializes every trial frame to gray, then overwrites frames covered by each non-omitted stimulus presentation with the corresponding image index.

ii.
```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
```

```python
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
if name == 'omitted':
    continue
...
img_idx = image_names_list.index(name)
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The notes justify this as matching the actual flashed stimulus sequence, including gray inter-stimulus intervals. The full-conversion notes later use the gray fraction as a sanity check against the expected 500/750 ms duty cycle.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-identity trace is built directly on the trial’s ophys timestamps. A trial-specific mask selects the same frames used for neural data, and each stimulus presentation is mapped onto those trial timestamps.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The AI’s stated rationale is that all outputs should be aligned to the ophys clock; the notes call this out as a key mapping decision.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from the stimulus-presentation table’s `is_change` flag and `start_time`, not from the trials table’s `go` and `change_time`.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The notes map `is_change` from stimulus presentations directly to `output[1]`, and the later sanity checks compare the resulting trace against raw NWB stimulus timing.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every stimulus presentation flagged as a change and lying inside the trial window, the code sets a single 1 at the first ophys frame at or after the stimulus onset. All other frames are 0.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
if not is_change[si]:
    continue
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The notes describe `image_change` as "1 at change onset frame, 0 otherwise." The later trajectory also treats the extreme class imbalance as expected for such a sparse event code.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary with categories `0 = no_change` and `1 = change`. There is no additional thresholding beyond the boolean `is_change` flag and onset mapping.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    outcome_names,
]
```

iii. This follows the task requirement that image change be binary and time-varying.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is built on the same per-trial ophys timestamps used to slice neural activity. The onset is snapped to the first trial frame at or after the stimulus change onset.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. The notes emphasize alignment to ophys timestamps for all time-varying outputs, and the sanity-check section claims this trace was spot-checked against the raw NWB timing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from NWB `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes map `running_speed` directly from the NWB running data and describe interpolation onto the ophys clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its own timestamps to the full-session ophys timestamps. Percentile bin edges are then computed **within that experiment/session**, and trial-level running traces are discretized with those session-specific edges. NaNs are later mapped to bin 0.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The notes explicitly say: "Running speed: Interpolate from 60 Hz to ophys timestamps using linear interpolation" and "Compute percentiles across the entire session ... then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into 5 percentile bins per session using the empirical percentiles of the non-NaN `running_at_ophys` values. The first and last edges are widened to `-inf` and `inf`, and NaNs are assigned to category 0.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges
```

```python
def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. The notes justify 5 equal percentile bins because the task asks for that representation, and the trajectory later points to the roughly balanced bin counts as evidence that this choice worked.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running-speed signal is interpolated to the experiment’s ophys timestamps before trial segmentation. Trial extraction then uses the same frame mask as the neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes repeatedly state that behavioral outputs should be aligned to the ophys timestamps and then sliced trial-by-trial with the same indices as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_area`, eye-tracking timestamps, and `likely_blink`.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The notes say the AI chose to compute diameter from pupil area using the whitepaper formula and to use the blink mask provided by the dataset.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked samples are set to `NaN`; pupil diameter is computed as `2 * sqrt(area / pi)` for positive areas; the result is linearly interpolated to the ophys timestamps; then session-specific percentile edges are computed and applied to each trial.

ii.
```python
pupil_area = nwb_data['pupil_area'].copy()
likely_blink = nwb_data['likely_blink']
pupil_area[likely_blink] = np.nan

pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)

pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The notes justify this as matching the whitepaper’s area-to-diameter conversion and blink handling. The trajectory explicitly says NaN blink frames were kept and later mapped into bin 0 rather than creating a sixth category.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into 5 percentile bins within each experiment/session, using only non-NaN values to compute edges. NaNs are assigned to category 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

```python
result[~valid] = 0
```

iii. The notes and trajectory both defend NaN-to-0 as a deliberate design choice because the task asked for 5 bins, not 5 bins plus a blink class.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil trace is converted to the ophys timebase before trial segmentation, and then trial samples are selected with the same frame mask as neural activity.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes treat this the same way as running speed: first synchronize to ophys timestamps, then use common trial masks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
             'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
             'initial_image_name', 'change_image_name', 'is_change']:
    if key in trials:
        trial_data[key] = trials[key][:]
```

```python
def get_trial_outcome(trial_data, idx):
    if trial_data['hit'][idx]:
        return 'hit'
    elif trial_data['miss'][idx]:
        return 'miss'
    elif trial_data['false_alarm'][idx]:
        return 'false_alarm'
    elif trial_data['correct_reject'][idx]:
        return 'correct_reject'
    else:
        return 'unknown'
```

iii. The notes map trial outcome directly from these four trial labels and list the corresponding output categories.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code turns the outcome booleans into a string label with ordered precedence `hit -> miss -> false_alarm -> correct_reject`, converts that label to an integer by position in `outcome_names`, and then broadcasts the resulting class index across every time bin in the trial output array.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The notes describe trial outcome as a static per-trial target, but the code comments show the AI consciously chose to repeat the static value across time so all outputs fit in a single `(5, T)` array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable experiments are skipped if the NWB file is absent, the experiment has zero cells, there is no stimulus table, or too few valid trials survive. Missing pupil data produces all-NaN pupil traces, which are later mapped to bin 0. Blink-marked pupil samples are set to NaN. Omitted stimuli are left as gray. Unknown trial outcomes fall back to class 0.

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}")
    return None
...
if n_cells == 0:
    return None
...
if stim_data is None:
    return None
...
if len(valid_trial_idx) < 2:
    return None
```

```python
if nwb_data['pupil_area'] is not None:
    pupil_area[likely_blink] = np.nan
    ...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

```python
if name == 'omitted':
    continue
...
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```

iii. The notes and trajectory frame these as pragmatic robustness measures. In particular, the trajectory explicitly discusses the NaN-to-bin-0 choice for pupil data as intentional.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are per-experiment NWB loading and the extra full-dataset scan used to collect image names before the real conversion pass. The notes estimate roughly 1.7 s/session for NWB loading and mention that image-name collection adds about 14 s total.

ii.
```python
all_image_names = get_all_image_names(exp_table)
```

```python
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
```

iii. `CONVERSION_NOTES.md` says "Load NWB ~1.7s" and explicitly notes "Image name collection adds ~14s overhead."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several serial loops could be vectorized or reduced: the loop over NWB files to collect downloaded IDs, the scan over every NWB in `get_all_image_names`, the stimulus-presentation loops in `build_image_identity_trace` and `build_image_change_trace`, and the per-trial loop in `process_experiment`.

ii.
```python
for f in nwb_files:
    eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
    downloaded_ids.add(eid)
```

```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

```python
for trial_idx in valid_trial_idx:
    ...
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI did not explicitly justify these loops, but the notes emphasize simplicity and direct NWB access rather than aggressive vectorization or caching.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats at least three kinds of work: it scans every NWB once in `get_all_image_names` and then reopens the same NWBs again for conversion; it recomputes trial masks/timestamps separately in the main trial loop and again inside the image-identity and image-change helper functions; and it builds temporary `output_tv` and `output_static` arrays that are immediately superseded by `output_full`.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
result = process_experiment(exp_id, row, image_names_list, outcome_names,
                            show_processing=args.show_processing)
```

```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
...
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
```

```python
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

iii. The notes acknowledge the image-name pre-scan overhead, but the rest of the repeated work is implicit in the implementation rather than documented as a deliberate tradeoff.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is loading `cell_roi_ids` and never using them, constructing `output_tv` and `output_static` and then discarding them, and computing some trial-local variables only to replace them with other derived arrays.

ii.
```python
if 'image_segmentation' in f['processing']['ophys']:
    ...
    data['cell_roi_ids'] = seg[key]['id'][:]
```

```python
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
output_trials.append(output_full)
```

iii. There is no explicit justification for this discarded work in the notes. It appears to be leftover scaffolding from development and format decisions rather than intentional downstream functionality.
