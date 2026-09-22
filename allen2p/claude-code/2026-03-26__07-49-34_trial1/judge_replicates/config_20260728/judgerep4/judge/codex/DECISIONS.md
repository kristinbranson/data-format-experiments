# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a local metadata CSV (`ophys_experiment_table.csv`), enumerates downloaded NWB files from the local dataset directory, filters the table to experiments whose NWB files are present, excludes passive sessions, and then loads each experiment NWB file directly with `h5py`. It does not use the AllenSDK project cache.

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
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table

with h5py.File(nwb_path, 'r') as f:
    data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
```

iii. In `CONVERSION_NOTES.md` Step 6, the agent justified this as "Loads NWB files directly via h5py (fast, no AllenSDK overhead)" and limited itself to "active sessions" from the downloaded subset.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the experiment table. A `subject_map` assigns each unique mouse ID to an integer index, and the stored subject labels are strings.

ii.
```python
subject_map = {}  # mouse_id -> index
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
all_subject_idx.append(subject_map[mouse_id])
```

iii. This matches the agent’s mapping plan in `CONVERSION_NOTES.md` Step 5: "`mouse_id` -> `subjects`, `subject_idx`".

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` (one NWB file / one imaging plane) as one output session. It does not group multiple experiments sharing the same `ophys_session_id`. For multiscope data, the notes explicitly say each experiment/plane is a separate output "session".

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    ...
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
    session_metadata.append({
        'exp_id': result['exp_id'],
        'ophys_session_id': result['ophys_session_id'],
```

iii. `CONVERSION_NOTES.md` Step 5 states: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are defined from the NWB `intervals/trials` table. For each valid trial, the agent takes the full time window from `start_time` to `stop_time` and extracts all ophys frames satisfying `start_time <= t < stop_time`, producing variable-length trials.

ii.
```python
trial_data = nwb_data['trials']
valid_trial_idx = get_valid_trials(trial_data)
...
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent wrote: "Trial definition: Use `start_time` and `stop_time` from trials table for Go and Catch trials only."

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials where `(go | catch) & ~aborted & ~auto_rewarded`. It skips experiments with fewer than 2 such trials and skips individual trials with fewer than 2 ophys frames. It does not explicitly require non-null `change_time`.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]

if len(valid_trial_idx) < 2:
    return None
...
if n_trial_frames < 2:
    continue
```

iii. The justification in `CONVERSION_NOTES.md` Step 5 was: "Trial curation rules (for this decoder task): Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB dF/F fluorescence array at `processing/ophys/dff/traces/data`, plus the corresponding ophys timestamps from `processing/ophys/dff/traces/timestamps`.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
```

iii. The notes say dF/F is already precomputed and is the chosen neural signal: "Neural data: Use dF/F traces (not events/deconvolved)."

## 2-b. How is the `neural` data processed?

i. The agent transposes the NWB dF/F array from `(n_frames, n_cells)` to `(n_cells, n_frames)`, then slices per trial with the trial frame mask. It does not apply additional normalization, denoising, or plane-merging; each experiment is processed independently.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 3 says dF/F is already precomputed; Step 5 says "dF/F is the standard calcium imaging signal." Step 5 also states the decision to keep each multiscope plane as a separate output session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filtering in the script. The agent includes all cells present in each NWB dF/F table and only skips experiments with zero cells.

ii.
```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```

iii. In `CONVERSION_NOTES.md` Step 10, the agent justified this by claiming: "No explicit valid_roi filter ... OK - all ROIs in downloaded NWB files are valid."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial windows on the ophys timebase. For each trial, the agent selects the ophys frames between trial `start_time` and `stop_time`; metadata later records the alignment event as `trial_start`.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
...
'temporal_alignment_event': 'trial_start',
```

iii. `CONVERSION_NOTES.md` Step 5 says: "Temporal alignment: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment keeps its native ophys frame rate, with `dt` computed as the median difference of `ophys_ts`. Because the agent includes both Scientifica and Multiscope experiments, different sessions can have different native frame rates.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': float(median_dt),
```

iii. The notes explicitly justify this choice: "Time bin: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope)."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation interval table, specifically `image_name`, `start_time`, and `stop_time`, not from the trial table’s `initial_image_name` / `change_image_name`.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
...
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. In `CONVERSION_NOTES.md` Step 5, the mapping plan says: "`image_name` from stimulus presentations -> `output[0]`: image_identity".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent first scans all experiments to collect the union of image names, prepends a synthetic `"gray"` category, and then, per trial, initializes every ophys frame to `"gray"` and overwrites frames that fall inside a stimulus presentation with that presentation’s image code. `'omitted'` presentations are skipped and therefore remain gray.

ii.
```python
GRAY_LABEL = 'gray'
...
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
if name == 'omitted':
    continue
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. `CONVERSION_NOTES.md` Step 5 says: "Image identity: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned on the same per-trial ophys frames as the neural data. The agent computes the image trace over `trial_ts = ophys_ts[trial_mask]`, and the same `trial_mask` is used to slice neural data.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
img_trace, _ = build_image_identity_trace(
    ophys_ts, stim_data, t_start, t_stop, image_names_list
)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes describe this as "time-varying outputs aligned to ophys timestamps."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` flag and each presentation’s `start_time`, rather than from trial-level `change_time` and `go`.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. `CONVERSION_NOTES.md` Step 5 maps "`is_change` from stimulus presentations" to the `image_change` output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every stimulus presentation within the trial whose `is_change` flag is true, the agent marks only the first ophys frame at or after that presentation onset as `1`. All other frames remain `0`.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    s_start = stim_starts[si]
    if s_start < trial_start or s_start >= trial_stop:
        continue
    frame_idx = np.searchsorted(trial_ts, s_start)
    if frame_idx < n_frames:
        trace[frame_idx] = 1
```

iii. The mapping plan in `CONVERSION_NOTES.md` Step 5 says: "`is_change` from stimulus presentations -> binary, 1 at change onset frame, 0 otherwise."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is applied beyond binary coding. The output categories are fixed as `0 = no_change` and `1 = change`.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
```

iii. The agent’s notes consistently describe image change as a binary event trace.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned to the same per-trial ophys frames as the neural data; change onsets are snapped to the first ophys frame at or after the presentation onset.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The justification is the same as for image identity: all outputs are first put on the ophys timebase, then paired with the neural trial slices.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running-speed stream: `processing/running/speed/data` and its `timestamps`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `running_speed` directly to the output after interpolation and binning.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent linearly interpolates running speed onto the full-session ophys timestamps, computes percentile bin edges from the entire interpolated session, and then bins each trial’s running trace with those session-specific edges. NaNs are assigned to bin 0.

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

iii. `CONVERSION_NOTES.md` Step 5 states: "Interpolate from 60 Hz to ophys timestamps using linear interpolation" and "Compute percentiles across the entire session ... then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into 5 equal-percentile bins, but the percentile edges are computed separately for each experiment/session rather than globally across the whole dataset.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
...
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
```

iii. The notes explicitly justify session-wise percentile bins in Step 5.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The full-session running trace is first interpolated onto `ophys_ts`, then the same per-trial `frame_mask` used for the neural slice is applied to obtain the running slice.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
running_trial = running_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The justification in the notes is that all streams should be aligned on the ophys timestamps before trial extraction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking stream’s pupil `area`, `timestamps`, and `likely_blink` values in the NWB file.

ii.
```python
pt = et['pupil_tracking']
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. `CONVERSION_NOTES.md` Step 5 says: "Pupil diameter: Compute from pupil area as `2*sqrt(area/pi)`."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent sets blink frames to NaN, converts pupil area to a diameter estimate using `2 * sqrt(area / pi)` for positive areas, interpolates that diameter trace to `ophys_ts`, then bins it into 5 session-specific percentile bins. NaNs map to bin 0.

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

iii. The justification is recorded in `CONVERSION_NOTES.md` Step 5 and Step 10: blinks become NaN, diameter is derived from area, and NaN-to-bin-0 is treated as an acceptable design choice.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into 5 equal-percentile bins, with percentile cutoffs computed separately per experiment/session.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. The notes state the same session-wide percentile-binning rule used for running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the experiment’s ophys timestamps first, and each trial then uses the same `frame_mask` as the neural trial slice.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The agent’s alignment plan is consistent across all behavioral outputs: interpolate to `ophys_ts`, then slice with the trial mask.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
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
```

iii. `CONVERSION_NOTES.md` Step 5 maps "Trial outcome (hit/miss/FA/CR)" directly to `output[4]`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent converts the outcome string to an integer index using `outcome_names`, then broadcasts that single class label across every time bin in the trial output matrix. Unknown outcomes fall back to index 0.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
...
output_full[4] = outcome_idx
```

iii. The script comments show the agent considered making trial outcome static, then chose to repeat it across time so all outputs fit one `(5, T)` array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases by skipping or defaulting: missing NWB files, missing stimulus tables, zero-cell experiments, and experiments with fewer than 2 valid trials are dropped; individual trials with fewer than 2 frames are skipped; missing pupil data yields all-NaN pupil traces; blink samples are set to NaN; interpolation-produced NaNs are converted to bin 0.

ii.
```python
if not os.path.exists(nwb_path):
    return None
...
if stim_data is None:
    return None
...
if n_cells == 0:
    return None
...
if len(valid_trial_idx) < 2:
    return None
...
if n_trial_frames < 2:
    continue
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0
```

iii. `CONVERSION_NOTES.md` Step 10 explicitly defends NaN-to-bin-0 and documents these cases as benign edge handling rather than conversion bugs.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading each NWB file from disk and scanning all experiments once to collect image names. The notes estimate NWB loading at about 1.7 seconds per session and mention image-name collection as extra overhead.

ii.
```python
nwb_data = load_nwb_data(nwb_path)
...
all_image_names = get_all_image_names(exp_table)
```

iii. In `CONVERSION_NOTES.md` Step 7, the runtime estimate says: "Load NWB ~1.7s/session" and "Image name collection adds ~14s overhead."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The expensive Python loops are the repeated per-stimulus loops inside `build_image_identity_trace()` and `build_image_change_trace()` for every trial, plus the per-file and per-trial loops in `get_all_image_names()`, `load_experiment_table()`, and `process_experiment()`. The code builds boolean masks repeatedly instead of precomputing intervals once.

ii.
```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    ...
    frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. The agent did not explicitly justify these loops beyond favoring straightforward direct NWB processing; the trajectory and notes emphasize speed of implementation and raw I/O rather than vectorization.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several passes: it scans all experiments once to enumerate downloaded NWBs, scans all experiments again to collect image names, and then opens every experiment again for full processing. Within each trial it separately traverses stimulus presentations to build image identity and image change, and recomputes trial frame masks in multiple helper functions.

ii.
```python
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
...
all_image_names = get_all_image_names(exp_table)
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)
```

iii. The trajectory shows the agent accepted this redundancy because direct NWB access was "fast" enough and the final verification/training succeeded.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `cell_roi_ids` but never uses them; it constructs `output_tv` and `output_static` and then discards both in favor of `output_full`; it also records load/process timing fields that are only used for logs, not the final dataset. Optional plotting machinery (`--show-processing`) is also outside the downstream dataset.

ii.
```python
if 'image_segmentation' in f['processing']['ophys']:
    ...
    data['cell_roi_ids'] = seg[key]['id'][:]

output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
output_static = np.array([outcome_idx], dtype=np.int64)
...
result = {
    ...
    't_load': t_load,
    't_process': t_process,
}
```

iii. There is no explicit justification for these discarded computations in the notes; they appear to be implementation leftovers from earlier design choices.
