# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK loader. It reads the local NWB (HDF5) files directly with `h5py`, and uses the project metadata CSV (`project_metadata/ophys_experiment_table.csv`) as the index of what exists. The experiment table is filtered to (a) experiments whose NWB file is actually present on disk (284 files), and (b) experiments whose `session_type` does not contain "passive" (→ 202 experiments). No filter on `project_code` is applied, so the converted set contains both `VisualBehavior` (168 single-plane experiments) and `VisualBehaviorMultiscope` (34 plane-experiments from 6 real sessions of 1 mouse). Every remaining experiment is then opened once and all streams (dF/F + ophys timestamps, running speed, eye tracking, trials table, stimulus-presentations table) are pulled out of the HDF5 groups in one pass.

ii.
```python
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NWB_DIR = os.path.join(DATA_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')
METADATA_DIR = os.path.join(DATA_DIR, 'visual-behavior-ophys-1.1.0', 'project_metadata')

def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
    # Filter to active sessions only (exclude passive)
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table
```

```python
def load_nwb_data(nwb_path):
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed'] = f['processing']['running']['speed']['data'][:]
        ...
        trials = f['intervals']['trials']
        for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
                    'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
                    'initial_image_name', 'change_image_name', 'is_change']:
            if key in trials:
                trial_data[key] = trials[key][:]
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. From CONVERSION_NOTES.md Step 6: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)". Step 10 Check 3 argues the two loaders are equivalent ("h5py direct NWB read" vs `BehaviorOphysExperiment.from_nwb()` → "Equivalent") and that ROI curation is unnecessary because "all ROIs in downloaded NWB files are valid". Passive exclusion is justified in Step 4/Step 5 decision 10: "Only use active behavior sessions" (the decoder task needs behavioral trial outcomes). Multiscope data are knowingly retained (Step 4: "We have a mix of VisualBehavior (single-plane) and VisualBehaviorMultiscope (multi-plane) data … Need to handle both ~31 Hz and ~11 Hz ophys frame rates").

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiments. Mouse IDs are converted to `str` and registered into `subjects` in first-encountered order (experiments are iterated sorted by `ophys_experiment_id`); `subject_idx` records, per output session, the index of its mouse. Result: 38 mice.

ii.
```python
mouse_id = str(exp_row['mouse_id'])
...
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`mouse_id` → `subjects`, `subject_idx` — Map unique mice to indices". Step 2 cross-checks the count (38 downloaded subjects) against the full release (82 mice in Piet et al.) and concludes only a subset was downloaded.

## 1-c. How are the data split into sessions?

i. **One output "session" = one ophys *experiment* (one imaging plane)**, not one `ophys_session_id`. For the 168 single-plane `VisualBehavior` experiments this is identical to a session. For the 34 Multiscope plane-experiments (6 real sessions, mouse 457841) each plane becomes its own output session, so the same behavior/trial structure is duplicated across up to 7 output sessions, and those sessions run at ~11 Hz while the rest run at ~31 Hz. Final count: 202 sessions from 174 real recording sessions; mouse 457841 contributes 34 of the 202 sessions.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])     # one list-of-trials per experiment
    all_output.append(result['output'])
    session_metadata.append({'exp_id': result['exp_id'],
                             'ophys_session_id': result['ophys_session_id'], ...})
```

iii. CONVERSION_NOTES Step 5 decision 9: "**Multiscope handling**: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." The AI noticed the consequence at trajectory step 52 ("the Multiscope sessions have mean T around 91-92 (vs ~260 for Scientifica), because Multiscope records at ~11 Hz") and recorded it in Step 10 Check 5 as an accepted edge case rather than a problem.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is the interval `[start_time, stop_time)`; the ophys frames falling in that half-open window define the trial's time axis (variable length, ~224–389 frames at 31 Hz, ~77–135 at 11 Hz). Go and Catch trials are kept; Aborted and Auto-rewarded are dropped.

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
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 decision 3: "Use `start_time` and `stop_time` from trials table for Go and Catch trials only", following the instruction "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials". Step 3 documents the experimental trial structure (250 ms flash / 500 ms grey, change time drawn from a truncated exponential 2.25–8.25 s), so the full start→stop window covers pre-change flashes plus the response window.

## 1-e. How are trials filtered based on quality controls?

i. Four filters: (1) trial-type filter `(go|catch) & ~aborted & ~auto_rewarded`; (2) trials yielding fewer than 2 ophys frames are skipped; (3) experiments with fewer than 2 valid trials before processing, or fewer than 2 trials after processing, are dropped entirely; (4) experiments with 0 cells or with no stimulus-presentation table are dropped. No engagement/reward-rate filter, no d-prime filter, and no explicit ROI-level filter are applied. Total kept: 51,992 trials over 202 sessions.

ii.
```python
valid_trial_idx = get_valid_trials(trial_data)
if len(valid_trial_idx) < 2:
    print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}")
    return None
...
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
    return None
```

iii. CONVERSION_NOTES Step 3 notes the session-level QC (saturated pixels, photobleaching, z-drift, peak d-prime ≥ 1.0) is "already applied to data in table", so no further session curation was done. Step 10 Check 5: "Sessions with very few valid trials (e.g., 39) are retained if >= 2 trials." The ≥2-trial rule is taken straight from the format spec ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). At trajectory step 41 the AI investigated the 39-trial session and concluded it was legitimate ("1078 aborted trials … the mouse was poorly performing").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The pre-computed dF/F traces stored in the NWB at `processing/ophys/dff/traces/data`, with their time base `processing/ophys/dff/traces/timestamps`. Deconvolved events (`event_detection`) were considered and rejected.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. CONVERSION_NOTES Step 1: "dF/F is pre-computed in NWB files (no need to compute from scratch)"; Step 5 decision 1: "Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." Step 3 records the Allen pipeline's dF/F definition (600 s median-filter baseline, 3.33 s median-filter detrending) confirming it matches the reference processing.

## 2-b. How is the `neural` data processed?

i. No processing at all beyond slicing: the (n_frames, n_cells) array is transposed to (n_cells, n_frames), cast to `float32`, and column-sliced by the trial frame mask. No smoothing, no z-scoring, no baseline subtraction, no neuron-count equalization, no merging across planes (not needed, because each plane is its own session).

ii.
```python
dff = nwb_data['dff_traces']            # (n_cells, n_frames)
n_cells, n_frames = dff.shape
...
neural = dff[:, frame_mask].astype(np.float32)   # (n_cells, n_trial_frames)
...
neural_trials.append(neural)
```

iii. Step 10 Check 3 "dF/F computation | Pre-computed in NWB | Pre-computed (600s median filter + detrending) | Yes" — i.e. the Allen pipeline has already done all processing the reference describes, so nothing further was applied. Brain region is carried as metadata via `targeted_structure` rather than modifying the traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. The AI briefly intended to add a `valid_roi` filter (trajectory step 60: "Let me also add the valid_roi filter for safety") but did not, after concluding the NWB dF/F table already contains only valid ROIs. (Independently checkable: the 29,444 neurons the conversion emits exactly equal the number of rows in `ophys_cells_table.csv` for those 202 experiments, so the claim holds.) The only cell-related rejection is the whole-experiment drop when `n_cells == 0`.

ii.
```python
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```
(there is no other neuron-level filter in the script)

iii. CONVERSION_NOTES Step 1 notes `exclude_invalid_rois=True` is the SDK default and lists the ROI curation rules (non-cell, duplicate, edge, too small/dim). Step 10 Check 3: "ROI filtering | No explicit valid_roi filter | exclude_invalid_rois=True (default) | OK - all ROIs in downloaded NWB files are valid". Step 10 Check 4 cross-checks "Neurons | 29,444 | Matches cells_table.csv | PASS".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** on the native ophys clock: every stream is expressed on `ophys_timestamps`, and the trial is the boolean mask `(ophys_ts >= start_time) & (ophys_ts < stop_time)`. The same mask indexes neural, running and pupil, and the same `trial_ts` vector is used to place stimulus/change events, so all streams are aligned by construction. No resampling of the neural data and no fixed pre/post window around `change_time`; `off_start`/`off_end` are therefore recorded as `None`.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
n_trial_frames = frame_mask.sum()
...
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
pupil_trial = pupil_at_ophys[frame_mask]
```
```python
'temporal_alignment_event': 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.',
'off_start': None,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5 decision 4: "Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time", following the instruction "Temporally align based on ophys timestamp". Step 3 notes all streams are hardware-synchronised ("All streams synced via NI PCI-6612 at 100 kHz"), which is what makes interpolation onto the ophys clock legitimate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning.** Each trial keeps the native ophys frame grid: ~32.3 ms (30.94 Hz) for Scientifica single-plane experiments and ~90 ms (~11 Hz) for the 34 Multiscope plane-experiments. `time_bin_size` in metadata is the **median of the per-session medians = 32.32 ms**, i.e. a single number that is wrong for the ~17% of sessions recorded at 11 Hz. Per-session `dt_ms` is stored in `metadata['session_info']`.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
'ophys_frame_rate_hz': 1000.0 / median_dt,
```

iii. CONVERSION_NOTES Step 5 decision 2: "Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." Step 10 Check 5 lists the mixed rate as an accepted edge case: "Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) - different T per trial." The AI never reconciled this with the format requirement that time bins be the same size for all sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus-presentations table** (`intervals/Natural_Images_*_presentations`): `image_name`, `start_time`, `stop_time`, plus `omitted`. It is *not* derived from the trials table's `initial_image_name` / `change_image_name`. Because the presentation table is used, the variable resolves the real flash structure: it takes the image label only during the 250 ms flashes, and a dedicated `'gray'` category during the 500 ms inter-stimulus grey periods (and during omitted flashes). The category set is built globally by scanning every experiment: `['gray'] + 16 image names` = 17 classes (8 images per session, image sets A/B/G/H across the dataset).

ii.
```python
stim_key = None
for key in f['intervals'].keys():
    if 'Natural_Images' in key or 'natural_images' in key:
        stim_key = key
        break
...
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
```
```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names     # GRAY_LABEL = 'gray'
```

iii. CONVERSION_NOTES Step 5 decision 5: "Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category." Validated in Step 9/10 by the sanity check that the grey fraction is 66.9% vs the 500/750 ms = 66.7% expected from the stimulus protocol, and by a raw-NWB spot check ("Image identity trace | PASS - np.array_equal=True, verified at stimulus onset").

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per trial: initialise the whole trial trace to the `gray` code, then, for every stimulus presentation overlapping the trial window, set all ophys frames in `[stim_start, stim_stop)` to that image's global integer code. `omitted` flashes and names not in the global list are skipped (they stay `gray`). Byte strings from HDF5 are decoded to `str`. The resulting integer row becomes row 0 of the trial's `(5, T)` output matrix.

ii.
```python
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)

for si in range(len(stim_starts)):
    s_start = stim_starts[si]; s_stop = stim_stops[si]
    if s_stop < trial_start:
        continue
    if s_start >= trial_stop:
        break
    name = stim_names[si]
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    if name == 'omitted':
        continue
    if name in image_names_list:
        img_idx = image_names_list.index(name)
    else:
        continue
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The AI's rationale (Step 5 decision 5 and Step 7) is that the stimulus is physically a 250 ms flash followed by 500 ms of grey, so the time-varying label should follow the actual screen contents; the plots in `--show-processing` mode were used to confirm this ("Image identity correctly shows gray during ISI, images during flashes"). The 17-class global mapping makes codes comparable across sessions with different image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same frame grid as the neural data: the helper recomputes the identical trial mask `(ophys_ts >= trial_start) & (ophys_ts < trial_stop)` and assigns labels by comparing each ophys timestamp against the stimulus on/off times, so row 0 of `output` has the same length `T` as the neural matrix, frame for frame.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```
```python
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
output_full[0] = img_trace
```

iii. Step 10 Check 2 sanity check: "Image identity trace | 803736273 | PASS - np.array_equal=True, verified at stimulus onset" — an independent reload of the raw NWB reproduced the trace, including the frame at which each flash begins.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the stimulus-presentations table's `is_change` flag together with the presentation `start_time` (not from the trials table's `change_time`/`go`). Since `is_change` is only true when the displayed image actually differs from the previous flash, catch (sham-change) trials produce no positive samples — verified in the converted sample where 34 of 39 trials contain a change, matching the 34 go / 5 catch split.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    s_start = stim_starts[si]
    if s_start < trial_start or s_start >= trial_stop:
        continue
```

iii. CONVERSION_NOTES Step 5 mapping table: "`is_change` from stimulus presentations → `output[1]`: image_change — Binary, 1 at change onset frame, 0 otherwise | Time-varying". Using the presentation-level flag keeps the change marker on the same event grid as image identity, so `image_change == 1` coincides exactly with the frame where row 0 switches image code.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector of length `T` is created and a **single frame** is set to 1 — the first ophys frame at or after the change flash onset (`np.searchsorted`). The pulse is not extended over the flash duration or the following grey period. Consequence: only ~0.37% of all timepoints are labelled `change` (≈1 positive frame in ~260).

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
    # Find the first ophys frame at or after the change onset
    frame_idx = np.searchsorted(trial_ts, s_start)
    if frame_idx < n_frames:
        trace[frame_idx] = 1
```

iii. The AI read the instruction "Have value of 1 right after a change in image identity, otherwise 0" literally as a one-frame event ("1 at change onset frame, 0 otherwise", Step 5). When the resulting imbalance showed up it was explicitly assessed and accepted rather than changed — Step 12: "**image_change** (1.27x): Extreme class imbalance (99.6% no_change vs 0.4% change). Balanced accuracy penalizes heavily. The decoder correctly identifies change timepoints but the signal is very sparse. Not a conversion bug."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is natively binary, so no thresholding is needed: values are `{0, 1}` with `output_values[1] = ['no_change', 'change']`.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    ...
]
...
output_full[1] = change_trace
```

iii. Follows directly from the instruction that image change is a binary variable; no justification beyond that is given in CONVERSION_NOTES.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same trial mask as the neural data; the change frame index is found with `np.searchsorted(trial_ts, s_start)` on the trial's ophys timestamps, i.e. the first imaging frame at or after the physical change onset. Row 1 of the output matrix therefore has length `T` and is frame-synchronous with the dF/F.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    ...
    frame_idx = np.searchsorted(trial_ts, s_start)
    if frame_idx < n_frames:
        trace[frame_idx] = 1
```

iii. Step 5 planned sanity check "Check that image_change=1 aligns with actual change in image_identity"; the `--show-processing` plots (`processing_775614751.png`, `processing_788490510.png`) put the change trace next to the image-identity trace on a common time axis, and Step 7 reports "Change events aligned to stimulus onset."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. The filtered running speed and its timestamps: `processing/running/speed/data` and `processing/running/speed/timestamps` (60 Hz wheel encoder, already 10 Hz low-pass filtered by the Allen pipeline). The unfiltered variant is not used.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. CONVERSION_NOTES Step 3: "**Running speed**: 10 Hz lowpass Butterworth filter" is listed as part of the reference pipeline, and Step 10 Check 3 records "Running speed | Interpolated to ophys timestamps | SDK also interpolates | Yes" — i.e. the stored `speed` stream is the same object the SDK exposes as `running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps: (1) linear interpolation of the 60 Hz speed trace onto the full session's ophys timestamps (`bounds_error=False, fill_value=np.nan`, so samples outside the wheel-encoder coverage become NaN); (2) discretisation into 5 percentile bins (see 5-c). The interpolation is done once per session on the whole session, then sliced per trial.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    if len(signal) == 0 or len(ophys_ts_trial) == 0:
        return np.full(len(ophys_ts_trial), np.nan)
    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)

running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
```

iii. CONVERSION_NOTES Step 5 decision 7: "Interpolate from 60 Hz to ophys timestamps using linear interpolation" — downsampling a 10 Hz-bandlimited signal to ~31 Hz by linear interpolation preserves the waveform, and doing it once per session (rather than per trial) guarantees identical treatment of every trial.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count percentile bins (0–20–40–60–80–100th percentile), with bin edges computed **per session** from all non-NaN interpolated values of that session, then applied to each trial of that session. Outer edges are set to ±inf so nothing falls outside, and NaN samples are assigned bin 0. Because the edges are per-session, each session's running label is uniform by construction (globally: 0.190/0.199/0.204/0.208/0.200).

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result

running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. CONVERSION_NOTES Step 5 decision 8: "Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)." This implements the instruction "discretized into five equal percentile bins"; computing over the whole session rather than per trial avoids a trial's bins depending on that trial's own (short, possibly constant) speed profile. Step 12 confirms the result: "running_speed: Well-balanced (19-21% per bin)".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The interpolation target is the session's ophys timestamp vector itself, so the binned trace lives on the neural frame grid; the trial slice uses the identical `frame_mask` used for the dF/F.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
output_full[2] = running_binned
```

iii. Step 5 decision 4 (align everything to ophys timestamps) plus Step 10 Check 2 sanity check "Running speed bins | 803736273 | PASS - np.array_equal=True" — recomputed from the raw NWB and compared with `np.array_equal`. The `--show-processing` panels overlay the raw cm/s trace and the binned trace on the same trial time axis.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the DeepLabCut ellipse-fit pupil **area**, 30 Hz) with its `timestamps`, plus `acquisition/EyeTracking/likely_blink/data` to mask blinks. If the `EyeTracking` group or `pupil_tracking` is missing, the pupil signal is set to all-NaN for that session (the session is still kept).

ii.
```python
if 'EyeTracking' in f.get('acquisition', {}):
    et = f['acquisition']['EyeTracking']
    if 'pupil_tracking' in et:
        pt = et['pupil_tracking']
        data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
        data['pupil_timestamps'] = pt['timestamps'][:]
        data['likely_blink'] = et['likely_blink']['data'][:]
    else:
        data['pupil_area'] = None
else:
    data['pupil_area'] = None
```

iii. CONVERSION_NOTES Step 2 lists the available eye-tracking fields ("area, height, width @ 30 Hz") and Step 3 records "**Pupil**: DeepLabCut tracking, ellipse fit, blinks detected and set to NaN". Area was chosen (rather than the ellipse width/height) so that a single isotropic diameter could be derived — see 6-b.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Frames flagged `likely_blink` are set to NaN (the stored `area` is in fact already NaN there, so this is belt-and-braces); (2) area is converted to an equivalent-circle diameter `d = 2*sqrt(area/pi)` for the positive, non-NaN samples; (3) the diameter trace — **with the NaN gaps left in** — is linearly interpolated onto the ophys timestamps, which means blink periods come out as NaN rather than being bridged; (4) the result is percentile-binned (6-c), with NaN → bin 0. Empirically ≈5% of frames are blink-NaN in a typical session (measured 5.3% at the ophys grid for experiment 775614751), and 3 of the 202 sessions have no usable eye tracking at all and therefore get bin 0 at every timepoint.

ii.
```python
pupil_area = nwb_data['pupil_area'].copy()
likely_blink = nwb_data['likely_blink']
pupil_ts = nwb_data['pupil_timestamps']

# Set blink frames to NaN
pupil_area[likely_blink] = np.nan

# Compute diameter from area: diameter = 2 * sqrt(area / pi)
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)

pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. CONVERSION_NOTES Step 5 decision 6: "Compute from pupil area as `2*sqrt(area/pi)`. Set blink frames to NaN, then interpolate. Discretize non-NaN values", and Step 10 Check 3 claims the formula matches the whitepaper. On the NaN handling the AI made an explicit, documented choice (trajectory step 62): "since pupil_diameter is specified as 'discretized into five equal percentile bins', mapping NaN to bin 0 is a reasonable choice (lowest bin). A separate blink category would add a 6th class which isn't specified."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 5 equal-count percentile bins with edges computed per session over the non-NaN interpolated diameters, outer edges ±inf, NaN → bin 0. Global marginals end up at 0.200/0.199/0.214/0.221/0.166 (slightly non-uniform because of the blink/missing frames folded into bin 0 and of ties).

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
output_full[3] = pupil_binned
```

iii. Same rationale as running speed (Step 5 decision 8). Step 12 acknowledges the resulting skew: "pupil_diameter: Slightly unbalanced due to blink frames mapped to bin 0."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. As with running speed: interpolated once per session onto `ophys_timestamps` and sliced with the same `frame_mask` as the dF/F, so row 3 of the output is frame-synchronous with the neural matrix.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Step 5 decision 4 (everything on the ophys clock; the streams are hardware-synchronised per Step 3). Step 10 Check 2 sanity check: "Pupil diameter bins | 775614751 | PASS - np.array_equal=True, blinks=NaN confirmed".

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — evaluated for the trial's own index.

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
    else:
        return 'unknown'
```

iii. CONVERSION_NOTES Step 5 mapping table: "Trial outcome (hit/miss/FA/CR) → `output[4]`: trial_outcome — Categorical, static per trial | 4 categories". These are the canonical change-detection outcome labels; because aborted and auto-rewarded trials are already excluded, every retained go/catch trial carries exactly one of the four flags (independently verified: 0 unlabelled and 0 doubly-labelled trials across the first 25 active experiments, 5,276 trials).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to its index in the fixed list `['hit','miss','false_alarm','correct_reject']` (0–3) and then **broadcast across every timepoint of the trial**, so that the static variable occupies row 4 of the same `(5, T)` matrix as the time-varying ones. Anything that failed to match falls back to index 0 (i.e. would be silently labelled `hit`). Resulting distribution: hit 31.0%, miss 56.5%, FA 1.8%, CR 10.7%.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
output_full[0] = img_trace
output_full[1] = change_trace
output_full[2] = running_binned
output_full[3] = pupil_binned
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The in-code comments show the AI weighing the "(n_output, n_timepoints) or (n_output,)" ambiguity in the spec and choosing to emit one uniform `(5, T)` array per trial, with the static outcome constant over time. Step 9 validates the distribution against expectations: "Outcome: hit rate ~30% typical / 30.7% | Yes", "miss ~57% / 56.8%", "FA ~2% / 1.8%", "CR ~11% / 10.7%".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is filter-or-impute, with no global exception guard:
- Missing NWB file → warn and skip the experiment; 0 cells → skip; no stimulus-presentation table → skip; <2 valid trials (before or after processing) → skip.
- Missing eye-tracking group / `pupil_tracking` → pupil set to all-NaN for the whole session, and the session is **still emitted** (3 of 202 sessions therefore carry a constant `pupil = bin_0` label).
- NaN running/pupil samples (out-of-range interpolation, blinks) → bin 0, conflated with the lowest-value bin.
- Trials with <2 ophys frames → dropped; the half-open `< stop_time` mask means the last trial never runs past the end of the recording.
- HDF5 byte-string image names are decoded; unrecognised image names and `omitted` flashes fall back to `gray`.
- Unrecognised trial outcome → index 0 (`hit`). This fallback is latent (never triggered on the sampled active sessions) but would mislabel rather than flag.
- There is **no** `try/except` around per-experiment loading, so a corrupt file would abort the whole run.

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}")
    return None
if n_cells == 0: ...
    return None
if stim_data is None:
    print(f"  WARNING: No stimulus data in experiment {exp_id}")
    return None
if len(valid_trial_idx) < 2: ...
    return None
```
```python
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0  # NaN gets bin 0 (lowest bin)
```
```python
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```

iii. CONVERSION_NOTES Step 10 Check 5: "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)", elaborated at trajectory step 62 (adding a 6th "blink" class would deviate from the specified 5 bins). The skip-on-missing rules are justified by the format requirement of ≥2 trials per session and by the need for a stimulus table to build image identity.

## 9-a. What are the most time-consuming steps of the code?

i. Timing is printed per experiment and per phase. Over the full run (544.7 s = 9.1 min for 202 experiments): **NWB reading dominates at 380.8 s (70%)** — averaging ~1.7–1.9 s per experiment for `dff/traces/data` (up to 666 cells × ~140k frames), running, eye tracking and the stimulus table; **per-trial processing is 138.0 s (25%)**, ~0.5 s per experiment; the up-front `get_all_image_names` prescan costs 14.4 s (2.6%); assembling and pickling the 8.5 GB output takes the remaining ~10 s.

ii.
```python
t0 = time.time()
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
t_process = time.time() - t0 - t_load
...
print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, "
      f"dt: {result['dt']*1000:.1f}ms, "
      f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")
```

iii. CONVERSION_NOTES Step 7 extrapolated from the sample ("Load NWB ~1.7s ... Total ~750s (~12.5 min)") and concluded the run would come in under the 15-minute budget, so no parallelisation or I/O optimisation was pursued; the realised 9.1 min beat that estimate.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial Python loops scan the **entire** stimulus-presentation table for every trial:
- `build_image_identity_trace`: `for si in range(len(stim_starts))` — it `break`s once past the trial, but always restarts at presentation 0, so with ~250 trials × ~4,800 presentations this is O(n_trials × n_stim) ≈ 10^6 iterations per session, each doing a `list.index()` lookup (linear) and building a fresh boolean mask over the trial timestamps.
- `build_image_change_trace`: same full scan, again from index 0.
Both could be replaced by two `np.searchsorted` calls per presentation (or a single vectorised `np.searchsorted(ophys_ts, stim_starts/stops)` over the whole session, filling the label array once and then slicing per trial). The `image_names_list.index(name)` lookups could be a dict, and the per-trial `(ophys_ts >= t_start) & (ophys_ts < t_stop)` full-length boolean mask (built 3× per trial over ~140k timestamps) could be `np.searchsorted` index ranges. The outer experiment loop is embarrassingly parallel and could have used multiprocessing.

ii.
```python
for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    if s_stop < trial_start:
        continue                     # restarts from si=0 for every trial
    if s_start >= trial_stop:
        break
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)   # in process_experiment
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)  # again in build_image_identity_trace
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)  # again in build_image_change_trace
```

iii. The AI did not discuss vectorising these loops; its efficiency argument (Step 7) was purely that the projected total runtime was inside the 15-minute budget, which the measured 138 s of processing time (25% of runtime) supports.

## 9-c. What processing does the code repeat multiple times?

i. Three repetitions:
1. **Every NWB file is opened twice**: once by `get_all_image_names`, which reads the whole `image_name` column of all 202 files just to collect the 16 label strings (14.4 s), and again in `process_experiment`.
2. **The trial frame mask is computed three times per trial** over the full ~140k-sample timestamp vector — once in `process_experiment`, once inside `build_image_identity_trace`, once inside `build_image_change_trace` — and `trial_ts` is re-derived each time.
3. **The stimulus table is re-scanned from the beginning for every trial** in both helpers (see 9-b), and `image_names_list.index()` is re-evaluated for every presentation.

ii.
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        with h5py.File(nwb_path, 'r') as f:
            for key in f['intervals'].keys():
                if 'Natural_Images' in key or 'natural_images' in key:
                    names = f['intervals'][key]['image_name'][:]
```
```python
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
# each recomputes the same trial mask that process_experiment already has
```

iii. Not discussed in CONVERSION_NOTES; the prescan exists so that the categorical image code mapping is global and consistent across sessions (it prints "Found 16 unique images ... Time: 14.4s"), which is a correctness requirement — but it could have been satisfied from the `image_set` metadata column or by a single pass that also cached the data.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none expensive:
- `discretize_percentile()` is fully implemented (percentile edges + digitize) and **never called** — dead code superseded by `compute_session_percentile_edges`/`apply_percentile_bins`.
- `cell_roi_ids` is read from the image-segmentation group on every file and never used.
- Trials-table columns `catch` (used only inside the `go|catch` mask, redundant with `~aborted & ~auto_rewarded` in practice), `change_time`, `initial_image_name`, `change_image_name` and `is_change` are read from every NWB but never used — image identity and change are taken from the stimulus table instead.
- `stim_data['omitted']` is read but never used (omission is inferred from `image_name == 'omitted'`).
- `output_static = np.array([outcome_idx])` is constructed and then discarded in favour of broadcasting into `output_full[4]`.
- A per-session `dt` is computed and stored for all 202 sessions, but only the median is written to `metadata['time_bin_size']`.
- `--show-processing` keeps whole-session arrays (`ophys_ts`, `running_at_ophys`, `pupil_at_ophys`, `trial_data`, `stim_data`) alive in the result dict; harmless in the default path since the flag gates it.
- The trial outcome, a single value per trial, is materialised as a full `T`-length row (a deliberate format choice, but it multiplies that row's storage by ~260×).

ii.
```python
def discretize_percentile(values, n_bins=5):
    """Discretize values into n_bins equal percentile bins."""
    ...                      # never called anywhere in the script
```
```python
if 'image_segmentation' in f['processing']['ophys']:
    seg = f['processing']['ophys']['image_segmentation']
    for key in seg.keys():
        if 'id' in seg[key]:
            data['cell_roi_ids'] = seg[key]['id'][:]   # never used
            break
```
```python
output_static = np.array([outcome_idx], dtype=np.int64)   # never used
...
output_full[4] = outcome_idx
```

iii. Not discussed in CONVERSION_NOTES; the comment block around the output assembly shows the AI reasoning through the "(n_output, n_timepoints) or (n_output,)" ambiguity in the spec and abandoning the separate static array in favour of the uniform `(5, T)` layout, which is what leaves `output_static` orphaned.
