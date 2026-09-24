# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK `VisualBehaviorOphysProjectCache` API. Instead it reads the project metadata CSV (`project_metadata/ophys_experiment_table.csv`) with pandas, globs the locally available NWB files in `behavior_ophys_experiments/`, intersects the two (so only downloaded experiments are processed), drops every session whose `session_type` contains "passive", and sorts by `ophys_experiment_id` for reproducibility. Each remaining experiment is then opened directly with `h5py` and the relevant HDF5 datasets (ophys timestamps, dF/F traces, running speed, eye tracking, trials table, stimulus presentation table) are read into memory in one pass. No `project_code` filter is applied, so both `VisualBehavior` (single-plane, ~31 Hz) and `VisualBehaviorMultiscope` (8-plane, ~11 Hz) experiments are loaded. Result: 202 experiments / 38 mice / 51,992 trials / 29,444 neurons.

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
        ...
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. From CONVERSION_NOTES Step 6: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)". The AI documented in Step 1 that dF/F and events are pre-computed inside the NWB files, so the SDK adds no processing that is needed here; reading HDF5 directly avoids SDK object-construction overhead and let the full conversion finish in 9.1 min. It documented in Step 2/4 that only 284 of 1,936 released experiments are present locally, so the experiment table must be intersected with the files on disk. Passive sessions are excluded because "Only use active behavior sessions" — the decoder has to predict trial outcome and the mouse does not perform the task in OPHYS_2/OPHYS_5.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values from the experiment table. Mice are registered lazily, in the order in which their first experiment is processed (experiments are sorted by `ophys_experiment_id`), into `data['subjects']` (as strings), and `data['subject_idx']` stores the subject index for every session. 38 mice result.

ii.
```python
mouse_id = str(exp_row['mouse_id'])        # in process_experiment()
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
...
all_subject_idx.append(subject_map[mouse_id])
...
'subjects': all_subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The notes' variable-mapping table gives: "`mouse_id` → `subjects`, `subject_idx`; Map unique mice to indices". `mouse_id` is the canonical animal identifier in the Allen metadata table, and the AI cross-checked the count (38 downloaded mice vs. 82 in the full release) in Step 4.

## 1-c. How are the data split into sessions?

i. One **ophys experiment (imaging plane)** = one "session" in the output. The AI deliberately does *not* group experiments by `ophys_session_id`: for the 8-plane Multiscope recordings, each plane becomes its own entry in `data['neural']`, so the same behavioral session (same trials, same running/pupil/image/outcome labels) is emitted up to 8 times with different neurons. This is visible in the verification output (mouse 457841 has 34 "sessions"; blocks of 7–8 consecutive sessions have identical trial counts, e.g. 209 × 7, 287 × 7, 309 × 7). Passive sessions are dropped. Because Multiscope planes are sampled at ~10.7 Hz and Scientifica at ~31 Hz, the emitted sessions have two different time-bin sizes (168 sessions at 32.3 ms, 34 at 93.2 ms).

ii.
```python
# no grouping by ophys_session_id: the loop is over experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])     # one session entry per experiment
    all_output.append(result['output'])
    ...
    session_metadata.append({
        'exp_id': result['exp_id'],
        'ophys_session_id': result['ophys_session_id'],
        'session_type': result['session_type'],
        ...
        'dt_ms': result['dt'] * 1000,
    })
```

iii. Step 5, key decision 9: "**Multiscope handling**: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." Decision 10: "**Passive sessions excluded**: Only use active behavior sessions." In Step 10 Check 5 the AI notes the consequence — "Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) — different T per trial" — and in Step 52 of the trajectory it observed the two T distributions (~91 vs ~260 frames) but did not change the design.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is valid if it is a Go **or** Catch trial and is neither aborted nor auto-rewarded. Each trial's window is the full `[start_time, stop_time)` interval, and the ophys frames falling in that window define the trial (variable length: 217–391 frames at 31 Hz, ~77–135 frames at 11 Hz). Trials with fewer than 2 ophys frames are dropped.

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

iii. Step 5, key decision 3: "**Trial definition**: Use `start_time` and `stop_time` from trials table for Go and Catch trials only." This follows the Decoder Task instruction verbatim ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"). Using the full trial window (rather than a fixed window around the change) keeps the pre-change flashes and the post-change response window inside the trial, which is what makes the time-varying image-identity and image-change outputs meaningful.

## 1-e. How are trials filtered based on quality controls?

i. Filtering happens at three levels:
- **Session/experiment level**: missing NWB file, zero cells, missing stimulus-presentation table, or fewer than 2 valid trials → the experiment is skipped with a printed WARNING. Passive sessions are already removed from the experiment table.
- **Trial level**: aborted and auto-rewarded trials removed; only Go/Catch kept; trials whose ophys window contains <2 frames removed (this also handles trials running past the end of the recording, since the boolean mask simply returns fewer frames).
- **Post-processing**: if fewer than 2 trials survive processing, the whole experiment is discarded.
No behavioral-engagement (reward-rate) filter and no d′ filter is applied.

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}");  return None
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}");  return None
if stim_data is None:
    print(f"  WARNING: No stimulus data in experiment {exp_id}");  return None
valid_trial_idx = get_valid_trials(trial_data)
if len(valid_trial_idx) < 2:
    print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}");  return None
...
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
    return None
```

iii. Step 3 "Trial curation rules (for this decoder task): Include Go and Catch trials; Exclude Aborted and Auto-rewarded trials", taken from the instructions. Step 3 also records that Allen has already applied session-level QC ("<1000 saturated pixels, <20% photobleaching, correct targeting, <10 µm z-drift, peak d-prime ≥ 1.0"), so no further session QC was added. The ≥2-trial rule comes from the target-format requirement "There needs to be at least two trials within each session". Step 10 Check 5 explicitly accepts poorly-performing sessions: "Sessions with very few valid trials (e.g. 39) are retained if >=2 trials"; the AI investigated the 39-trial session in the trajectory and confirmed it had 1078 aborted trials, i.e. genuine behavior, not a bug.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The pre-computed dF/F traces in the NWB file: `processing/ophys/dff/traces/data`, with their own timestamps `processing/ophys/dff/traces/timestamps`. Stored as (n_frames, n_cells) in the file and transposed to (n_cells, n_frames). Deconvolved events were considered (documented in Step 1) but not used.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. Step 5, key decision 1: "**Neural data**: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." Step 1 established that dF/F is already computed in the NWB with the Allen pipeline (600 s median-filter baseline, 3.33 s median-filter detrending), so nothing needs to be recomputed.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the traces are transposed to (n_neurons, n_timepoints), sliced per trial, and cast to `float32`. There is no normalization, smoothing, z-scoring, or resampling. Because each experiment is its own session, no cross-plane stacking is done. Every neuron of a session is assigned one brain region taken from the experiment table's `targeted_structure` (VISp or VISl); imaging depth is kept only in `metadata['session_info']`, not in the region label.

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
...
brain_region = exp_row['targeted_structure']
...
all_brain_region_idx.append(
    np.full(result['n_cells'], region_map[brain_region], dtype=np.int64)
)
```

iii. Step 10 Check 3 records the comparison with the reference: "dF/F computation | Pre-computed in NWB | Pre-computed (600s median filter + detrending) | Yes". Since the Allen pipeline has already motion-corrected, neuropil-corrected and normalized the traces, the AI deliberately added nothing. `float32` was chosen for memory (the pickle is still 8.5 GB).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering at all. The AI checked whether the SDK's `exclude_invalid_rois=True` behaviour needed replicating and concluded it was a no-op for the published NWB files.

ii. N/A — there is no filtering code. The only related code is the (unused) read of ROI ids:
```python
if 'image_segmentation' in f['processing']['ophys']:
    seg = f['processing']['ophys']['image_segmentation']
    for key in seg.keys():
        if 'id' in seg[key]:
            data['cell_roi_ids'] = seg[key]['id'][:]
            break
```

iii. Step 1 flagged `exclude_invalid_rois` as the SDK's curation step; Step 10 Check 3 resolves it: "ROI filtering | No explicit valid_roi filter | exclude_invalid_rois=True (default) | OK - all ROIs in downloaded NWB files are valid". (I confirmed this independently: in a 25-file sample, `valid_roi` is True for every ROI and the number of dF/F columns equals the number of ROIs.) The total of 29,444 neurons was cross-checked against `ophys_cells_table.csv` in Step 10 Check 4.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Everything is aligned to the ophys timestamps, as the instructions require ("Temporally align based on ophys timestamp"). For each trial a boolean mask `(ophys_ts >= start_time) & (ophys_ts < stop_time)` selects the frames; the same mask is used to slice the neural matrix and all behavioral outputs, so the streams are aligned by construction. The alignment event recorded in metadata is the trial start; `off_start`/`off_end` are `None` because the trial length is variable.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
pupil_trial = pupil_at_ophys[frame_mask]
...
'temporal_alignment_event': 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.',
'off_start': None,
'off_end': None,
```

iii. Step 5, key decision 4: "**Temporal alignment**: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time." Step 10 Check 2 verified the alignment with a spot-check against the raw NWB (`np.allclose` on dF/F at a specific trial/frame/cell, max_diff = 0.0) and by checking that the image-identity trace switches at stimulus onsets.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of the neural data: the native ophys frame rate is kept. Consequently the dataset contains **two different bin sizes**: 32.3 ms (~31 Hz, 168 Scientifica sessions) and 93.2 ms (~10.7 Hz, 34 Multiscope plane-sessions). `metadata['time_bin_size']` is set to the **median across sessions**, 32.32 ms, which is wrong for the 34 Multiscope sessions; the true per-session value is preserved only in `metadata['session_info'][i]['dt_ms']`.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # per-experiment time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'time_bin_size': median_dt,
'ophys_frame_rate_hz': 1000.0 / median_dt,
```

iii. Step 5, key decision 2: "**Time bin**: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." The AI noticed the mixed rates during Step 9 (trajectory step 52: "the Multiscope sessions have mean T around 91-92 (vs ~260 for Scientifica), because Multiscope records at ~11 Hz") and recorded it as an accepted edge case in Step 10 Check 5, but did not harmonize the bins or restrict the project code.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentation table** `intervals/Natural_Images_*_presentations`, specifically `image_name`, `start_time`, `stop_time` and `omitted` — not the trials table. The set of category labels is built by scanning every NWB file once for the unique non-`omitted` image names (16 images across image sets A and B), and prepending an extra `'gray'` category for the inter-stimulus interval, giving 17 classes.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
```

```python
def get_all_image_names(exp_table):
    all_names = set()
    for _, row in exp_table.iterrows():
        ...
        names = f['intervals'][key]['image_name'][:]
        for n in names:
            if isinstance(n, bytes): n = n.decode('utf-8')
            if n != 'omitted': all_names.add(n)
    return sorted(all_names)
...
image_names_list = [GRAY_LABEL] + all_image_names
```

iii. Step 5 mapping table: "`image_name` from stimulus presentations → `output[0]`: image_identity; Map to categorical integer, time-varying at ophys rate; 8 images + gray screen". The presentations table gives the exact on/off time of every flash, which the trials table does not, so it supports a genuinely time-varying representation including the omitted flashes (5% omission rate, documented in Step 3).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial the trace is initialised to the `gray` code (index 0) and then, for every stimulus presentation overlapping the trial window, the frames falling inside `[start_time, stop_time)` of that flash are set to the image's index in the global name list. `omitted` flashes are skipped, so they stay `gray`. The resulting distribution is 66.9% gray and ~2% per image, matching the 250 ms-on / 500 ms-off stimulus cycle.

ii.
```python
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
for si in range(len(stim_starts)):
    s_start, s_stop = stim_starts[si], stim_stops[si]
    if s_stop < trial_start: continue
    if s_start >= trial_stop: break
    name = stim_names[si]
    if isinstance(name, bytes): name = name.decode('utf-8')
    if name == 'omitted':        # omitted flashes are gray screen
        continue
    if name in image_names_list:
        img_idx = image_names_list.index(name)
    else:
        continue
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. Step 5, key decision 5: "**Image identity**: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category." The AI validated the choice statistically in Step 9/10: "Gray screen fraction | 66.9% | 66.7% (500/750ms) | PASS" and in the trajectory: "the gray screen fraction (66.9%) matches the expected 66.7%". A single global (sorted) name list is used so codes are consistent across sessions and image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The trace is computed on exactly the same ophys frames as the neural data: the function re-derives the identical trial mask from `ophys_ts`, and within the trial each frame is assigned by comparing its ophys timestamp to the flash on/off times.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. Same rationale as 2-d: everything is put on the ophys clock, and the NWB streams are hardware-synchronised (Step 3: "Synchronization: All streams synced via NI PCI-6612 at 100 kHz"). Step 10 Check 2 spot-checked the trace against raw NWB stimulus times ("Image identity trace | PASS - np.array_equal=True, verified at stimulus onset").

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of the stimulus presentation table, together with the flash `start_time`. `is_change` is True only for the first flash of a genuinely new image, so catch (sham-change) trials contain no positive samples.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
```

iii. Step 5 mapping table: "`is_change` from stimulus presentations → `output[1]`: image_change; Binary, 1 at change onset frame, 0 otherwise; Time-varying". Using the presentations table's own change flag avoids having to re-derive the change from `change_time`/`go` in the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector of length n_frames, with a **single** element set to 1: the first ophys frame at or after the change flash onset (`np.searchsorted`). One frame therefore corresponds to 32 ms on Scientifica sessions and 93 ms on Multiscope sessions. Positive class fraction in the full dataset: 0.4%.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
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

iii. The instruction reads "Have value of 1 right after a change in image identity, otherwise 0", and the AI implemented the most literal reading (one frame "right after" the change). It was aware of the consequence and defended it in Step 12: "**image_change** (1.27x): Extreme class imbalance (99.6% no_change vs 0.4% change). Balanced accuracy penalizes heavily. The decoder correctly identifies change timepoints but the signal is very sparse. Not a conversion bug." Step 10 Check 4 also treated the fraction as a sanity check: "Image change fraction | 0.372% | ~0.3-0.4% | PASS".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is natively binary (`no_change` = 0, `change` = 1) and is stored as int64 with `output_values[1] = ['no_change', 'change']`.

ii.
```python
output_values = [
    image_names_list,          # image identity categories
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    ...
]
```

iii. The Decoder Task specifies image change as a binary variable, so the AI stored it directly as a 0/1 indicator time series, consistent with the format note "If an output is a time such as when a behavior occurred, represent it as a binary time series."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same frame grid as the neural data: the function recomputes the identical trial mask over `ophys_ts`, and the change is placed at `np.searchsorted(trial_ts, change_onset)`, i.e. the first ophys frame at or after the physical change onset (maximum alignment error = one frame).

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. Same ophys-clock rationale as above. The `--show-processing` plots were used to confirm visually that the change marker coincides with the step in the image-identity trace (Step 7: "Change events aligned to stimulus onset"), and Step 5 listed the planned check "Check that image_change=1 aligns with actual change in image_identity".

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed` (data + timestamps) from the NWB file — the Allen-filtered running speed (10 Hz low-pass Butterworth), sampled at 60 Hz, not `speed_unfiltered`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Step 3 Processing Details: "**Running speed**: 10 Hz lowpass Butterworth filter" — the AI recorded that the published `speed` stream is the processed one used by the SDK, so it reads it as-is.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The 60 Hz trace is linearly interpolated onto the full session's ophys timestamps once per experiment (`bounds_error=False, fill_value=np.nan`), then sliced per trial with the same frame mask as the neural data, then discretised (see 5-c).

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    if len(signal) == 0 or len(ophys_ts_trial) == 0:
        return np.full(len(ophys_ts_trial), np.nan)
    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)

running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
```

iii. Step 5, key decision 7: "**Running speed**: Interpolate from 60 Hz to ophys timestamps using linear interpolation." Interpolating the whole session once (rather than per trial) is also the efficient choice. Step 10 Check 2 verified the resulting bins against the raw NWB ("Running speed bins | PASS - np.array_equal=True").

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (0–20th, 20–40th, …, 80–100th). The percentile edges are computed **per session**, from all non-NaN interpolated samples of the **whole session** (including inter-trial periods), with the outer edges replaced by ±inf; the edges are then applied to the trial slices with `np.digitize` + `np.clip`. NaN → bin 0. Resulting full-dataset distribution: 0.190 / 0.199 / 0.204 / 0.208 / 0.200.

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

iii. Step 5, key decision 8: "**Percentile bins for running/pupil**: Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)." This implements the Decoder Task requirement "discretized into five equal percentile bins" and normalises out between-session differences in wheel calibration / behavioural state. Step 12 confirms the balance: "running_speed: Well-balanced (19-21% per bin)".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. By construction: the speed is interpolated directly onto `ophys_ts`, and the trial slice uses the same `frame_mask` as `dff`, so index *t* of the running row is the same ophys frame as column *t* of the neural matrix.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
```

iii. The AI's Step 5 alignment decision (align everything to the ophys timebase) plus Step 3's note that all streams are hardware-synchronised at 100 kHz, which makes cross-stream interpolation valid. The `--show-processing` plots overlay the raw speed and the binned trace on the trial time axis to make the alignment visually checkable.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the ellipse-fit pupil area, 30 Hz) with its timestamps, plus `acquisition/EyeTracking/likely_blink` to identify blinks. Diameter is derived from area; the available `width`/`height` columns are not used.

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

iii. Step 5 mapping table: "`pupil_tracking/area` → diameter → `output[3]`: pupil_diameter_bin". Step 3 records the Allen processing chain ("Pupil: DeepLabCut tracking, ellipse fit, blinks detected and set to NaN"), and Step 10 Check 3 claims the area→diameter conversion is "Same formula in whitepaper".

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, diameter is computed as the equivalent-circle diameter `2*sqrt(area/pi)` for positive areas, and the resulting trace is linearly interpolated onto the ophys timestamps; NaNs are then mapped to bin 0 at discretisation time. Because the NaNs are left **inside** the array passed to `interp1d` (rather than the blink samples being dropped first), the blink gaps are not bridged — `interp1d` returns NaN for every ophys frame that falls between two samples where one is NaN. Measured on experiment 1007107386: 9.19% of eye-tracking samples are blinks (they are already NaN in the NWB `area` dataset) and 9.54% of ophys frames end up NaN, versus 0.22% if the blink rows are dropped before interpolation as the reference does. Those frames all become pupil bin 0, which inflates bin 0 to 27.8% in that session. In addition, 3 of the 284 NWB files have no eye-tracking group at all; for those the pupil row is all-NaN → constant bin 0 for the whole session (visible in the verification log as sessions with pupil range [0.0, 0.0]).

ii.
```python
if nwb_data['pupil_area'] is not None:
    pupil_area = nwb_data['pupil_area'].copy()
    likely_blink = nwb_data['likely_blink']
    pupil_ts = nwb_data['pupil_timestamps']
    pupil_area[likely_blink] = np.nan                 # blinks -> NaN
    pupil_diameter = np.full_like(pupil_area, np.nan)
    valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
    pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
    pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. Step 5, key decision 6: "**Pupil diameter**: Compute from pupil area as `2*sqrt(area/pi)`. Set blink frames to NaN, then interpolate. Discretize non-NaN values." The AI explicitly revisited the NaN policy in the trajectory (step 62): "since pupil_diameter is specified as 'discretized into five equal percentile bins', mapping NaN to bin 0 is a reasonable choice (lowest bin). A separate blink category would add a 6th class which isn't specified. I'll leave it as is since it's a valid design choice", and in Step 10 Check 5: "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug".

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five equal-percentile bins with per-session edges computed from all non-NaN session samples, applied per trial, NaN → bin 0. Full-dataset distribution: 0.200 / 0.199 / 0.214 / 0.221 / 0.166.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Same as 5-c (Step 5, key decision 8) — the Decoder Task requires five equal percentile bins, and per-session percentiles normalise out differences in camera zoom/rig between sessions so that a bin means the same thing (relative arousal) everywhere. The AI noted the residual imbalance in Step 12: "pupil_diameter: Slightly unbalanced (16.6% to 22.1%)... Slightly unbalanced due to blink frames mapped to bin 0."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the session's ophys timestamps and sliced with the same trial `frame_mask` as the neural data, so pupil sample *t* and neural column *t* are the same ophys frame.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same ophys-timebase decision as running speed (Step 5, key decision 4); Step 10 Check 2 spot-checked the pupil bins against the raw NWB for session 775614751 ("PASS - np.array_equal=True, blinks=NaN confirmed").

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order.

ii.
```python
def get_trial_outcome(trial_data, idx):
    if trial_data['hit'][idx]:            return 'hit'
    elif trial_data['miss'][idx]:         return 'miss'
    elif trial_data['false_alarm'][idx]:  return 'false_alarm'
    elif trial_data['correct_reject'][idx]: return 'correct_reject'
    else:                                 return 'unknown'
```

iii. Step 5 mapping table: "Trial outcome (hit/miss/FA/CR) → `output[4]`: trial_outcome; Categorical, static per trial; 4 categories". These are the canonical go/no-go outcome labels of the change-detection task; the AI cross-checked the resulting rates against expectations in Step 9 (hit 30.7%, miss 56.8%, FA 1.8%, CR 10.7%) and spot-checked the first 20 trials of a session against the raw NWB in Step 10 Check 2.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to an integer 0–3 via `outcome_names.index(...)` and broadcast to every timepoint of the trial (row 4 of the (5, T) output array), so the static per-trial variable is stored as a constant time series. An outcome that matches none of the four flags falls back to index 0 (i.e. it would be labelled `hit`); this never triggers for Go/Catch non-aborted non-auto-rewarded trials, where exactly one flag is set.

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

iii. The comments in the script show the AI reasoning about mixing static and time-varying outputs in one array ("The format says: (n_output, n_timepoints) or (n_output,) ... Let's make output as (5, T) where last row is constant"), choosing the broadcast form so that a single homogeneous (5, T) array per trial satisfies the target format while keeping the format instruction "If at all possible, make it time-varying". `output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']` records the code order.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i.
- **Missing NWB file / no cells / no stimulus table / <2 valid trials**: WARNING printed, experiment skipped (returns `None`, not appended to the dataset).
- **Missing eye-tracking group** (3/284 files): `pupil_at_ophys` is set to all-NaN, and the session is still kept — every pupil label is bin 0.
- **NaN behavioural samples** (blinks, extrapolation outside the recorded range): excluded from percentile-edge computation, assigned bin 0 in the output.
- **Trials extending past the end of the recording**: handled implicitly by the boolean mask (fewer frames); trials that end up with <2 frames are dropped.
- **Unrecognised image names / `omitted` flashes**: skipped, leaving the `gray` label.
- **Unmatched trial outcome**: silently defaults to index 0 (`hit`).
- There is **no** try/except around per-experiment loading, so a corrupt file would abort the whole run (this did not happen: all 202 experiments converted).

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}");  return None
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0          # NaN -> lowest bin
...
if n_trial_frames < 2:
    continue
...
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```

iii. Step 10 Check 5 ("Check for edge cases") documents the accepted behaviours: few-trial sessions retained if ≥2 trials, mixed frame rates accepted, "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)". The trajectory (step 62) argues that adding a 6th "blink/missing" class would violate the five-bin specification, which is why NaN is folded into the lowest bin rather than flagged.

## 9-a. What are the most time-consuming steps of the code?

i. Reading the NWB files. The script instruments every experiment with `t_load` / `t_process` and prints them: over the 202 experiments the totals were **380.8 s loading vs 138.0 s processing**, out of 544.7 s wall clock (9.1 min), plus 14.4 s for the extra image-name scan and the final 8.5 GB pickle write. Loading is dominated by pulling the entire (140k × n_cells) dF/F array plus the 270k-sample running and 136k-sample eye-tracking arrays into memory.

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

iii. Step 7 estimated the runtime before the full run ("Load NWB ~1.7s/session → ~340s; Process trials ~0.4s → ~80s; Total ~750s (~12.5 min)"), i.e. under the 15-minute budget in the instructions, so no further optimisation was attempted. The AI chose raw h5py over the AllenSDK specifically to cut this I/O cost ("fast, no AllenSDK overhead").

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops account for essentially all of the 138 s of processing time and are all vectorisable:
- `build_image_identity_trace`: for every trial it walks the session's ~4,800-row stimulus presentation table from index 0 (with a `break` once past the trial), and for each flash builds a boolean mask over the trial's timestamps. `np.searchsorted` of the flash on/off times into `ophys_ts` would assign all flashes of the whole session in one vectorised pass.
- `build_image_change_trace`: the same loop but **without** a `break`, so it scans all ~4,800 presentations for each of the ~300 trials (~1.4 M iterations per session) just to place one or two 1s. All change frames could be computed with a single `np.searchsorted(ophys_ts, change_times)`.
- The trial loop recomputes `(ophys_ts >= t_start) & (ophys_ts < t_stop)` — a full scan of the ~140k-element timestamp vector — for every trial, and again inside each of the two trace builders; `np.searchsorted` on the (sorted) timestamps would be O(log n).
- `image_names_list.index(name)` is a linear list search executed once per presentation per trial; a dict lookup would be O(1).

ii.
```python
    for si in range(len(stim_starts)):          # build_image_change_trace: no break
        if not is_change[si]:
            continue
        s_start = stim_starts[si]
        if s_start < trial_start or s_start >= trial_stop:
            continue
        frame_idx = np.searchsorted(trial_ts, s_start)
```
```python
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)   # per flash, per trial
        trace[frame_mask] = img_idx
```
```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)   # recomputed 3x per trial
```

iii. The AI's only recorded efficiency rationale is Step 6 ("Loads NWB files directly via h5py (fast, no AllenSDK overhead)") and Step 7's runtime estimate; the template fields "Code inefficiencies identified" / "Code speedups added" were left unfilled. Implicitly, the justification is that the estimate (12.5 min) and the actual runtime (9.1 min) were within the instructions' 15-minute budget and that I/O, not computation, dominates.

## 9-c. What processing does the code repeat multiple times?

i.
- **A second full pass over every NWB file**: `get_all_image_names()` opens all 202 files purely to read the `image_name` column before the main conversion loop (14.4 s of extra I/O). The image set is already available in the experiment table's `image_set` column, and the names could have been collected during the main pass.
- **The trial frame mask** is computed three times per trial (once in `process_experiment`, once in `build_image_identity_trace`, once in `build_image_change_trace`), each time scanning the full session timestamp array.
- **The stimulus presentation table** is scanned twice per trial (once for identity, once for change).
- `image_names_list.index(GRAY_LABEL)` and `.index(name)` are re-evaluated inside the per-flash loop.
- Interpolation of running/pupil and the percentile edges are correctly computed once per session, not repeated.

ii.
```python
all_image_names = get_all_image_names(exp_table)      # opens every NWB file a second time
...
    img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
    change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)
```
```python
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)   # in both builders
    trial_ts = ophys_ts[trial_mask]
```

iii. Not discussed in CONVERSION_NOTES. The design intent of `get_all_image_names` is documented in the code ("Scan a few NWB files to collect all unique image names across sessions") and in Step 9 — a single global, sorted image-code mapping shared by all sessions requires knowing all names before the conversion loop starts; the AI accepted the extra pass (14.4 s, ~3% of runtime) as the cost of that consistency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of dead work, none of it material:
- Per trial, `output_tv` and `output_static` are built with `np.stack`/`np.array` and then never used — `output_full` is assembled separately from the same components.
- `discretize_percentile()` is defined but never called (superseded by `compute_session_percentile_edges` + `apply_percentile_bins`).
- `cell_roi_ids` is read from the image-segmentation group and never used; the trials-table columns `initial_image_name`, `change_image_name` and `is_change` are read from every file and never used (image identity/change come from the presentations table instead).
- Running speed and pupil are interpolated over the entire session including the inter-trial periods; those samples are not exported, though they are used for the percentile edges, so this is only partly wasted.
- The plotting/diagnostic fields (`ophys_ts`, `running_at_ophys`, `trial_data`, `stim_data`, …) are only attached to the result under `--show-processing`, so nothing is wasted in the full run.

ii.
```python
        # dead: recomputed into output_full below
        output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
        output_static = np.array([outcome_idx], dtype=np.int64)
        ...
        output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
        output_full[0] = img_trace
        ...
```
```python
def discretize_percentile(values, n_bins=5):   # never called
```
```python
for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
             'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
             'initial_image_name', 'change_image_name', 'is_change']:
```

iii. Not discussed in CONVERSION_NOTES; these are leftovers from the exploratory design (the comment block around `output_tv`/`output_static` shows the AI working out how to reconcile time-varying and static outputs in one array and then settling on the broadcast `(5, T)` form without deleting the earlier attempt). Reading extra trials-table columns was a hedge kept from the Step 5 mapping plan, where the trials-table image names were one of the candidate sources for image identity.
