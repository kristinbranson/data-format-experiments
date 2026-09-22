# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the NWB (HDF5) files directly with `h5py`. It builds the working set by (1) reading `project_metadata/ophys_experiment_table.csv`, (2) intersecting it with the set of `behavior_ophys_experiment_<id>.nwb` files actually present on disk (284 files), and (3) dropping every experiment whose `session_type` contains "passive" (OPHYS_2 and OPHYS_5). This leaves 202 experiments (168 `VisualBehavior` + 34 `VisualBehaviorMultiscope`), 38 mice. Each experiment is then loaded in full by `load_nwb_data()`, which pulls ophys timestamps, the dF/F matrix, running speed, eye tracking + blink flags, the trials table, and the `Natural_Images_*_presentations` stimulus table out of fixed HDF5 paths. A separate first pass (`get_all_image_names`) re-opens every NWB file to build the global image-name vocabulary.

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
with h5py.File(nwb_path, 'r') as f:
    data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
    data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
    data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
    data['running_speed']      = f['processing']['running']['speed']['data'][:]
    ...
    trials = f['intervals']['trials']
    ...
    for key in f['intervals'].keys():
        if 'Natural_Images' in key or 'natural_images' in key:
            stim_key = key
```

iii. From CONVERSION_NOTES Step 6: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)". Step 2 justifies the passive exclusion: "Active: OPHYS_1, 3, 4, 6 ... Passive: OPHYS_2, 5 ... We should only use active sessions for the decoder task", restated as Key Decision 10 ("Passive sessions excluded: Only use active behavior sessions"). Step 4 notes only a subset of the release is on disk (38 of 82 mice) and resolves to "use what's available". Step 10 Check 3 argues the h5py read is "Equivalent" to `BehaviorOphysExperiment.from_nwb()`.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiments, taken from the experiment table and stored as strings. Subject indices are assigned lazily in the order mice are first encountered while iterating experiments sorted by `ophys_experiment_id` (not sorted by mouse id, not grouped per mouse). 38 subjects result.

ii.
```python
mouse_id = str(exp_row['mouse_id'])          # in process_experiment()
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

iii. CONVERSION_NOTES Step 5 variable mapping: "`mouse_id` → `subjects`, `subject_idx` | Map unique mice to indices". `mouse_id` is the canonical animal identifier in the Allen metadata tables; no other justification is offered because none is needed.

## 1-c. How are the data split into sessions?

i. One *experiment* (= one imaging plane) becomes one *session* in the output. For the single-plane `VisualBehavior` project this is a 1:1 mapping with `ophys_session_id`. For the 34 `VisualBehaviorMultiscope` experiments that survive the passive filter, the 7–8 simultaneously recorded planes of a single physical session each become their own output "session", so the same behavioural session's trials, running, pupil, images and outcomes are duplicated across up to 8 entries of `data['neural']`. Sessions are ordered by `ophys_experiment_id`, not by acquisition date. Final count: 202 sessions from 174 distinct `ophys_session_id`s.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])      # one list entry per experiment == one session
    all_output.append(result['output'])
    ...
    session_metadata.append({'exp_id': result['exp_id'],
                             'ophys_session_id': result['ophys_session_id'], ...})
```

iii. CONVERSION_NOTES Step 5, Key Decision 9: "**Multiscope handling**: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." Step 4 flags the consequence in advance: "For Multiscope sessions, multiple experiments (planes) per session share the same trials/behavior" and "Need to handle both ~31 Hz and ~11 Hz ophys frame rates." The duplication of behaviour across plane-sessions is never revisited; Step 10 Check 5 only records that "Multiscope sessions have ~11 Hz frame rate (vs ~31 Hz Scientifica) - different T per trial".

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if it is a Go **or** Catch trial and is neither Aborted nor Auto-rewarded. The trial window is the full `[start_time, stop_time)` interval, sliced on the ophys timebase, giving variable-length trials (~224–389 frames at 31 Hz, ~77–135 frames at 11 Hz, i.e. ~8 s). Trials yielding fewer than 2 ophys frames are dropped. 51,992 trials result.

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
    t_stop  = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "**Trial definition**: Use `start_time` and `stop_time` from trials table for Go and Catch trials only." Step 3 "Trial curation rules (for this decoder task): Include: Go trials and Catch trials; Exclude: Aborted trials and Auto-rewarded trials" — a direct transcription of the task instructions. Using the whole `start_time → stop_time` window (rather than a fixed window around the change) is what makes the time-varying image/change/running/pupil outputs meaningful within a trial.

## 1-e. How are trials filtered based on quality controls?

i. Quality control is entirely at the trial-type level plus two degenerate-case guards: (a) Aborted and Auto-rewarded trials excluded, only Go|Catch kept; (b) a trial with <2 ophys frames in its window is skipped; (c) an experiment producing <2 usable trials (checked both before and after the trial loop) is dropped entirely; (d) experiments with no cells, no stimulus table, or a missing NWB file are dropped. There is no `change_time` validity check, no engagement/reward-rate filter, and no per-trial behavioural QC.

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

iii. Step 3 of CONVERSION_NOTES records that session-level QC (saturated pixels, photobleaching, z-drift, peak d-prime ≥ 1.0, sync) is "already applied to the data in the table", so no further session QC was applied. The ≥2-trial rule is implicitly driven by the task spec ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). Step 10 Check 5 notes: "Sessions with very few valid trials (e.g. 39) are retained if >=2 trials." Piet et al.'s engagement threshold (>2 rewards/min) is recorded in Step 3 but deliberately not applied, since the decoder task did not ask for it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. dF/F calcium traces, read from the NWB path `processing/ophys/dff/traces/data`, with the matching `timestamps` used as the session timebase. Deconvolved events (`processing/ophys/event_detection`) were identified but not used.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "**Neural data**: Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." Step 1 notes that dF/F is already computed in the NWB files (600 s median-filter baseline, 3.33 s median-filter detrending), so nothing needs to be recomputed; Step 10 Check 3 lists dF/F as matching the SDK reference.

## 2-b. How is the `neural` data processed?

i. Essentially none. The (n_frames, n_cells) array is transposed to (n_cells, n_frames), cast to `float32` when sliced per trial, and sliced by the trial frame mask. No normalisation, smoothing, z-scoring, or cross-plane merging is done (planes stay in separate output sessions). Each neuron's brain region is the experiment's `targeted_structure` (VISp / VISl only, without imaging depth).

ii.
```python
data['dff_traces'] = dff_raw.T                       # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)       # (n_cells, n_trial_frames)
...
all_brain_region_idx.append(
    np.full(result['n_cells'], region_map[brain_region], dtype=np.int64))
```

iii. Step 1: "dF/F is pre-computed in NWB files (no need to compute from scratch)"; Step 10 Check 3 compares against the SDK and marks dF/F computation as "Pre-computed (600s median filter + detrending) — Yes". `float32` is used per the instructions' efficiency guidance.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. The AI checked that the SDK's `exclude_invalid_rois=True` behaviour is already baked into the released NWB files and therefore did not re-implement it. (Independently verified here: `dff` has exactly 89/142 traces for experiments 775614751/788490510, matching `ophys_cells_table.csv` row counts; the full-run total of 29,444 neurons is exactly the cells-table total for the AI's 202 active experiments.) Experiments with zero cells are dropped.

ii.
```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    print(f"  WARNING: No cells in experiment {exp_id}")
    return None
```
(there is no `valid_roi` mask anywhere in the script)

iii. CONVERSION_NOTES Step 10 Check 3: "ROI filtering | No explicit valid_roi filter | exclude_invalid_rois=True (default) | OK - all ROIs in downloaded NWB files are valid". Step 10 Check 4 cross-checks the neuron total against `ophys_cells_table.csv` and marks it PASS. Step 1 had identified `exclude_invalid_rois` as the SDK's curation function.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the ophys timebase, trial by trial: every ophys frame whose timestamp falls in `[trial.start_time, trial.stop_time)` is taken, and the dF/F columns for exactly those frames form the trial matrix. The same boolean mask is reused for running and pupil, and the same timestamps drive image identity / image change, so all streams are frame-locked by construction. `metadata['temporal_alignment_event']` is set to "Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time."; `off_start`/`off_end` are `None` because the window is trial-relative, not event-relative.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
pupil_trial   = pupil_at_ophys[frame_mask]
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "**Temporal alignment**: Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time." This follows the task instruction "Temporally align based on ophys timestamp." Step 3 notes all streams are hardware-synced ("NI PCI-6612 at 100 kHz"), which is why raw timestamps can be compared directly. Step 10 Check 2 verifies a spot-checked trial's dF/F against the raw NWB with `np.allclose` (max_diff = 0.0).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling at all — each session keeps its native ophys frame period. That is 32.3 ms (~30.9 Hz) for the 168 Scientifica sessions and **93.2 ms (~10.7 Hz) for the 34 Multiscope sessions** (confirmed by `grep "dt:" conversion_full_out.txt` → 168 × 32.3 ms, 34 × 93.2 ms). `metadata['time_bin_size']` is the *median* across sessions, 32.32 ms, and `README.md` reports a single "Time bin | ~32.3 ms (~31 Hz)". Per-session `dt` values are, however, preserved in `metadata['session_info'][i]['dt_ms']`.

ii.
```python
dt = np.median(np.diff(ophys_ts))   # per-experiment, inside process_experiment
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
...
'metadata': {..., 'time_bin_size': median_dt,
             'ophys_frame_rate_hz': 1000.0 / median_dt, ...}
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "**Time bin**: Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." Step 4 lists "Need to handle both ~31 Hz and ~11 Hz ophys frame rates" as a resolution note, and Step 6 advertises "Handles both Scientifica (~31 Hz) and Multiscope (~11 Hz) sessions" as a feature. No justification is given for reporting a single median bin size, and the conflict with the format requirement "Time bins should be the same size for all trials and sessions" is never discussed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the stimulus presentations table `intervals/Natural_Images_*_presentations`, specifically the `image_name`, `start_time` and `stop_time` columns (the `omitted` flashes are skipped). It is *not* derived from the trials table's `initial_image_name` / `change_image_name`. An extra category `'gray'` is prepended to the vocabulary and used for every ophys frame that is not inside an image flash (the 500 ms inter-stimulus grey and omitted flashes), giving 17 classes = gray + 16 images.

ii.
```python
GRAY_LABEL = 'gray'
...
image_names_list = [GRAY_LABEL] + all_image_names
...
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    stim_starts = stim_data['start_time']; stim_stops = stim_data['stop_time']
    stim_names  = stim_data['image_name']
```

iii. CONVERSION_NOTES Step 5, Key Decision 5: "**Image identity**: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category." Step 9/10 treat the resulting 66.9 % gray fraction as a *consistency check* against the 500/750 ms duty cycle ("Gray screen fraction | 66.9% | 66.7% (500/750ms) | PASS"), i.e. the gray class is presented as evidence that the stimulus timing was reproduced correctly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. (1) A global vocabulary is built by re-opening every NWB file and collecting all non-`omitted` `image_name` values (16 images across image sets A and B), sorted alphabetically, with `'gray'` prepended at index 0. (2) For each trial, a per-frame integer trace is initialised to the gray code and then, for every stimulus presentation overlapping the trial, the frames inside `[stim.start_time, stim.stop_time)` are set to that image's code. Byte strings are decoded; `omitted` presentations and names not in the vocabulary are skipped (left gray).

ii.
```python
def get_all_image_names(exp_table):
    all_names = set()
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            for key in f['intervals'].keys():
                if 'Natural_Images' in key or 'natural_images' in key:
                    names = f['intervals'][key]['image_name'][:]
                    for n in names:
                        if isinstance(n, bytes): n = n.decode('utf-8')
                        if n != 'omitted': all_names.add(n)
    return sorted(all_names)
```
```python
    for si in range(len(stim_starts)):
        s_start = stim_starts[si]; s_stop = stim_stops[si]
        if s_stop < trial_start: continue
        if s_start >= trial_stop: break
        name = stim_names[si]
        if isinstance(name, bytes): name = name.decode('utf-8')
        if name == 'omitted': continue
        if name in image_names_list: img_idx = image_names_list.index(name)
        else: continue
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
```

iii. A global, sorted vocabulary keeps codes consistent across sessions that use different image sets (A vs B); CONVERSION_NOTES Step 9 checks "Images/session | 8 | 8 per session (16 total across sets) | Yes". Omitted flashes are treated as grey because they physically *are* grey screen (Step 3 records the 5 % omission rate).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The trace is computed on exactly the same ophys frames as the neural matrix: `build_image_identity_trace` recomputes the identical `(ophys_ts >= t_start) & (ophys_ts < t_stop)` mask and assigns image codes by comparing each frame's timestamp to the stimulus `start_time`/`stop_time`. Length therefore always equals the neural `n_timepoints`, and it is written to row 0 of the (5, T) output array.

ii.
```python
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)   # inside the flash
trace[frame_mask] = img_idx
...
output_full[0] = img_trace
```

iii. Step 5 planned sanity check "Verify no temporal misalignment by plotting neural + stimulus for sample trials"; Step 7 reports "Image identity correctly shows gray during ISI, images during flashes"; Step 10 Check 2 reports a raw-NWB spot check of the image identity trace (`np.array_equal=True`, "verified at stimulus onset").

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` and `start_time` columns of the stimulus presentations table. Because the Allen `is_change` flag is only True when the image actually changed, catch (sham-change) trials automatically receive an all-zero trace — the trials table's `go`/`catch` columns are not needed for this.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`is_change` from stimulus presentations → `output[1]`: image_change | Binary, 1 at change onset frame, 0 otherwise | Time-varying." Using the stimulus-table flag rather than `change_time` makes the indicator refer to the actual displayed change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector of length `n_frames` is created and a single 1 is written at the first ophys frame at or after the change flash's onset (`np.searchsorted` into the trial's timestamps). No window, no smoothing, no per-trial repetition. Verified empirically on experiment 775614751: exactly one `1` in each of the 34 go trials and zero in each of the 5 catch trials. Across the full dataset this yields 0.4 % positive frames.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
for si in range(len(stim_starts)):
    if not is_change[si]: continue
    s_start = stim_starts[si]
    if s_start < trial_start or s_start >= trial_stop: continue
    frame_idx = np.searchsorted(trial_ts, s_start)
    if frame_idx < n_frames:
        trace[frame_idx] = 1
```

iii. This is a literal reading of the decoder-task spec: "Image change, binary variable. Have value of 1 right after a change in image identity, otherwise 0." CONVERSION_NOTES Step 10 Check 4 validates the resulting density ("Image change fraction | 0.372% | ~0.3-0.4% | PASS"), and Step 12 explicitly accepts the imbalance: "Extreme class imbalance (99.6% no_change vs 0.4% change). Balanced accuracy penalizes heavily... Not a conversion bug."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required or done — the variable is already binary (0 = `no_change`, 1 = `change`), with `output_values[1] = ['no_change', 'change']`.

ii.
```python
output_values = [
    image_names_list,
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    ...
]
```

iii. The task spec defines the variable as binary, so the only "categorisation" choice was how wide the positive window should be (see 4-b: one frame).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the same `(ophys_ts >= t_start) & (ophys_ts < t_stop)` mask is recomputed inside `build_image_change_trace`, and the positive sample is placed with `np.searchsorted(trial_ts, s_start)`, i.e. the first imaging frame at or after the change onset. It becomes row 1 of the (5, T) output array, the same length as the neural matrix.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_idx = np.searchsorted(trial_ts, s_start)
trace[frame_idx] = 1
...
output_full[1] = change_trace
```

iii. Step 5 planned sanity check: "Check that image_change=1 aligns with actual change in image_identity"; Step 7 reports "Change events aligned to stimulus onset" from the `--show-processing` plots.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. The NWB `processing/running/speed` TimeSeries (`data` + `timestamps`), i.e. the SDK's already-filtered running speed at 60 Hz, in cm/s.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed']      = f['processing']['running']['speed']['data'][:]
```

iii. Step 2 identifies "Running: `processing/running/speed` (270K samples @ 60 Hz)"; Step 3 records that the SDK applies a "10 Hz lowpass Butterworth filter" upstream, so no additional filtering is needed. Step 10 Check 3 marks running speed as matching the SDK reference.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the 60 Hz speed onto the full session's ophys timestamps (`bounds_error=False, fill_value=np.nan`), then percentile discretisation. No additional filtering, no absolute value, no clipping.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)
...
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
```

iii. Step 5, Key Decision 7: "**Running speed**: Interpolate from 60 Hz to ophys timestamps using linear interpolation." Justified by hardware synchronisation of the clocks (Step 3: "All streams synced via NI PCI-6612 at 100 kHz"). Step 5 also planned the sanity check "Verify running speed range is reasonable (typically 0-80 cm/s)".

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (0–20th, …, 80–100th), with the bin edges computed **per session** from all non-NaN interpolated samples of that session's *entire* recording (not only the retained trial windows), and then applied to the trial slices. Outer edges are replaced by ±inf so nothing falls outside; NaN maps to bin 0. The resulting global distribution is [0.190, 0.199, 0.204, 0.208, 0.200].

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf; edges[-1] = np.inf
    return edges

def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
...
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. Step 5, Key Decision 8: "**Percentile bins for running/pupil**: Compute percentiles across the entire session (all valid timepoints), then apply per-trial. Use 5 equal bins (0-20th, 20-40th, ..., 80-100th percentile)." This follows the task spec "discretized into five equal percentile bins"; the session scope is chosen so that each session contributes a balanced set of labels.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolation is done once onto the full session's ophys timestamps, so the running vector is index-parallel to the dF/F matrix; the trial slice uses the very same boolean `frame_mask` as the neural slice. It occupies row 2 of the (5, T) output array.

ii.
```python
running_at_ophys = interpolate_to_ophys(nwb_data['running_speed'],
                                        nwb_data['running_timestamps'], ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
running_trial = running_at_ophys[frame_mask]
...
output_full[2] = running_binned
```

iii. Alignment is guaranteed by construction (shared `frame_mask`). Step 10 Check 2 records a raw-NWB spot check of running-speed bins for session 803736273 (`np.array_equal=True`), and the `--show-processing` plots overlay raw cm/s on the binned trace per trial.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking` — the ellipse **area** (plus its 30 Hz timestamps) — and `acquisition/EyeTracking/likely_blink`. Diameter is derived from area, not read from `pupil_width`/`pupil_height`. If an experiment has no `EyeTracking` group, pupil is filled with NaN for the whole session.

ii.
```python
if 'EyeTracking' in f.get('acquisition', {}):
    et = f['acquisition']['EyeTracking']
    if 'pupil_tracking' in et:
        pt = et['pupil_tracking']
        data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
        data['pupil_timestamps'] = pt['timestamps'][:]
        data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. Step 2 identifies "Eye tracking: `acquisition/EyeTracking/pupil_tracking` (area, height, width @ 30 Hz)" and "Blinks: `acquisition/EyeTracking/likely_blink`". Step 5, Key Decision 6: "**Pupil diameter**: Compute from pupil area as `2*sqrt(area/pi)`." Step 10 Check 3 claims the formula matches the whitepaper.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Frames flagged `likely_blink` are set to NaN; (2) equivalent circular diameter `d = 2·sqrt(area/π)` is computed for the remaining positive-area frames, everything else staying NaN; (3) the NaN-containing series is passed straight into `interp1d` and evaluated at the ophys timestamps. Because the NaNs are left *inside* the interpolant rather than dropped, linear interpolation propagates them: measured on experiment 1007107386, 9.19 % of eye-tracking frames are blinks and **9.54 % of ophys frames end up NaN**, versus 0.22 % if the blink rows are dropped before interpolation (as the human reference does). All of those NaN frames are then assigned pupil bin 0.

ii.
```python
pupil_area = nwb_data['pupil_area'].copy()
likely_blink = nwb_data['likely_blink']
pupil_ts = nwb_data['pupil_timestamps']
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. Step 5, Key Decision 6: "Set blink frames to NaN, then interpolate. Discretize non-NaN values." Step 3 notes the upstream pipeline is "DeepLabCut tracking, ellipse fit, blinks detected and set to NaN". The trajectory (step 60/62) shows the AI explicitly raised the issue — "mapping NaN to bin 0 could confuse the decoder" — and then dismissed it: "since pupil_diameter is specified as 'discretized into five equal percentile bins', mapping NaN to bin 0 is a reasonable choice (lowest bin). A separate blink category would add a 6th class which isn't specified. I'll leave it as is." Interpolating *across* blinks was never considered.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same machinery as running speed: five equal-percentile bins with per-session edges computed from the non-NaN interpolated pupil trace over the whole session, applied to the trial slices, with NaN (i.e. blinks, ~9.5 % of frames, plus whole sessions lacking eye tracking) forced into bin 0. The realised distribution is [0.200, 0.199, 0.214, 0.221, 0.166] — the only output whose bins deviate noticeably from 20 % each.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
...
pupil_trial = pupil_at_ophys[frame_mask]
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
...
output_full[3] = pupil_binned
```

iii. Step 5, Key Decision 8 (shared with running speed). Step 10 Check 5 lists it as a conscious edge-case decision: "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)"; Step 12 acknowledges the cost: "pupil_diameter (1.41x): Slightly unbalanced due to blink frames mapped to bin 0."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated once onto the full session ophys timebase, then sliced with the identical `frame_mask` used for the neural data, so it is frame-locked by construction and occupies row 3 of the (5, T) array.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same hardware-sync justification as running speed (Step 3). Step 10 Check 2 records a raw-NWB spot check of pupil bins for session 775614751 (`np.array_equal=True`, "blinks=NaN confirmed").

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that fixed priority order for the trial's row index.

ii.
```python
def get_trial_outcome(trial_data, idx):
    if trial_data['hit'][idx]:            return 'hit'
    elif trial_data['miss'][idx]:         return 'miss'
    elif trial_data['false_alarm'][idx]:  return 'false_alarm'
    elif trial_data['correct_reject'][idx]: return 'correct_reject'
    else:                                 return 'unknown'
```

iii. Step 5 variable mapping: "Trial outcome (hit/miss/FA/CR) → `output[4]`: trial_outcome | Categorical, static per trial | 4 categories." Step 9 validates the resulting rates against expectation (hit 30.7 %, miss 56.8 %, FA 1.8 %, CR 10.7 %); Step 10 Check 2 spot-checks the first 20 trials of session 803736273 against the raw NWB flags.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to an index into `['hit', 'miss', 'false_alarm', 'correct_reject']` and then **broadcast** across all time bins of the trial so it can live in the same (5, T) array as the time-varying outputs (the format spec permits static per-trial variables, but a single array per trial is required). An outcome that matches none of the four flags falls back to index 0, i.e. it is silently labelled `hit`.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. CONVERSION_NOTES does not discuss the broadcast explicitly beyond marking the variable "static per trial"; the code comments show the reasoning ("The format says: (n_output, n_timepoints) or (n_output,) ... Let's make output as (5, T) where last row is constant"). The `else 0` fallback is undocumented. (Checked on experiment 775614751: 0 of 39 valid trials are `unknown`, so the fallback does not fire there.)

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases: missing NWB file → warn and skip the experiment; zero cells → skip; no `Natural_Images_*` stimulus table found → skip; <2 valid trials before or after processing → skip the whole experiment; a trial window containing <2 ophys frames → skip that trial; behavioural samples outside the interpolation range, and all blink frames, → NaN → discretised to bin 0; an experiment with no `EyeTracking` group → pupil all-NaN → the whole session's pupil output is bin 0; `image_name` values not in the global vocabulary → left as gray. Not handled: there is **no** try/except around `process_experiment`, so a single corrupt file would abort the whole run (the human reference wraps each session); and an unrecognised trial outcome is silently recoded as `hit` rather than flagged.

ii.
```python
if not os.path.exists(nwb_path):
    print(f"  WARNING: NWB file not found for experiment {exp_id}"); return None
if n_cells == 0: ... return None
if stim_data is None: ... return None
if len(valid_trial_idx) < 2: ... return None
if n_trial_frames < 2: continue
if len(neural_trials) < 2: ... return None
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0          # NaN -> bin 0
...
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```
Only `get_all_image_names` is defensive:
```python
        except Exception as e:
            print(f"  WARNING: Could not read images from {eid}: {e}")
```

iii. Step 10 Check 5 ("Check for edge cases") lists the retained low-trial sessions, the dual frame rates, and the NaN→bin-0 rule, concluding "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)" and "No critical issues found. All checks pass." The full-run log contains no WARNING lines, i.e. none of the skip paths fired on this dataset.

## 9-a. What are the most time-consuming steps of the code?

i. The AI instrumented load vs process time per experiment and printed both. From `conversion_full_out.txt`: total 544.7 s for 202 experiments (~2.7 s each), of which reading the NWB file (`load_nwb_data` — the full (n_frames × n_cells) dF/F matrix plus its transpose-copy, 270 k running samples, 136 k eye samples, trials and ~4,800 stimulus presentations) is ~1.3–1.8 s, and the per-trial processing loop ~0.1–0.8 s. The `get_all_image_names` pre-pass over every NWB file adds 14.4 s, and pickling the 8.5 GB output adds the remainder. So the bottleneck is disk I/O of the dF/F traces, ~60 % of wall-clock.

ii.
```python
t0 = time.time()
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
t_process = time.time() - t0 - t_load
...
print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, dt: ..., "
      f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")
```

iii. CONVERSION_NOTES Step 7 gives the extrapolation that justified running the full conversion: "Load NWB ~1.7s/session → ~340s; Process trials ~0.4s → ~80s; Total ~3.7s → ~750s (~12.5 min)", under the instructions' 15-minute budget (actual 9.1 min).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three clear candidates, none of which the AI documented:
- `build_image_identity_trace` and `build_image_change_trace` loop over **all ~4,800 stimulus presentations for every trial** (only the trailing side is short-circuited with `break`; earlier presentations are walked with `continue`), so the cost is O(n_trials × n_stim) ≈ 1.2 M iterations per session. A pair of `np.searchsorted` calls on the presentation start/stop times would reduce this to O(n_trials log n_stim). Inside that loop, `image_names_list.index(name)` is a linear list scan per presentation, where a dict would be O(1).
- The trial window is obtained by a boolean comparison over the **entire** session timestamp vector (~140 k elements) rather than by `searchsorted` on the sorted timestamps, and this mask is rebuilt three times per trial (once in `process_experiment`, again in each of the two trace builders), i.e. ~3 × 250 × 140 k ≈ 10⁸ comparisons per session.
- The per-trial outer loop itself (slicing dF/F, binning running/pupil) could be done with index arrays, though this is the minor term.

ii.
```python
    for si in range(len(stim_starts)):          # over ~4800 presentations, per trial
        s_start = stim_starts[si]; s_stop = stim_stops[si]
        if s_stop < trial_start: continue
        if s_start >= trial_stop: break
        ...
        img_idx = image_names_list.index(name)  # linear scan of the vocabulary
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
```
```python
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)   # full-length mask, per trial
    ...
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)   # recomputed in each builder
```

iii. The AI never identified these; CONVERSION_NOTES Step 6 omits the template's "Code inefficiencies identified" / "Code speedups added" entries entirely, and Step 7 simply concludes the estimated ~12.5 min is acceptable. The implicit justification is that loading dominates: processing is only ~100 s of the 545 s total, so vectorising the loops would have saved <20 % of wall-clock.

## 9-c. What processing does the code repeat multiple times?

i. Three repetitions:
- **Every NWB file is opened and read twice** — once by `get_all_image_names` purely to harvest `image_name`, and again by `load_nwb_data` during conversion (14.4 s of avoidable I/O; the vocabulary could have been accumulated during the single conversion pass, at the cost of a second assembly pass).
- **The trial frame mask is computed three times per trial** (`process_experiment`, `build_image_identity_trace`, `build_image_change_trace`), each over the full session timestamp vector; `trial_ts` is likewise re-derived twice.
- **The full stimulus presentation table is re-scanned from index 0 for every trial**, in both trace builders — the same ~4,800 rows traversed ~500 times per session.

ii.
```python
all_image_names = get_all_image_names(exp_table)      # pass 1: opens all 202 NWB files
...
    nwb_data = load_nwb_data(nwb_path)                # pass 2: opens each file again
```
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)             # in process_experiment
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)     # in build_image_identity_trace
...
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)     # in build_image_change_trace
```

iii. Not documented anywhere in CONVERSION_NOTES. The design intent is readability/modularity — the two trace builders are written as self-contained functions taking `(ophys_ts, stim_data, trial_start, trial_stop)`, which forces them to recompute the mask; and the vocabulary pre-pass exists so that image codes are globally consistent before any trial is encoded.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- **Dead per-trial computation**: `output_tv` (a `np.stack` of the four time-varying rows) and `output_static` are built for every one of the 51,992 trials and then never used — `output_full` is assembled independently two lines later. The surrounding "thinking out loud" comments were left in the file.
- **Unused loaded data**: `cell_roi_ids` is read from the image-segmentation group and never referenced; the trials table columns `change_time`, `initial_image_name`, `change_image_name` and `is_change` are read into `trial_data` and never used (image identity/change come from the stimulus table instead).
- **Unused function**: `discretize_percentile()` is defined but never called (superseded by `compute_session_percentile_edges` + `apply_percentile_bins`).
- **Whole-session interpolation**: running and pupil are interpolated onto all ~140 k ophys frames although only the ~30 % of frames inside retained trials are exported — though this is needed to compute the session-wide percentile edges, so it is only partly wasted.
- **Empty input arrays**: a `(0, T)` float32 array is allocated per trial for `input` even though `input_names` is empty.

ii.
```python
        # Stack outputs
        output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
        output_static = np.array([outcome_idx], dtype=np.int64)
        # Combine: (5, n_trial_frames) for time-varying, last row repeated
        # Actually, static outputs should be (n_output,) shape
        ...
        output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
        output_full[0] = img_trace
```
```python
        if 'image_segmentation' in f['processing']['ophys']:
            ...
                    data['cell_roi_ids'] = seg[key]['id'][:]       # never used
        for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
                     'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
                     'initial_image_name', 'change_image_name', 'is_change']:   # last 4 unused
```
```python
all_input.append([np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in result['neural']])
```

iii. No justification is given — none of this is mentioned in CONVERSION_NOTES, whose Step 6 section lists only features. The residual code and comments indicate the dead `output_tv`/`output_static` block is a leftover from working out how to represent a static output inside a per-trial array; the extra trials-table columns were loaded speculatively before the AI settled on deriving image identity/change from the stimulus presentations table instead.
