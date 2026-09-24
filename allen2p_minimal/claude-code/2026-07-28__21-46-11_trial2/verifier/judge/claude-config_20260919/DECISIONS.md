# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the released NWB (HDF5) files directly with `h5py`, using the CSV manifest `project_metadata/ophys_experiment_table.csv` as the listing of experiments. The table is (1) restricted to experiments whose NWB file is actually present on disk (284 files), and (2) filtered to `passive == False`, leaving 202 active-behavior experiments. **No filter on `project_code` was applied**, so both `VisualBehavior` (single-plane Scientifica, ~31 Hz, 168 experiments) and `VisualBehaviorMultiscope` (multi-plane Mesoscope, ~11 Hz, 34 experiments from one mouse, 457841) are included. Every per-experiment field (dF/F, event detection, cell specimen table, stimulus presentations, trials, running speed, eye tracking, imaging rate) is pulled from fixed HDF5 paths in `load_nwb_data()`. The files are read three separate times: once to collect image names, once to accumulate running/pupil statistics, and once to actually segment trials.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
    # Filter for active behavior only
    exp_table = exp_table[exp_table['passive'] == False]
```
```python
with h5py.File(nwb_path, 'r') as f:
    ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
    events = f['processing/ophys/event_detection/data'][:]
    dff = f['processing/ophys/dff/traces/data'][:]
    cst = f['processing/ophys/image_segmentation/cell_specimen_table']
    ...
    trials = f['intervals/trials']
    data['running_speed'] = f['processing/running/speed/data'][:]
    ...
```
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
```

iii. From the trajectory (step 31) and `CONVERSION_NOTES.md`: "Each NWB file = one ophys experiment (one imaging plane) ... Filter for active behavior sessions only (passive=False)". Passive sessions were dropped because the lick spout is retracted, so the behavioral outputs (trial outcome) are degenerate. The AI deliberately kept all project codes: "**All project codes included**: VisualBehavior (single-plane, ~31 Hz) and VisualBehaviorMultiscope (multi-plane, ~11 Hz)", on the grounds that the paper's neural analysis used the multiplane rig and that "I should include them since they are part of the dataset" (step 34). Two background sub-agents were used to cross-check that the raw HDF5 paths correspond to the SDK accessors (`experiment.events`, `experiment.ophys_timestamps`, `experiment.running_speed`, `experiment.eye_tracking.pupil_width`, `experiment.trials`); the AI reported that they matched (steps 90, 92).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the filtered experiment table, cast to `str` and sorted. Each output session gets a `subject_idx` pointing into that list. This yields 38 mice (37 single-plane + the one Multiscope mouse).

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_list.append(subject_to_idx[mouse_id])
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx_list, dtype=int),
```

iii. Not explicitly argued in the trajectory beyond the exploration of the experiment table (steps 24-25), where the AI confirmed `mouse_id` is the animal identifier and counted "Unique mice: 38". The summary reports "202 sessions from 38 mice".

## 1-c. How are the data split into sessions?

i. **Each ophys *experiment* (i.e. each imaging plane / each NWB file) is emitted as one "session"** in the output. The AI does not group experiments by `ophys_session_id`. For the 168 single-plane `VisualBehavior` experiments this coincides with one session per recording; for the 34 Multiscope experiments it does not: those 34 planes come from only 6 actual recording sessions of mouse 457841, so the same behavioral session (same trials, same running/pupil/stimulus streams) is emitted up to 7 times with different neurons each time. Sessions are ordered by the row order of the experiment table (not sorted by acquisition date).

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    neural_all.append(final_neural)
    input_all.append(final_input)
    output_all.append(final_output_trials)
    subject_idx_list.append(subject_to_idx[mouse_id])
    region_idx = region_to_idx[structure]
    brain_region_idx_all.append(np.full(n_neurons, region_idx, dtype=int))
```
From `CONVERSION_NOTES.md`: "Each experiment = 1 session in the output".

iii. Step 31: "Each NWB file = one ophys experiment (one imaging plane) ... Each experiment is a 'session' for our purposes." The AI noted in step 34 that "For the Multiscope data, each experiment corresponds to one imaging plane" and that those planes have few cells (4-23) and a different frame rate, but decided to keep them as separate sessions rather than merge planes: "I should include them since they are part of the dataset." The whitepaper text the AI read (`methods.txt`) states explicitly that "For multi-plane imaging experiments, there can be up to 8 imaging planes (8 experiments) per session"; the AI did not act on this distinction.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if it is not `aborted`, not `auto_rewarded`, is `go` or `catch`, and has a resolvable outcome (one of hit/miss/false_alarm/correct_reject). The trial window is the half-open interval `[start_time, stop_time)` selected on the ophys timestamps, giving variable-length trials (77-389 frames in the final dataset). Trials with fewer than 2 ophys frames in the window are dropped.

ii.
```python
def get_trial_mask(trial_idx, nwb_data):
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True
```
```python
for trial_idx in range(n_trials_total):
    if not get_trial_mask(trial_idx, nwb_data):
        continue
    outcome = get_trial_outcome(trial_idx, nwb_data)
    if outcome == -1:
        continue
    t_start = nwb_data['trial_start_times'][trial_idx]
    t_stop = nwb_data['trial_stop_times'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    frame_indices = np.where(frame_mask)[0]
    if len(frame_indices) < 2:
        continue
```

iii. `CONVERSION_NOTES.md`: "Trials segmented using `start_time` and `stop_time` from NWB trials table ... Ophys frames within [start_time, stop_time) extracted for each trial ... Trials with < 2 ophys frames are excluded." Step 31 explains the rationale for the full window: "Each trial spans from start_time to stop_time and contains multiple image presentations, with the critical change occurring at change_time—either a real image change for Go trials or a sham change for Catch trials", and "since trials vary in duration from about 7 to 12 seconds, I'll need to handle variable-length sequences".

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order: (a) aborted trials excluded; (b) auto-rewarded trials excluded; (c) trials that are neither go nor catch excluded; (d) trials whose outcome is not one of the four canonical labels excluded; (e) trials with < 2 ophys frames excluded; (f) experiments yielding < 2 trials or 0 neurons are skipped entirely (0 experiments were actually skipped in the full run). No filtering is done on all-zero neural activity — the verification log reports 2,620 trials (~5% of 51,992) in which every neuron has zero detected events for the whole trial.

ii.
```python
    if not get_trial_mask(trial_idx, nwb_data):   # aborted / auto-rewarded / not go|catch
        continue
    outcome = get_trial_outcome(trial_idx, nwb_data)
    if outcome == -1:
        continue
    ...
    if len(frame_indices) < 2:
        continue
```
```python
if neural_trials is None or len(neural_trials) < 2:
    print(f"    SKIPPED: insufficient valid trials ...")
    skipped_experiments.append(eid)
    continue
n_neurons = neural_trials[0].shape[0]
if n_neurons < 1:
    print(f"    SKIPPED: no valid neurons")
```

iii. `CONVERSION_NOTES.md`: "**Included**: Go trials and Catch trials. **Excluded**: Aborted trials (mouse licked before change) and Auto-rewarded trials (free rewards). This matches the task specification and standard analysis practice from the reference papers." The ≥2-trial rule follows the instruction that "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are the **detected calcium events**, read from `processing/ophys/event_detection/data` (shape `(n_timepoints, n_cells)`), restricted to columns flagged `valid_roi` in the cell specimen table and transposed to `(n_neurons, n_timepoints)`. dF/F is loaded but never used for the output. Timestamps come from `processing/ophys/dff/traces/timestamps` (identical length/values to the event-detection timestamps).

ii.
```python
events = f['processing/ophys/event_detection/data'][:]
...
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
```

iii. Step 31: "The paper specifies using discrete calcium events rather than raw fluorescence, since they regressed events from the traces to remove the slow decay dynamics of GCaMP6f, so I should work with the events data." `CONVERSION_NOTES.md` quotes the paper directly: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces". In step 92 the AI also checked the SDK distinction between `events` and `filtered_events` and concluded "I'm using raw events which is appropriate".

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the SDK/pipeline-provided event traces are transposed, cast to `float32`, and sliced per trial. There is no normalization, smoothing, z-scoring, or spike-count rebinning, and no concatenation of neurons across imaging planes (each plane is its own output session, see 1-c). Brain region is recorded per session from `targeted_structure` (`VISp`/`VISl`), with the same index assigned to every neuron in the session; imaging depth is not encoded in the region label.

ii.
```python
neural_trials.append(trial_neural.astype(np.float32))
...
all_brain_regions = sorted(exp_table['targeted_structure'].unique())
region_to_idx = {r: i for i, r in enumerate(all_brain_regions)}
...
region_idx = region_to_idx[structure]
brain_region_idx_all.append(np.full(n_neurons, region_idx, dtype=int))
```

iii. The AI treated the Allen pipeline output as final: the event traces already reflect motion correction, demixing, neuropil subtraction, dF/F and event extraction as described in `methods.txt`. `CONVERSION_NOTES.md` lists only "Used detected calcium events ... Only valid ROIs included" under "Neural Data", with no further processing steps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is the `valid_roi` boolean from the NWB cell specimen table; in the released NWB files every ROI is already flagged valid, so this filter is a no-op (verified: `valid_roi.sum() == len(valid_roi)` for the checked file). Sessions with zero neurons would be dropped, but none were. No filtering on SNR, event rate, neuron count, or all-zero trials; sessions with as few as 4 neurons are kept.

ii.
```python
cst = f['processing/ophys/image_segmentation/cell_specimen_table']
data['valid_roi'] = cst['valid_roi'][:]
...
valid = nwb_data['valid_roi']
events = events[:, valid]
...
if n_neurons == 0:
    return None, None, None, None, None, None
```

iii. Step 92: "The SDK also mentions `exclude_invalid_rois=True` as default, which my script handles by filtering on `valid_roi`." The AI relied on the Allen QC pipeline (`methods.txt` "ROI FILTERING" section) for everything else. It noted the consequence in "Known Limitations": "VISl experiments (from Multiscope) have very few neurons per session (4-23)" and "Some trials have all-zero neural data (sparse calcium events)", but chose not to filter them out.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the ophys frame clock: a boolean mask over `ophys_timestamps` selects the frames in `[trial start_time, trial stop_time)`, and the identical frame indices are used to slice the neural matrix and every output stream. Trials therefore start at the trial start (not at `change_time`), and are variable-length. `metadata['temporal_alignment_event']` is set to `'Ophys imaging frame timestamps'`, with `off_start`/`off_end` set to `None`.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
...
trial_neural = events[frame_indices, :].T
trial_image  = image_at_ophys[frame_indices]
trial_change = is_change_at_ophys[frame_indices]
trial_running = running_at_ophys[frame_indices]
trial_pupil  = pupil_at_ophys[frame_indices]
```
```python
'temporal_alignment_event': 'Ophys imaging frame timestamps',
'off_start': None,
'off_end': None,
```

iii. The instruction said "Temporally align based on ophys timestamp". `CONVERSION_NOTES.md`: "All data aligned to ophys imaging frame timestamps. Ophys timestamps serve as the common temporal reference. Running speed, pupil diameter, and stimulus information all interpolated/mapped to these timestamps." The whitepaper section on DATA SYNCHRONIZATION (all clocks recorded on one IO board at 100 kHz) justifies treating the streams as directly comparable.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning or resampling is applied** — data are kept at each experiment's native ophys frame rate. Because both rigs are included, the delivered dataset mixes two bin sizes: ~32.26 ms (31 Hz single-plane, 168 sessions) and ~93.2 ms (11 Hz Multiscope, 34 sessions). `metadata['time_bin_size']` reports a single number, the median imaging rate across sessions → 32.26 ms, which is wrong for the 34 Multiscope sessions. Trial lengths in the final data range from 77 to 389 frames.

ii.
```python
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
...
'time_bin_size': time_bin_ms,
```

iii. The AI explicitly recognized the conflict and decided against fixing it (step 34): "The metadata format demands a single time_bin_size value, but I have two different native frame rates depending on the imaging setup. I could resample everything to match, use the average, or just pick one and note the discrepancy. The cleanest solution is probably to stick with native timestamps and set time_bin_size to the single-plane rate since that's the majority of the data. I'll stick with native ophys timestamps across all experiments and let the decoder handle the variable temporal resolution since it processes each trial independently anyway." `CONVERSION_NOTES.md` records this as a "Known Limitation": "Multi-plane (Multiscope) experiments have lower frame rate (~11 Hz vs ~31 Hz), leading to fewer timepoints per trial."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations table** (`intervals/Natural_Images_Lum_Matched_set_*_presentations`): `image_name`, `start_time`, `stop_time`, and `omitted`. (All 284 NWB files contain exactly one such `Natural...` interval table, so the fallback branch in the key lookup is never used.) The trials table's `initial_image_name` / `change_image_name` are loaded but not used for this output.

ii.
```python
stim_keys = [k for k in f['intervals'] if 'Natural' in k or 'natural_scene' in k.lower()]
if len(stim_keys) == 0:
    stim_keys = [k for k in f['intervals'] if k not in ['trials', 'spontaneous_presentations', 'natural_movie_one_presentations']]
stim = f[f'intervals/{stim_key}']
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_image_names'] = np.array([s.decode() ... for s in stim['image_name'][:]])
data['stim_omitted'] = stim['omitted'][:]
```

iii. `CONVERSION_NOTES.md`: "For each ophys timepoint, the currently displayed image is determined from stimulus presentation intervals. Images are displayed for ~250 ms with ~500 ms ISI (gray screen)." Using the per-flash table is the most direct encoding of "the image presented during the non-grey screen" required by the instruction.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global, sorted list of all non-`omitted` image names is built in a first pass across all experiments (16 images: 8 from set A + 8 from set B) and used as the categorical code book. A session-length integer vector is initialized to −1 and filled with the image code for every ophys frame falling inside a (non-omitted) flash window. Within each trial, grey-screen and omitted frames are then **forward-filled** with the last displayed image, and any leading frames before the first flash are **back-filled** with the first image of the trial; a trial in which no image is ever present is dropped.

ii.
```python
all_image_names = sorted(all_image_names_set)
image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    img_name = stim_names[i]
    if img_name == 'omitted':
        continue
    img_idx = image_name_to_idx[img_name]
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```
```python
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
if trial_image[0] < 0:
    first_valid = np.where(trial_image >= 0)[0]
    if len(first_valid) > 0:
        trial_image[:first_valid[0]] = trial_image[first_valid[0]]
    else:
        continue
```

iii. `CONVERSION_NOTES.md`: "During gray screen periods, the last shown image identity is carried forward. Omitted stimuli are treated as gray screen (the image continues to be the last shown non-omitted image). Image indices map to sorted unique image names across all experiments." The forward fill keeps the variable defined at every time bin (the decoder requires a valid class at each bin) while preserving the identity of the most recently presented image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image code is computed on the full session ophys grid and then indexed with exactly the same `frame_indices` used for the neural matrix, so it is aligned frame-by-frame by construction. It is stored as row 0 of the `(5, n_timepoints)` output array.

ii.
```python
trial_image = image_at_ophys[frame_indices]
...
output_combined = np.stack([
    img_id.astype(np.int64),
    img_change.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_t, outcome, dtype=np.int64),
], axis=0)  # (5, n_timepoints)
```

iii. Same rationale as 2-d: everything is mapped onto the ophys frame grid first, so all streams share one index set. The sanity checks in the script assert `out.shape[1] == nt.shape[1]` for every trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` flag of the **stimulus presentations table**, together with that flash's `start_time`/`stop_time`. (The trials table's `change_time`/`go`/`is_change` are loaded but not used here.) Because the stimulus-table `is_change` is True only for genuine image changes, catch (sham-change) trials correctly receive all zeros.

ii.
```python
data['stim_is_change'] = stim['is_change'][:]
...
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. `CONVERSION_NOTES.md`: "Binary signal (0/1) marking timepoints during which a stimulus change occurred. Derived from `is_change` field in stimulus presentations table. Value is 1 during the change stimulus presentation, 0 otherwise." Step 31 notes that a catch trial is a "sham change", so the stimulus-level `is_change` flag is the right discriminator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Only the construction of the binary indicator: frames inside the change flash window are set to 1, everything else 0. The window is the flash itself (~250 ms, i.e. ~8 frames at 31 Hz / ~3 frames at 11 Hz); the following 500 ms grey period is **not** included. In the final dataset only 2.6% of time bins are labelled "change" (the human reference, which uses a 750 ms flash+grey window and marks only go trials, yields 7.7%).

ii. See the snippet in 4-a; the per-trial slice is
```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. The instruction asked for a variable that is "1 right after a change in image identity, otherwise 0"; the AI took this as literally the duration of the changed image flash. `CONVERSION_NOTES.md` states the 250 ms/500 ms stimulus structure explicitly and defines the indicator as "1 during the change stimulus presentation".

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is already binary. Values are `{0, 1}` with `output_values` `['no_change', 'change']`, and a sanity check asserts binarity.

ii.
```python
output_values = [
    all_image_names,
    ['no_change', 'change'],
    ...
]
...
assert np.all((out[1] == 0) | (out[1] == 1)), \
    f"Session {s_idx}, trial {t_idx}: image change not binary"
```

iii. Directly follows the instruction ("Image change, binary variable").

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Computed on the session-level ophys grid and sliced with the same `frame_indices` as the neural data; stored as row 1 of the output array. Alignment is therefore exact to the ophys frame, with up to one frame (32 ms / 93 ms) of quantization relative to the true monitor change time.

ii.
```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. Same rationale as 2-d and 3-c — a single shared ophys index set for all streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and `processing/running/speed/timestamps` from the NWB file — the SDK's filtered `running_speed` (cm/s), i.e. the wrap-corrected, transient-removed, 10 Hz low-pass filtered signal described in `methods.txt`.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. Step 92: "Running speed via `processing/running/speed/data` matches `experiment.running_speed.speed`" — confirmed against the SDK by a sub-agent. The whitepaper describes this attribute as the filtered running speed intended for end users.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation onto the ophys timestamps with `np.interp` (which clamps rather than extrapolates outside the recorded range, so no NaNs are produced), then discretization into 5 global percentile bins (see 5-c). This interpolation is performed twice: once in the statistics pass and again in the processing pass.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    """Interpolate a behavioral signal to ophys timestamps."""
    return np.interp(ophys_ts, signal_ts, signal)
...
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    ophys_ts
)
```

iii. `CONVERSION_NOTES.md`: "Running speed and pupil diameter are recorded at different rates than ophys. Both are linearly interpolated to ophys timestamps using `np.interp`." The behavioral streams are sampled at ~60 Hz/30 Hz, above the ophys rate, so linear interpolation to the (coarser) ophys grid is a light-touch resampling.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (0/20/40/60/80/100th percentiles) with edges computed **globally over all sessions**, using `np.percentile` on the concatenation of the interpolated running speed for **every ophys frame of every included experiment** — including inter-trial intervals, the grey-screen epochs and the movie block, not only the frames that end up inside trials. `np.digitize(values, edges[1:-1])` then maps each frame to bin 0-4. Edges: `[-23.98, 0.017, 1.83, 19.02, 35.06, 99.92]` cm/s. Because the edges come from all frames rather than from trial frames, the delivered per-bin fractions are 0.196/0.197/0.190/0.214/0.204 rather than exactly 0.2.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

def digitize_to_bins(values, edges):
    bins = np.digitize(values, edges[1:-1])
    return bins
```
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):   # first pass
    ...
    all_running.append(running)
all_running_flat = np.concatenate(all_running)
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
```

iii. `CONVERSION_NOTES.md`: "Global percentile bins computed across ALL sessions' running speed data (interpolated to ophys timestamps). 5 equal percentile bins (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)." A single global code book keeps the categories comparable across sessions, as required for a decoder trained on pooled data.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated onto the ophys grid before trial segmentation and then sliced with the same `frame_indices` as the neural data; stored as row 2 of the output array.

ii.
```python
trial_running = running_at_ophys[frame_indices]
...
running_binned = digitize_to_bins(running, running_edges)
```

iii. Same rationale as 2-d: all streams are put on the ophys clock first, which the whitepaper's hardware-synchronization section makes valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` (the fitted pupil-ellipse width), with timestamps from `acquisition/EyeTracking/eye_tracking/timestamps` and the blink flag from `acquisition/EyeTracking/likely_blink/data`. `pupil_area` is also loaded but unused. If the `pupil_tracking` group is absent (3 of the 202 active experiments), pupil is set to all-NaN.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking' in f:
    pupil = f['acquisition/EyeTracking/pupil_tracking']
    data['pupil_width'] = pupil['width'][:]
    data['pupil_area'] = pupil['area'][:]
    eye_ts = f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
    data['eye_timestamps'] = eye_ts
    data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
else:
    data['pupil_width'] = None
```

iii. Step 38 was a dedicated check of "the pupil diameter convention" in the NWB file, and step 92 records the confirmation that "Pupil tracking via `acquisition/EyeTracking/pupil_tracking/width` matches `experiment.eye_tracking.pupil_width`". `CONVERSION_NOTES.md`: "Pupil width from fitted ellipse used as diameter measure."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Samples flagged `likely_blink` are set to NaN and **excluded from the interpolation nodes**, so blink gaps are bridged by linear interpolation between the surrounding valid samples; the signal is then linearly interpolated onto the ophys grid with `np.interp` and discretized into 5 global percentile bins. If fewer than 10 valid samples exist, the session's pupil trace is all-NaN. As with running speed, this is computed twice (statistics pass and processing pass).

ii.
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(
        ophys_ts,
        nwb_data['eye_timestamps'][valid_mask],
        pupil_raw[valid_mask]
    )
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. `CONVERSION_NOTES.md`: "Pupil data during likely blinks (flagged in NWB) is treated as NaN before interpolation. NaN pupil values are excluded from interpolation; valid values are used for interp." This uses the SDK's pre-computed blink detector rather than re-deriving one, and prevents blink artifacts from entering the binned output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical scheme to running speed: 5 equal-percentile bin edges computed globally over all ophys frames of all sessions that have pupil data (edges `[4.81, 36.81, 41.44, 46.17, 52.78, 252.21]` px), applied with `np.digitize`. Sessions with no usable pupil signal are assigned the constant **middle bin (2)** for every frame of every trial. Delivered per-bin fractions are 0.188/0.185/0.223/0.208/0.195.

ii.
```python
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    # If no pupil data, use median bin
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. `CONVERSION_NOTES.md`: "Global percentile bins computed across all valid pupil data. 5 equal percentile bins. Sessions without pupil data: assigned median bin (bin 2)." The middle bin was chosen as the least-committal imputation, keeping the array shapes and class range valid.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the ophys grid before segmentation, then sliced with the same `frame_indices`; stored as row 3 of the output array.

ii.
```python
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. Same rationale as running speed — the eye-tracking camera (30 Hz) is hardware-synchronized to the ophys clock, so interpolating onto the ophys timestamps is valid and guarantees index-level alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
def get_trial_outcome(trial_idx, nwb_data):
    if nwb_data['trial_hit'][trial_idx]:
        return 0
    elif nwb_data['trial_miss'][trial_idx]:
        return 1
    elif nwb_data['trial_false_alarm'][trial_idx]:
        return 2
    elif nwb_data['trial_correct_reject'][trial_idx]:
        return 3
    else:
        return -1
```

iii. `CONVERSION_NOTES.md`: "Determined from trial flags: hit, miss, false_alarm, correct_reject." These are the canonical labels the whitepaper describes for the go/no-go change-detection task ("this trial structure leads to a sampling of 'GO' and 'CATCH' trials, that when combined with mouse responding, yields 'HIT', 'MISS', 'FALSE ALARM', and 'CORRECT REJECTION' trials").

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to a fixed integer code 0-3 and **broadcast across every time bin of the trial** so that the output array is uniformly `(5, n_timepoints)`; it is row 4. Trials that match none of the four flags return −1 and are dropped rather than emitted. A sanity check asserts the values are within 0-3. Final distribution: hit 31.0%, miss 56.5%, false alarm 1.8%, correct reject 10.7%.

ii.
```python
output_combined = np.stack([
    ...,
    np.full(n_t, outcome, dtype=np.int64),
], axis=0)
...
assert np.all(out[4] >= 0) and np.all(out[4] <= 3), \
    f"Session {s_idx}, trial {t_idx}: trial outcome out of range"
```

iii. `CONVERSION_NOTES.md`: "Static per trial (replicated across all timepoints for time-varying format)." The instruction asked for the outcome as a static per-trial value, but also said "If at all possible, make it time-varying"; replicating keeps a single homogeneous output tensor shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling present in the code:
- **Missing eye tracking** (3 of 202 experiments): pupil set to all-NaN, then the whole session is labelled with the constant middle bin 2.
- **Blinks**: dropped before interpolation (see 6-b).
- **Behavioural samples outside the ophys range**: `np.interp` clamps to the first/last value instead of producing NaN, so no NaN ever reaches `np.digitize`.
- **Omitted stimulus flashes / grey screen**: image identity forward-filled; leading frames back-filled; a trial with no stimulus at all is skipped.
- **Degenerate trials/sessions**: trials with <2 ophys frames dropped; trials with unresolvable outcome dropped; experiments with <2 trials or 0 neurons skipped and recorded in `metadata['skipped_experiments']` (empty in the full run).
- **Self-checks**: after assembly the script asserts trial-count, neuron-count and timepoint consistency, NaN/Inf-free neural data, and that every categorical value is in range.

Not handled: there is **no try/except around per-file loading**, so one unreadable or structurally atypical NWB file would abort the whole conversion (the human reference wraps each session in a try/except); and the 2,620 trials whose neural data is entirely zero are kept, only noted in the limitations section.

ii.
```python
    else:
        pupil_binned = np.full(len(pupil), 2, dtype=int)
```
```python
        if trial_image[0] < 0:
            first_valid = np.where(trial_image >= 0)[0]
            if len(first_valid) > 0:
                trial_image[:first_valid[0]] = trial_image[first_valid[0]]
            else:
                continue
```
```python
        assert out.shape[1] == nt.shape[1], f"Session {s_idx}, trial {t_idx}: timepoint mismatch"
        ...
        assert np.all(out[3] >= 0) and np.all(out[3] <= 4), \
            f"Session {s_idx}, trial {t_idx}: pupil bins out of range"
```

iii. `CONVERSION_NOTES.md` lists these under the processing pipeline and "Known Limitations" ("Pupil data quality varies across sessions; some sessions have no usable pupil data"; "Some trials have all-zero neural data (sparse calcium events)"). The AI's stated philosophy was to keep array shapes valid and the categorical ranges legal so that the decoder's format verification passes, and to log anything it could not fix.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is NWB file I/O, and the code incurs it **three times per experiment**: (1) an image-name pass that opens every file and reads its `image_name` column, (2) a statistics pass that calls the full `load_nwb_data()` (which reads the complete `events` *and* `dff` matrices, ~48,000 × n_cells float64 each, plus all stimulus/trial/behaviour arrays) merely to compute running/pupil percentiles, and (3) the processing pass that calls `load_nwb_data()` again. The second largest cost is the per-stimulus masking loops in `process_experiment`, which are O(n_flashes × n_ophys_frames) ≈ 4,800 × 48,000 ≈ 2×10⁸ element comparisons per experiment, done twice (image identity and image change). A third cost is the final `pickle.dump` of an 8.5 GB object.

ii.
```python
    for idx, (_, row) in enumerate(exp_table.iterrows()):   # pass 2: statistics only
        nwb_data = load_nwb_data(nwb_path)                  # reads events + dff + everything
        running = interpolate_to_ophys(...)
        all_running.append(running)
...
    for idx, (_, row) in enumerate(exp_table.iterrows()):   # pass 3: processing
        nwb_data = load_nwb_data(nwb_path)
```
```python
    for i in range(len(stim_starts)):
        ...
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        image_at_ophys[mask] = img_idx
```

iii. The AI did not discuss runtime or profile the code anywhere in the trajectory; the three-pass structure follows from its decision to compute *global* percentile bins and a *global* image code book before emitting any trial, which requires knowing all the data up front.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops are vectorizable:
1. The two per-stimulus-presentation loops that build `image_at_ophys` and `is_change_at_ophys` — each iteration allocates and scans a full session-length boolean mask. Both could be replaced by a single `np.searchsorted(stim_start_times, ophys_ts, side='right') - 1` lookup plus a validity test against the corresponding `stop_time`.
2. The per-frame Python loop that forward-fills image identity across grey periods, which is a standard "last valid value" scan expressible with `np.maximum.accumulate` over `np.where(valid, np.arange(n), 0)`.
3. The per-trial `frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)` scan, which is O(n_trials × n_frames) and could be two `np.searchsorted` calls (as the human reference does).

ii.
```python
    for i in range(len(stim_starts)):
        if stim_omitted[i] == 1.0:
            continue
        ...
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        image_at_ophys[mask] = img_idx
```
```python
        last_img = -1
        for k in range(len(trial_image)):
            if trial_image[k] >= 0:
                last_img = trial_image[k]
            elif last_img >= 0:
                trial_image[k] = last_img
```
```python
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        frame_indices = np.where(frame_mask)[0]
```

iii. No justification is given in the trajectory — the AI wrote the conversion script in a single `Write` call (step 41) and only revisited it to fix the output dtype and NaN-handling order (steps 49-59). Efficiency was never raised as a consideration.

## 9-c. What processing does the code repeat multiple times?

i. Repeated work:
- Every NWB file is **opened three times** (image-name pass, statistics pass, processing pass), and fully parsed twice via `load_nwb_data()`.
- Running-speed and pupil interpolation onto the ophys grid is performed **twice per experiment** with identical code (once to gather percentile statistics, once to segment).
- Blink masking / NaN filtering of the pupil trace is likewise duplicated.
- The stimulus `image_name` column is read in the image-name pass and again in `load_nwb_data()`.
- The final sanity-check block re-walks every session and trial several times (dimension checks, NaN scan, range checks).

ii.
```python
    # first pass
    running = interpolate_to_ophys(nwb_data['running_speed'], nwb_data['running_timestamps'],
                                   nwb_data['ophys_timestamps'])
    ...
    pupil = np.interp(nwb_data['ophys_timestamps'], nwb_data['eye_timestamps'][valid_mask],
                      pupil_raw[valid_mask])
```
```python
    # second pass, inside process_experiment()
    running_at_ophys = interpolate_to_ophys(nwb_data['running_speed'],
                                            nwb_data['running_timestamps'], ophys_ts)
    ...
    pupil_at_ophys = np.interp(ophys_ts, nwb_data['eye_timestamps'][valid_mask],
                               pupil_raw[valid_mask])
```

iii. Not discussed. The structural reason is that the global percentile edges must be known before any trial is written, and the AI chose to recompute from disk rather than cache the (small) interpolated traces or the per-trial slices in memory, as the human reference does.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose result is never used:
- **dF/F traces** are read in full from every NWB file (`data['dff']`) in both passes and never used — the neural output is the event traces. This is the single largest piece of wasted I/O.
- `pupil_area`, `events_timestamps`, `cell_specimen_ids`, `stim_is_change` (loaded in the image-name pass), and several trials-table columns (`trial_change_times`, `trial_is_change`, `trial_change_image`, `trial_initial_image`) are loaded but never consumed.
- `image_at_ophys` and `is_change_at_ophys` are computed for **every frame of the whole session**, including the ~10 minutes of grey screen, the movie block and all aborted-trial periods, although only the frames inside kept trials survive.
- The `valid_roi` filter is a no-op on the released NWB files (all ROIs are flagged valid).
- `valid_trial_indices` is built up but always equals `range(len(raw_output_trials))`, and `final_neural = [neural_trials[i] for i in valid_trial_indices]` is therefore a pure copy.
- `process_experiment` returns `running_at_ophys`, `pupil_at_ophys` and `events_full` to the caller, which ignores all three.

ii.
```python
        dff = f['processing/ophys/dff/traces/data'][:]
        data['dff'] = dff  # shape: (n_timepoints, n_cells)
```
```python
        data['pupil_area'] = pupil['area'][:]
        ...
        data['trial_change_times'] = trials['change_time'][:]
        data['trial_is_change'] = trials['is_change'][:]
        data['trial_change_image'] = np.array([...])
        data['trial_initial_image'] = np.array([...])
```
```python
            final_output_trials.append(output_combined)
            valid_trial_indices.append(t_idx)      # always every index
        ...
        final_neural = [neural_trials[i] for i in valid_trial_indices]
```
```python
    neural_trials, raw_output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full = result
    # running_at_ophys, pupil_at_ophys, events_full are never used again
```

iii. Not justified anywhere. The dF/F load is described in the code comment as being kept "for fallback / comparison" (the AI initially considered dF/F vs events before settling on events per the paper), and the unused trials-table columns are leftovers from the exploration phase in which the AI examined both the trials table and the stimulus-presentations table as candidate sources for image identity and image change.
