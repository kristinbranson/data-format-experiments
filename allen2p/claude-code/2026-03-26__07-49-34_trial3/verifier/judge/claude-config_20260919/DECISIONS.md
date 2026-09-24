# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK entirely and read the NWB (HDF5) files directly with `h5py`. The candidate experiment list comes from the project metadata CSV `project_metadata/ophys_experiment_table.csv`, intersected with the NWB files actually present on disk (284 files) and then filtered by `session_type` to the four *active* behavior sessions (`OPHYS_1/3/4/6`). This yields 202 experiments / 38 mice (168 from project `VisualBehavior`, 34 from `VisualBehaviorMultiscope`). Per experiment it reads: `processing/ophys/image_segmentation/cell_specimen_table` (valid_roi), `processing/ophys/dff/traces/timestamps` (ophys timebase), `processing/ophys/event_detection/data` (neural), `intervals/trials`, the natural-images `intervals/<...>_presentations` table, `processing/running/speed`, and `acquisition/EyeTracking/pupil_tracking`. Three passes are made over every file: (1) collect global image names, (2) collect all running/pupil samples for global percentile bins, (3) load + process.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'
ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]

def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        eid = int(f.stem.split('_')[-1]); nwb_ids.add(eid)
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()
```
```python
with h5py.File(nwb_path, 'r') as f:
    cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
    valid_roi = cell_table['valid_roi'][()].astype(bool)
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    events_valid = events_data[:, valid_roi]
    trials_grp = f['intervals']['trials']
    ...
    running_speed = f['processing']['running']['speed']['data'][()]
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
```

iii. From CONVERSION_NOTES Step 1/2: the AI inspected the AllenSDK loaders (`BehaviorOphysExperiment.from_nwb`, `DFFTraces.from_nwb`, `Events.from_nwb`, `Trials.from_nwb`, …) to learn *where* each field lives inside the NWB file, then read those same HDF5 paths with h5py for speed ("h5py reads same data as AllenSDK NWB reader", Step 10 Check 3). Only the 284 files on disk are used; the metadata table lists 1,936 experiments, so the on-disk set is explicitly treated as a subset. Passive sessions (`OPHYS_2`, `OPHYS_5`) are excluded because "passive viewing (no lick spout, satiated mice). No meaningful trial outcomes."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiments, sorted and stored as strings; each session row gets `subject_idx` = index of its mouse. 38 mice result.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. Key Decision 12 in CONVERSION_NOTES: "Subject IDs: Use mouse_id from experiment table." `mouse_id` is the Allen unique animal identifier; the count (38) was cross-checked against the on-disk subset in Step 4.

## 1-c. How are the data split into sessions?

i. **One NWB experiment (= one imaging plane) becomes one "session"** in the output. No grouping by `ophys_session_id` is performed. Combined with the `session_type` filter this gives 202 output sessions: 168 single-plane `VisualBehavior` experiments (where plane == session, identical to grouping) and 34 `VisualBehaviorMultiscope` planes that actually come from only 8 physical sessions of a single mouse (457841), which therefore appears with 34 "sessions" in the verification output while other mice have 3–9. Passive sessions are dropped before this step.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
    region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. Key Decision 13: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." The AI was aware of multi-plane structure (Step 1 notes: "Multi-plane handling: Automatic timestamp interleaving for Multiscope. Each plane gets `ophys_timestamps[plane_group::group_count]`") and of the mixed project codes (Step 2 lists all four project codes), but chose the plane as the unit because each plane carries its own neuron set and its own `targeted_structure`. It did not discuss the consequence that a multiscope mouse's behavioral trials are duplicated across planes.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if it is a Go **or** Catch trial and is not aborted and not auto-rewarded. The trial window is the full `start_time` → `stop_time` interval (variable length, mean ≈ 255 bins ≈ 8.5 s at 30 Hz, range 210–377), taken as the 30 Hz grid samples falling in `[start_time, stop_time)`.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
```

iii. Instructions state "Segment each recording session into individual trials based on how they are defined in the experiment. Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"; the AI used the experiment's own trials table and its boolean flags verbatim (CONVERSION_NOTES Step 5, decisions 3 and 5: "Trial window: Use trial start_time to stop_time from trials table. Variable length across trials.").

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order: (a) experiment level — drop passive session types, drop an experiment with zero valid ROIs, drop an experiment whose stimulus-presentation table cannot be found, drop an experiment with <2 valid trials or <2 successfully processed trials; (b) trial level — drop aborted and auto-rewarded, drop trials with fewer than 3 time bins in the window, drop trials that match none of hit/miss/false_alarm/correct_reject. No quality filter on behavioural signals or on d-prime/engagement. In the full run, no experiment was dropped (202/202 kept) and the minimum trials per session was 39.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) < 2:
    print(f"  WARNING: Only {len(valid_indices)} valid trials ... skipping")
    return None
...
    if len(trial_time_indices) < 3:
        continue
...
    if trials['hit'][trial_idx]:      outcome = 0
    elif trials['miss'][trial_idx]:   outcome = 1
    elif trials['false_alarm'][trial_idx]: outcome = 2
    elif trials['correct_reject'][trial_idx]: outcome = 3
    else:
        continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} processed trials ... skipping")
    return None
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": aborted trials are excluded from performance calculations by the Allen pipeline because the mouse licked before the change; auto-rewarded trials are free rewards. The ≥2-trial requirement comes from the format spec ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). The 10 whitepaper session-QC criteria were noted in Step 3 but deliberately not re-implemented, since the released dataset has already passed them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The deconvolved calcium **events**: `processing/ophys/event_detection/data` (shape `(n_timepoints, n_cells)`), with the ophys timebase taken from `processing/ophys/dff/traces/timestamps`. dF/F is *not* used.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```
```python
'neural_signal': 'calcium events (FastLZeroSpikeInference)',
```

iii. The AI deliberated between dF/F and events in its reasoning (trajectory step 36) and chose events on the strength of the reference paper: "Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." (Key Decision 1). The methods text it read states "For all analysis of neural data we used the detected calcium events as described in Garrett et al." Its initial instinct was dF/F, which it overruled to match the paper.

## 2-b. How is the `neural` data processed?

i. Three operations only: (1) select `valid_roi` columns; (2) linearly interpolate every cell's event trace from the native ophys timestamps onto a uniform 30 Hz grid spanning the whole session; (3) clip to ≥ 0. No event filtering/smoothing (the SDK's causal half-gaussian `filtered_events` is not used), no normalisation, no z-scoring, no baseline subtraction, no merging of planes. Stored as float32 `(n_neurons, n_timepoints)` per trial.

ii.
```python
def interpolate_to_regular_grid(timestamps, data, target_timestamps):
    if data.ndim == 1:
        return np.interp(target_timestamps, timestamps, data)
    else:
        result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
        return result
```
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)   # events should be >= 0
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES Step 6: "Resamples all data streams to 30 Hz via linear interpolation … Events are clipped to >=0 after interpolation." The 30 Hz grid is justified by the paper quote "linearly interpolating onto a consistent set of 30hz timestamps relative to the triggering behavioral event", and by the need for one common bin size across the 31 Hz Scientifica and 11 Hz Multiscope rigs. The sparsity consequence (2,602 trials with all-zero neural data) was noted in Step 10 and dismissed: "This is NOT a bug: calcium events are sparse (~0.25% of timepoints nonzero)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `valid_roi` boolean from the `cell_specimen_table` is applied; nothing else (no SNR, event-rate, or trace-quality filter). In practice this filter is a no-op on the released NWB files — every file checked has `valid_roi` all True, because the SDK already excluded invalid ROIs when writing. All 29,444 ROIs in the 202 selected experiments are kept.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. Step 1/Step 3 notes: "Cell filtering: Only automatic filter is `valid_roi` boolean (SVM binary classifier output). `exclude_invalid_rois=True` by default." Step 10 Check 3(b): "valid_roi filter matches SDK default (exclude_invalid_rois=True)." The AI verified the neuron count against the NWB independently (12 valid ROIs in NWB vs 12 in converted session 0).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: the 30 Hz samples with `start_time <= t < stop_time` are taken. All streams (neural, running, pupil, image identity, image change) are indexed by the *same* `trial_time_indices` into the *same* session-wide 30 Hz grid, so they are aligned by construction. Metadata records `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'`, `off_start = 0.0`, `off_end = None` (variable-length trials).

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
running_trial = running_resampled[trial_time_indices]
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Key Decision 6: "Alignment event: Trial start time (stimulus onset). off_start=0, off_end=None (variable)." The instructions require aligning on the ophys timestamp; the AI built one master 30 Hz timebase derived from the ophys timestamps and resampled every other stream onto it, arguing (Step 3, item 4) that the hardware NI board synchronises all clocks so cross-stream interpolation is valid. I verified independently that the change-signal onset falls within 13 ms (< 1 bin) of the NWB `change_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 33.333 ms (30 Hz), identical for every trial and every session. Yes — every stream is rebinned (resampled) by linear interpolation from its native rate onto `np.arange(ophys_ts[0], ophys_ts[-1], 1/30)`. For Scientifica single-plane data this is a slight downsample (≈30.95 → 30 Hz); for the 34 Multiscope planes it is a ~2.8× **upsample** (≈10.7 → 30 Hz), which creates interpolated samples that carry no new information.

ii.
```python
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
t_start = ophys_ts[0]; t_end = ophys_ts[-1]; dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. Key Decision 4: "Resample to 30 Hz: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz). time_bin_size = 33.33 ms." This also satisfies the format requirement "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The natural-images stimulus-presentation table inside `intervals/` (e.g. `Natural_Images_Lum_Matched_set_training_2017_presentations`): its `image_name`, `start_time`, and `omitted` columns. The trials table's `initial_image_name` / `change_image_name` are loaded but not used for this output.

ii.
```python
stim = f['intervals'][stim_key]
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. Step 5 mapping table: "`stimulus_presentations.image_name` → output[0]: image_identity, map to categorical int, time-varying per ophys frame, 8 natural images". Using the presentation table gives the image actually on screen at each moment rather than inferring it from trial-level fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted flashes are removed; for each 30 Hz sample the most recent remaining flash onset is found with `searchsorted`, and that flash's image name is assigned — so the identity is *held through the 500 ms grey period and through omitted flashes*. Names are first mapped to a per-experiment sorted index, then remapped to a global sorted vocabulary built in Pass 1 over all 202 files (16 names: the 8 images of set A plus the 8 of set B). Samples before the first flash of the session get index 0.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
        else:
            image_idx[i] = 0
    return image_idx
```
```python
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    local_to_global[local_idx] = global_image_names.index(name) if name in global_image_names else 0
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. Key Decisions 7 and 8: "Image identity during gray screen: Use the identity of the image that was just shown (last presented image)"; "Image identity for omitted flashes: Continue with previous image identity." The global vocabulary makes the integer codes comparable across sessions/image sets; the resulting distribution is near-uniform (each of 16 images 5.8–6.7 % of samples), matching the "8 images per session, 2 image sets" expectation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at exactly the trial's 30 Hz sample times (`trial_ts = regular_ts[trial_time_indices]`), the same indices used to slice the neural array, so both have the same length and the same time origin.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Step 10 Check 2 sanity check: "Image identity (session 0, trial 5, timepoint 10): expected 'im063', got 'im063'. PASS." I independently confirmed in `sample_data.pkl` that the image-identity code changes on exactly the bin where the change indicator turns on, in 345/345 change trials of the two sample sessions.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The stimulus-presentation table's `is_change`, `omitted` and `start_time` columns (not the trials table's `change_time`, which is loaded but unused).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. Step 5 mapping: "`trials.is_change` + stimulus timing → output[1]: image_change, binary 1 at change timepoint, 0 otherwise, time-varying, 1 for one 750ms window at change." `is_change` is True only for genuine identity changes, so catch (sham-change) trials automatically get an all-zero indicator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is built per trial: 1 for the 750 ms starting at each change-flash onset (i.e. the changed image's 250 ms presentation plus the following 500 ms grey), 0 elsewhere. For each trial the code loops over every change in the whole session and masks the trial's timepoints.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_signal = np.zeros(len(timepoints), dtype=np.int64)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. Step 6: "Image change signal: 1 during 750ms window starting at change onset", justified by the 250 ms stimulus + 500 ms grey = 750 ms image interval documented in the whitepaper. The AI checked the resulting statistic against expectation: 7.7 % of samples flagged, "matches expected 1/13 flashes"; and "Catch trials: 0/6,515 have image_change signal; Go trials: 0/45,477 missing image_change signal."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: `output_values[1] = ['no_change', 'change']`, 0/1, no thresholding of a continuous quantity. The only "threshold" is the temporal one — the 0.75 s window after change onset.

ii.
```python
mask = (timepoints >= cs) & (timepoints < cs + 0.75)
change_signal[mask] = 1
...
['no_change', 'change'],  # image change values
```

iii. As above — the window length is set to one full image interval so that the label marks the transient change event rather than the whole post-change period. I measured the realised window in `sample_data.pkl`: 22–23 bins = 733–767 ms, contiguous, in every change trial.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: evaluated on `trial_ts`, the identical 30 Hz sample times used to slice the neural matrix.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
```

iii. Alignment is guaranteed by construction (single master timebase). My independent check against NWB `change_time` for experiment 951980471, trial 5: the first `1` bin is 13 ms after `change_time` (one 30 Hz bin), and the neural slice reproduced exactly (`np.allclose(..., True)`).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` (the running-wheel encoder speed in cm/s, ~60 Hz).

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. Step 2 notes identify this as the SDK's running-speed stream (~60 Hz); Step 5 maps `running/speed/data → output[2]`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the ~60 Hz encoder timestamps onto the session's 30 Hz grid (`np.interp`, which clamps rather than extrapolating), then per-sample digitisation into 5 bins. No smoothing, no outlier rejection; negative (backward) speeds are retained.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Step 6: "Resamples all data streams to 30 Hz via linear interpolation." Same justification as the neural data — one master ophys-derived timebase for every stream.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (`np.percentile` at 0/20/40/60/80/100). Crucially, the edges are computed in Pass 2 from **all raw 60 Hz samples of all 202 experiments over the whole session**, including inter-trial, grey-screen, movie and spontaneous periods — not from the trial-window, 30 Hz-resampled values that are actually binned. Edges are made strictly increasing by an epsilon nudge; NaN → bin 0. Realised distribution in the converted data: 19.9 / 20.3 / 18.4 / 20.9 / 20.5 % (target 20 % each). Full-run edges: `[-24.13, -0.0041, 0.336, 14.18, 33.13, 99.92]`.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    return edges

def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```
```python
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
```

iii. Instructions require "discretized into five equal percentile bins". Global (rather than per-session) edges were chosen so the category labels mean the same thing in every session; Step 7 notes "Running speed bins slightly unequal within sessions (expected, bins computed globally)" and Step 9 records 18–21 % per bin as acceptable.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is resampled onto the session-wide 30 Hz grid before trial segmentation and then sliced with the same `trial_time_indices` as the neural data.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Same single-master-timebase argument (Step 3 item 4: all clocks are hardware-synchronised via the NI PCI-6612 board, so cross-stream interpolation is legitimate).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (pupil **area** from the fitted ellipse) with its timestamps, plus `acquisition/EyeTracking/likely_blink/data` to mark blinks. Pupil width/diameter columns are not used.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. Step 5 mapping row: "`EyeTracking/pupil_tracking/area` → output[3]: pupil_diameter … **Use pupil area as proxy for diameter**." Because the output is discretised into percentile bins, a monotone transform of diameter gives (nearly) the same categories.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN; all NaNs (blinks and tracking failures) are filled by linear interpolation over *sample index*; the filled trace is then linearly interpolated onto the 30 Hz grid and digitised into 5 percentile bins. If an experiment has no eye-tracking group at all, the whole pupil channel is NaN for that experiment and therefore becomes bin 0 everywhere (this happened for 3 of 202 experiments: 795953296, 833631914, 806456687).

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```
```python
def interpolate_nans(arr):
    nans = np.isnan(arr)
    if not nans.any():  return arr.copy()
    if nans.all():      return np.zeros_like(arr)
    result = arr.copy(); x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result
```

iii. Key Decision 9: "Pupil NaN handling: During blinks (likely_blink=True or NaN), linearly interpolate. Compute percentile bins from non-blink data." Step 4 measured ~9 % blink frames, so bridging them was judged preferable to leaving holes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 5 equal-percentile bins with edges computed globally in Pass 2 from every experiment's non-blink raw pupil-area samples (full session, native 30 Hz camera rate), applied to the trial-window resampled values; NaN → bin 0. Full-run edges `[125.6, 4374, 5528, 6747, 8599, 323783]`; realised distribution 22.8 / 18.5 / 19.1 / 19.0 / 20.6 %.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Same rationale as running speed (global bins → session-comparable labels). The deviation from exactly 20 % (bin 0 at 22.8 %) is partly the global-vs-trial edge mismatch and partly the 3 experiments whose pupil channel is entirely bin 0.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Resampled onto the session 30 Hz grid before segmentation, then sliced with the same `trial_time_indices` as the neural data; if absent, an all-NaN vector of the trial length is used so shapes always match.

ii.
```python
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. Same single-master-timebase justification; the eye camera runs at 30 Hz, so resampling to the 30 Hz grid is close to a one-to-one mapping.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
'hit': trials_grp['hit'][()].astype(bool),
'miss': trials_grp['miss'][()].astype(bool),
'false_alarm': trials_grp['false_alarm'][()].astype(bool),
'correct_reject': trials_grp['correct_reject'][()].astype(bool),
```

iii. Step 5 mapping: "`trials.hit/miss/false_alarm/correct_reject` → output[4]: trial_outcome, static per-trial categorical, 4 classes." These are the canonical go/no-go outcome labels for the change-detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A priority cascade maps the booleans to 0=hit, 1=miss, 2=false_alarm, 3=correct_reject; a trial matching none of the four is dropped. The scalar is then broadcast to a constant row of length `n_timepoints` so the output block is uniformly `(5, n_timepoints)`. Realised distribution over the full dataset: hit 30.2 %, miss 57.2 %, FA 1.7 %, CR 10.8 %.

ii.
```python
if trials['hit'][trial_idx]:            outcome = 0
elif trials['miss'][trial_idx]:         outcome = 1
elif trials['false_alarm'][trial_idx]:  outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
else:
    continue  # Unknown outcome, skip
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. The format spec says outputs should be time-varying "if at all possible", and the verifier expects a consistent `(n_output, n_timepoints)` block, so the static label is repeated across bins. The AI cross-checked outcome consistency (Step 12: hit/miss only on change trials, FA/CR only on catch trials) — I reproduced this on the sample data and it holds exactly.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) Whole-file failures are caught and the experiment is skipped with a printed `ERROR`/`WARNING` (`load_experiment_data` returns `None`); (b) missing eye-tracking group → `pupil_area = None` → all-NaN pupil for that experiment → bin 0 (3 experiments, kept in the dataset with a constant pupil label); (c) blinks and NaNs in pupil → linearly interpolated; (d) NaN after any digitisation → bin 0; (e) negative interpolation artefacts in events → clipped to 0; (f) byte-string image names decoded, float `omitted` column cast to bool; (g) trials shorter than 3 bins, or with no recognised outcome, are skipped; (h) experiments with <2 usable trials are skipped; (i) image names absent from the global vocabulary fall back to index 0. `np.interp` clamps at the edges, so streams that start/end slightly inside the ophys range are held constant rather than producing NaN.

ii.
```python
    except Exception as e:
        print(f"  ERROR loading {experiment_id}: {e}")
        return None
```
```python
            except (KeyError, Exception):
                print(f"  WARNING: No pupil tracking data in {experiment_id}")
```
```python
binned[np.isnan(values)] = 0
...
events_resampled = np.maximum(events_resampled, 0)
...
if stim_data['omitted'].dtype == float:
    stim_data['omitted'] = stim_data['omitted'].astype(bool)
```

iii. Step 6: "Handle missing data appropriately"; Step 10 Check 5 reports 8 edge-case sub-checks passing (no NaN in neural, all events non-negative, all output values in range, outcome constant within trials, valid region/subject indices, ≥2 trials per session). The AI did not discuss the consequence of the 3 pupil-less experiments being labelled all-bin-0 rather than dropped.

## 9-a. What are the most time-consuming steps of the code?

i. Measured from `conversion_full_out.txt` (total 358.2 s for 202 experiments): Pass 3 = 312 s, split into NWB reading 183 s (59 %) and processing 125 s (40 %); Passes 1+2 (re-opening every file to collect image names and running/pupil samples) ≈ 37 s (10 %); pickling the 8.3 GB output 9.1 s. Within "processing", the dominant cost is interpolating every cell's full-session trace onto the 30 Hz grid (a Python loop calling `np.interp` once per cell over ~140 k source samples). The AI's own Step 7 estimate was ~2–3 min; the real run took 6 min (~2–3× the estimate), which it did not revisit.

ii.
```python
t0 = time.time()
raw_data = load_experiment_data(nwb_path, eid)
t_load = time.time() - t0
result = process_single_experiment(raw_data, exp_meta)
t_process = time.time() - t0 - t_load
print(f"  Loaded in {t_load:.1f}s, processed in {t_process:.1f}s, total {t_total:.1f}s")
```

iii. Step 6: "Code inefficiencies identified: Sequential processing of experiments (could parallelize); Multiple passes over NWB files." Step 7 concluded the projected ~2–3 min was "well under 15 min limit", so no further optimisation was done.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain: (1) `get_image_at_timepoints` runs a Python `for i in range(n_tp)` over every timepoint of every trial although `insert_idx` is already a vectorised `searchsorted` result — this is pure fancy-indexing (`np.asarray(stim_names)[insert_idx]`); (2) `interpolate_to_regular_grid` loops over cells calling `np.interp` per column (`scipy.interpolate.interp1d(..., axis=0)` handles the 2-D case at once); (3) `get_image_change_at_timepoints` loops over *all* change presentations of the session for *every* trial (~300 × 300 per session) when only the one change inside the trial matters — a `searchsorted` on change onsets would do it in O(1) per trial. Also `[local_to_global.get(v, 0) for v in trial_data_out['image_identity']]` is a per-timepoint Python loop that could be a lookup-array index, and `ProcessPoolExecutor` is imported but never used, so experiments are processed strictly sequentially despite being embarrassingly parallel.

ii.
```python
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
```
```python
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```
```python
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```
```python
from concurrent.futures import ProcessPoolExecutor, as_completed   # never used
```

iii. Step 6 claims "Code speedups added: Vectorized interpolation using np.interp; Efficient searchsorted for image identity assignment". Both mechanisms are present, but each is still wrapped in a Python loop, so the claimed speedups are only partly realised. The AI judged further optimisation unnecessary because the total runtime was under the 15-minute budget.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB file is opened and read **three** times: Pass 1 re-reads the stimulus table to collect image names, Pass 2 re-reads running speed and pupil area (and re-applies the blink mask) to compute percentile edges, Pass 3 re-reads everything again. The set of image names is computed twice (globally in Pass 1 and again per experiment as `all_stim_images` inside `process_single_experiment`). Blink masking of the pupil trace is likewise done twice. Measured cost of the redundant passes: ≈37 s of 358 s (~10 %).

ii.
```python
    # First pass: determine global image names
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f: ...
    # --- Pass 2 ---
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        with h5py.File(nwb_path, 'r') as f:
            running = f['processing']['running']['speed']['data'][()]
            pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]...
    # --- Main processing pass ---
    for idx, (_, row) in enumerate(exp_table.iterrows()):
        raw_data = load_experiment_data(nwb_path, eid)
```
```python
    all_stim_images = sorted(set(
        raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
    ))   # recomputed per experiment, already known globally from Pass 1
```

iii. Step 6 explicitly lists "Multiple passes over NWB files" under "Code inefficiencies identified", but the AI chose not to fix it: the three-pass structure is what lets it build the *global* image vocabulary and *global* percentile bins before writing any trial, and the runtime stayed inside budget. (Caching the small per-file quantities from Pass 1/2, or reading them once in Pass 3 with a deferred binning step, would have removed the repetition.)

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Events, running speed and pupil are interpolated onto the 30 Hz grid for the **entire session** — including aborted trials, inter-trial intervals, the natural-movie block and spontaneous/grey periods — but only the Go/Catch trial windows are kept, so roughly a third of the most expensive computation is thrown away. (2) For the 34 Multiscope planes the ~10.7 Hz data is upsampled ~2.8× to 30 Hz, tripling storage and downstream compute without adding information. (3) The `valid_roi` filter is a no-op: the released NWB files already contain only valid ROIs (verified: every file sampled has `valid_roi` all True). (4) Several loaded fields are never used: `trials['change_time']`, `trials['is_change']`, `cell_specimen_ids` (loaded *and* index-filtered), and `change_stops`/`ce` in the change loop. (5) Pass 2 concatenates every raw running and pupil sample of all 202 experiments into memory purely to take 6 percentiles. (6) `interpolate_nans` fills the whole pupil trace, most of which lies outside trials. (7) `ProcessPoolExecutor`/`as_completed` are imported but unused.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)              # whole session
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
# ... only trial_time_indices are ever used
```
```python
'change_time': trials_grp['change_time'][()],     # never read again
'is_change':   trials_grp['is_change'][()].astype(bool),  # never read again
'cell_specimen_ids': cell_specimen_ids[valid_roi],        # never read again
    for cs, ce in zip(change_starts, change_stops):       # ce unused
```

iii. CONVERSION_NOTES does not identify any discarded work under this heading; the closest statements are the acknowledged multi-pass I/O (Step 6) and the decision to resample whole streams up front so that "all data streams" share one timebase (Step 6), which is what makes the whole-session interpolation convenient. The wasted work is real but modest relative to file I/O, and none of it affects correctness of the saved data.
