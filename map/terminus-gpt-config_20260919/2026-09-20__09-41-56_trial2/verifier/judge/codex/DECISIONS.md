# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files with a sorted glob over `/app/data/sub-*/*.nwb`. It does a first `session_inventory()` pass with `h5py` to reject sessions that have no retained units and collect region labels, then reopens each kept NWB file in `process_session()` and reads the needed HDF5 datasets directly (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`).

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
inventory, brain_regions = session_inventory(paths)
```

```python
with h5py.File(path, 'r') as f:
    class_good = classifier_mask(f)
    ...
    tr = f['intervals/trials']
    events = f['acquisition/BehavioralEvents']
```

iii. In `CONVERSION_NOTES.md`, the AI says it chose a "metadata-only inventory prepass" so it could detect exclusions and collect regions before full conversion, and it describes direct NWB/HDF5 access as a way to use the source data directly rather than a MATLAB-exported representation.

## 1-b. How are the data split into subjects?

i. The AI treats the parent directory name (`sub-<id>`) as the subject identifier. It strips the `sub-` prefix per session, then builds `subjects` as the sorted unique set and `subject_idx` as a per-session lookup into that list.

ii.
```python
subject = os.path.basename(os.path.dirname(path)).removeprefix('sub-')
```

```python
subjects = sorted({os.path.basename(os.path.dirname(x[0])).removeprefix('sub-') for x in inventory})
subject_lookup = {x:i for i,x in enumerate(subjects)}
...
subject_idx.append(subject_lookup[subject])
```

iii. The AI's notes say the dataset layout is one directory per subject and one NWB per session, so it treated the folder name as the canonical subject split.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list after inventory filtering, and each session id is derived from the filename stem before `"_behavior"`.

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
```

```python
session_id = os.path.basename(path).split('_behavior')[0]
```

iii. In the notes, the AI states that `/app/data` contains one NWB per session, so the file boundary is the session boundary and no additional grouping logic is needed.

## 1-d. How are the data split into trials?

i. The AI does not use all behavioral trial rows directly. Instead, it sets the session trial count to `units/is_good_trials.shape[1]`, takes the first `n_trials` rows of `intervals/trials`, and maps one go cue and the last pre-go sample event into each of those trial intervals.

ii.
```python
n_trials = f['units/is_good_trials'].shape[1]
tr = f['intervals/trials']
trial_starts = tr['start_time'][:n_trials]
trial_stops = tr['stop_time'][:n_trials]
...
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
tone = trial_event_mapping(trial_starts, trial_stops, sample_events,
                           'sample/tone onset', before=go, use_last=True)
```

iii. The AI justified this in `CONVERSION_NOTES.md` by noting that some sessions have fewer `is_good_trials` columns than trial-table rows and concluding that conversion should use the "represented" neural trials when building trial-aligned arrays.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways. First, the AI restricts the session to the `is_good_trials` trial count and keeps only units that are valid on every represented trial. Second, after neural binning it drops any trial whose entire population response is zero across all units and all bins. It does not filter `free_water` trials explicitly.

ii.
```python
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
```

```python
rates = bin_selected_units(f, unit_indices, edges_abs)
trial_keep = np.any(rates != 0, axis=(1, 2))
trial_indices = np.flatnonzero(trial_keep)
...
rates = rates[trial_keep]
trial_starts = trial_starts[trial_keep]
trial_stops = trial_stops[trial_keep]
go = go[trial_keep]
tone = tone[trial_keep]
```

iii. The notes say the AI added the always-valid-unit filter after finding false `is_good_trials` cells in a few sessions, and later added the zero-spike trial exclusion after verifying that the all-zero trials were raw source gaps rather than a binning bug.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`, with the per-trial absolute bin edges defined from each trial's go cue in `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
spike_data = f['units/spike_times']
endpoints = f['units/spike_times_index'][:]
starts = np.r_[0, endpoints[:-1]]
```

```python
go_events = events['go_start_times/timestamps'][:]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
edges_abs = go[:, None] + EDGES_REL[None, :]
```

iii. The AI's notes describe the NWB spike times as the direct neural source and the go cue as the event needed to place the requested decoder window.

## 2-b. How is the `neural` data processed?

i. The AI bins each retained unit's spike times into 80 non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue. It uses `np.searchsorted` on the flattened bin-edge array for all trials at once, differences adjacent cumulative counts, and divides by `BIN_SIZE` to output firing rates in spikes/s.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

```python
for j, unit in enumerate(unit_indices):
    spikes = spike_data[starts[unit]:endpoints[unit]]
    positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
    rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
```

iii. In the notes, the AI explicitly says it matched the reference histogram logic while changing only the geometry to the task-required contiguous 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `units/classification == 'good'` and then further requires those units to have `True` for every represented `is_good_trials` entry. If no such units remain, the session is skipped.

ii.
```python
def classifier_mask(f):
    return decode_array(f['units/classification'][:]) == 'good'
```

```python
class_good = classifier_mask(f)
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
if not mask.any():
    print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
    continue
```

iii. The notes say classifier QC was chosen to match the paper, and the extra `is_good_trials` requirement was added to avoid variable neuron availability across trials within a session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go-cue onset. The AI finds the go cue that falls inside each retained trial interval and adds the fixed relative edge grid `[-2.5, 1.5]` to that absolute go time to define the neural bin edges.

ii.
```python
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
edges_abs = go[:, None] + EDGES_REL[None, :]
centers_abs = go[:, None] + CENTERS_REL[None, :]
```

iii. The notes repeatedly state that all NWB streams are already on one absolute time base, so alignment only requires identifying the correct go cue per trial and constructing the shared go-centered grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins and always has 80 time bins per trial over a 4 s window. No additional smoothing or rebinning is applied after the single histogramming step.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
N_TIME = len(CENTERS_REL)
```

iii. The AI's notes say it kept the decoder task's required 50 ms windowing, rather than the 40 ms / 3.4 ms geometry used elsewhere in the reference pipeline.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` and the per-trial go-cue time. The AI chooses the last sample-start event before the go cue within the trial.

ii.
```python
sample_events = events['sample_start_times/timestamps'][:]
tone = trial_event_mapping(trial_starts, trial_stops, sample_events,
                           'sample/tone onset', before=go, use_last=True)
```

iii. The notes explain that early licks can replay the sample epoch, so there may be multiple sample/tone events per trial and the final pre-go one is the relevant tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes absolute bin centers aligned to go cue and subtracts the trial's tone-onset time, producing a continuous time-varying value in seconds for each bin.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. The notes justify this as the direct "seconds since tone onset" representation required by the decoder task.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns it by computing it on the same `centers_abs` grid used for the neural bins, so each input timepoint corresponds exactly to one neural time bin.

ii.
```python
edges_abs = go[:, None] + EDGES_REL[None, :]
centers_abs = go[:, None] + CENTERS_REL[None, :]
...
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. The notes state that all aligned streams are expressed on the common go-centered grid built from the absolute NWB timestamps.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the absolute event streams `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `acquisition/BehavioralEvents/photostim_stop_times/timestamps`, not from the trials-table `photostim_onset`/`photostim_duration` fields.

ii.
```python
laser_starts = events['photostim_start_times/timestamps'][:]
laser_stops = events['photostim_stop_times/timestamps'][:]
```

iii. In the notes, the AI says it used exact NWB event start/stop intervals and sampled them directly on the decoder grid rather than reconstructing them from string-valued trial-table offsets.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI builds a binary time series by testing whether each absolute bin center falls inside any laser interval `[start, stop)`. It outputs `1.0` when the laser is on and `0.0` otherwise.

ii.
```python
def build_photostim(centers_abs, starts, stops):
    state = np.zeros(centers_abs.shape, dtype=bool)
    for a, b in zip(starts, stops):
        state |= (centers_abs >= a) & (centers_abs < b)
    return state.astype(np.float32)
```

```python
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```

iii. The notes describe this as an exact interval-membership calculation on the aligned time axis, producing the requested time-varying binary input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned by evaluating the laser state at the same absolute bin centers used for neural binning. No separate interpolation or trial-local offset is applied.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```

iii. The notes say the photostim events and the go-centered neural bins already share the same session clock, so alignment reduces to comparing the same timestamps.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from the NWB file. The AI derives it from the trial-table `outcome` and `trial_instruction` columns.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
instruction = decode_array(tr['trial_instruction'][:])[trial_indices]
```

iii. The notes say this follows the task definition: a hit means the animal chose the instructed side, a miss means it chose the opposite side, and an ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `ignore -> 2` ("no lick"), `hit -> instructed side`, and `miss -> opposite side`, with left coded as `0` and right as `1`. The resulting per-trial label is tiled across all 80 bins.

ii.
```python
choice = np.empty(n_trials, dtype=np.int64)
for i, (o, side) in enumerate(zip(outcome_str, instruction)):
    if o == 'ignore': choice[i] = 2
    elif o == 'hit': choice[i] = 0 if side == 'left' else 1
    else: choice[i] = 1 if side == 'left' else 0
```

```python
outputs[:, 0, :] = choice[:, None]
```

iii. The notes justify tiling because choice is per-trial but the output tensor must share a common rectangular `(n_output, n_timepoints)` shape with the time-varying tongue output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial-table `outcome` column.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
```

iii. The notes say the NWB trials table already contains exactly the requested outcome categories, so no additional derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `'ignore' -> 0`, `'miss' -> 1`, and `'hit' -> 2`, then repeats that categorical value across all 80 bins of output row 1.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_str], dtype=np.int64)
...
outputs[:, 1, :] = outcome[:, None]
```

iii. The notes say this direct categorical mapping follows the decoder task's required label order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick comes directly from the trial-table `early_lick` column.

ii.
```python
early_str = decode_array(tr['early_lick'][:])[trial_indices]
```

iii. The notes describe this as an authoritative per-trial label already present in the source data.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early' -> 0` and `'early' -> 1`, then tiles that per-trial label across all 80 bins of output row 2.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int64)
...
outputs[:, 2, :] = early[:, None]
```

iii. The notes say this preserves early-lick trials because early lick is itself one of the required decoder outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using the `timestamps`, the y-coordinate `data[:, 1]`, and the tracking likelihood `data[:, 2]`.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t = tongue['timestamps'][:]
video_data = tongue['data'][:]
```

```python
y = np.asarray(data[:, 1], dtype=np.float64).copy()
likelihood = np.asarray(data[:, 2], dtype=np.float64)
```

iii. The AI's notes say all sessions include side-camera tongue tracking and that this NWB stream is the direct source for the requested tongue output.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI applies a multi-step cleanup and sampling procedure. It keeps only frames with `likelihood >= 0.9`, computes frame-to-frame tongue-y velocities on contiguous visible pairs, marks 5-sigma velocity outliers, linearly interpolates the outlier y values from neighboring clean visible frames, computes per-session thresholds from the cleaned visible y values, then samples the nearest video frame to each neural bin center if that frame is within 20 ms.

ii.
```python
LIKELIHOOD_CUTOFF = 0.9
MAX_VIDEO_DT = 0.020
```

```python
visible = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_CUTOFF)
...
velocity[valid_pair] = np.diff(y)[valid_pair] / dt[valid_pair]
...
if outlier.any():
    y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
```

```python
target = centers_abs.ravel()
video_idx = nearest_indices(video_t, target)
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
sampled_visible = visible[video_idx] & close
sampled_y = clean_y[video_idx]
```

iii. The notes justify this as matching the method-paper marker-cleanup ideas: use high-confidence frames, repair implausible jumps, and align video to the decoder grid by nearest timepoint.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI computes the 40th and 60th percentiles from the cleaned visible tongue-y samples over the session, then assigns class `0` below q40, `1` between q40 and q60 inclusive, `2` above q60, and `3` when the tongue is not visible or no nearby frame is available.

ii.
```python
q40, q60 = np.percentile(y[final_visible], [40, 60])
return y, likelihood, final_visible, outlier, np.array([q40, q60])
```

```python
tongue_class = np.full(target.shape, 3, dtype=np.int64)
tongue_class[sampled_visible & (sampled_y < percentiles[0])] = 0
tongue_class[sampled_visible & (sampled_y >= percentiles[0]) & (sampled_y <= percentiles[1])] = 1
tongue_class[sampled_visible & (sampled_y > percentiles[1])] = 2
```

iii. The notes say the AI wanted session-wise q40/q60 thresholds as required by the task, but applied them after confidence filtering and outlier cleanup so the thresholds were defined on "cleaned visible" tongue positions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output by taking the nearest video frame to each neural bin center on the go-centered time grid and requiring that nearest frame to be within 20 ms; otherwise the bin becomes class `3`.

ii.
```python
target = centers_abs.ravel()
video_idx = nearest_indices(video_t, target)
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
sampled_visible = visible[video_idx] & close
...
tongue_class = tongue_class.reshape(n_trials, N_TIME)
```

iii. The notes justify nearest-frame sampling by citing the ~300 Hz video rate and arguing that the nearest frame is typically much closer than 20 ms to the neural bin center.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly. Sessions with no retained units are skipped. Event-mapping failures raise an error. Trials with all-zero neural activity are excluded after binning. Low-confidence or temporally distant tongue samples are mapped to the "not visible" class, while high-confidence velocity outliers are interpolated instead of discarded.

ii.
```python
if not mask.any():
    print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
    continue
```

```python
if len(candidates) == 0:
    raise ValueError(f'trial {i}: no {name} event in [{a}, {hi}]')
```

```python
trial_keep = np.any(rates != 0, axis=(1, 2))
...
sampled_visible = visible[video_idx] & close
tongue_class = np.full(target.shape, 3, dtype=np.int64)
```

iii. The notes say the zero-spike-trial removal was added only after direct raw-data checks showed physiologically implausible acquisition gaps, and the tongue interpolation was justified as a reference-style cleanup for brief high-confidence tracking artifacts.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies HDF5/NWB file I/O, loading the large spike-time and tongue-tracking arrays, the per-unit `searchsorted` spike binning loop, and final pickle writing as the dominant costs.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    rates = bin_selected_units(f, unit_indices, edges_abs)
```

```python
video_t = tongue['timestamps'][:]
video_data = tongue['data'][:]
```

```python
with open(args.outpicklefile, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says the full runtime is dominated by reading each NWB file, the spike buffer, the tongue array, the per-unit searchsorted loop, and pickling the large result.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI partially vectorized the neural computation already, but the code still contains explicit Python loops over sessions, units, event intervals, and trial intervals. The clearest remaining vectorization opportunities are `trial_event_mapping()`, `build_photostim()`, and the per-trial/per-unit loops that remain around otherwise vectorized NumPy operations.

ii.
```python
for path in paths:
    with h5py.File(path, 'r') as f:
        ...
```

```python
for i, (a, b) in enumerate(zip(starts, stops)):
    ...
```

```python
for j, unit in enumerate(unit_indices):
    spikes = spike_data[starts[unit]:endpoints[unit]]
    positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
```

```python
for a, b in zip(starts, stops):
    state |= (centers_abs >= a) & (centers_abs < b)
```

iii. The notes say the AI intentionally kept a per-unit `searchsorted` loop because spike trains are ragged, while vectorizing across the trial dimension and most of the array arithmetic session-wise.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats a metadata prepass over all files before the real conversion pass. That means every kept file is opened at least twice, and some work such as classifier masking, `is_good_trials` reduction, and electrode-region extraction is repeated in both `session_inventory()` and `process_session()`.

ii.
```python
def session_inventory(paths):
    for path in paths:
        with h5py.File(path, 'r') as f:
            class_good = classifier_mask(f)
            ...
            regions = electrode_regions(f, mask)
```

```python
inventory, brain_regions = session_inventory(paths)
...
for i, (path, expected_units, expected_trials) in enumerate(inventory):
    n, x, y, subject, ridx, stats = process_session(path, region_lookup, ...)
```

iii. The notes justify this duplication as a deliberate "metadata-only inventory prepass" used to determine which sessions are usable and to build the global region list before full conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does extra diagnostic and bookkeeping work beyond what the decoder needs. The metadata prepass, optional processing plots, timing/logging, and per-session statistics such as tongue percentiles, class counts, and zero-spike-trial counts are not needed to build the core `neural`/`input`/`output` tensors used downstream.

ii.
```python
stats = {
    'session_id': session_id, 'subject': subject, 'n_trials': n_trials,
    'n_neurons': len(unit_indices), 'tongue_q40': float(percentiles[0]),
    'tongue_q60': float(percentiles[1]), 'tongue_outliers': int(outliers.sum()),
    'choice_counts': np.bincount(choice, minlength=3).tolist(),
    'outcome_counts': np.bincount(outcome, minlength=3).tolist(),
    'early_counts': np.bincount(early, minlength=2).tolist(),
    'tongue_counts': np.bincount(tongue_class.ravel(), minlength=4).tolist(),
    'photostim_on_bins': int(photo.sum()), 'excluded_zero_spike_trials': n_zero,
}
```

```python
if make_plot:
    make_processing_plot(session_id, rates, inputs, outputs, percentiles,
                         video_t, clean_y, likelihood, visible, outliers, go[0])
```

iii. The notes say this extra work was intentional for validation, reporting, and debugging rather than because the downstream decoder required it.
