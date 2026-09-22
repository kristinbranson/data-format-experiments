# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs every `/app/data/sub-*/*.nwb` file, performs an HDF5 metadata inventory, and then opens each retained file again with `h5py` for conversion. NWB groups supply units, trials, events, video, electrodes, and subject/session identifiers.

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
inventory, brain_regions = session_inventory(paths)
...
with h5py.File(path, 'r') as f:
    tr = f['intervals/trials']
    events = f['acquisition/BehavioralEvents']
```

iii. The notes say direct NWB/HDF5 access uses the same underlying MAP streams while avoiding export ambiguity. The inventory prepass deterministically finds 174 files and rejects the one session without classifier-good units.

## 1-b. How are the data split into subjects?

i. A subject is the parent `sub-*` directory name with `sub-` removed. Unique IDs are sorted, and each session receives an index into that list.

ii.
```python
subjects = sorted({os.path.basename(os.path.dirname(x[0])).removeprefix('sub-') for x in inventory})
subject_lookup = {x:i for i,x in enumerate(subjects)}
...
subject = os.path.basename(os.path.dirname(path)).removeprefix('sub-')
```

iii. The notes describe filename/path parsing as analogous to the reference loader and verify 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Output session order is sorted path order; the session ID is parsed from the filename.

ii.
```python
for i, (path, expected_units, expected_trials) in enumerate(inventory):
    n, x, y, subject, ridx, stats = process_session(path, region_lookup, ...)
...
session_id = os.path.basename(path).split('_behavior')[0]
```

iii. The agent identified the dataset as one NWB per session and reports 173 retained sessions, matching the paper after excluding the unclassified file.

## 1-d. How are the data split into trials?

i. The AI takes the number of columns in `units/is_good_trials` as the represented trial count, uses that prefix of the trials table, and maps the unique go event (and latest pre-go tone) within each trial's start/stop interval. After neural binning, population-wide all-zero windows are removed with the same mask from every stream.

ii.
```python
n_trials = f['units/is_good_trials'].shape[1]
trial_starts = tr['start_time'][:n_trials]
trial_stops = tr['stop_time'][:n_trials]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
...
trial_keep = np.any(rates != 0, axis=(1, 2))
trial_indices = np.flatnonzero(trial_keep)
```

iii. The notes say nine files contain behavior beyond the neural represented prefix. The zero-population filter was added after verification found 2,576 physiologically implausible four-second gaps despite true source validity flags.

## 1-e. How are trials filtered based on quality controls?

i. Trials are limited to the `is_good_trials` represented prefix and then filtered if every retained neuron has zero spikes in the requested window. Early, miss, ignore, photostimulation, auto-water, and free-water trials are otherwise retained. Sessions must retain at least two trials.

ii.
```python
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
...
trial_keep = np.any(rates != 0, axis=(1, 2))
...
if n_trials < 2:
    raise ValueError(f'{path}: fewer than two valid nonzero neural trials')
```

iii. The AI argues the requested decoder labels require retaining behavioral categories normally excluded by movement analyses, and that all-zero population windows are acquisition gaps rather than genuine silence. It explicitly retains free-water trials, unlike the human reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times`, with `units/spike_times_index` delimiting units. Go-cue timestamps define absolute bin edges, and unit classification/validity fields determine retained units.

ii.
```python
spike_data = f['units/spike_times']
endpoints = f['units/spike_times_index'][:]
...
edges_abs = go[:, None] + EDGES_REL[None, :]
```

iii. The notes identify spike times as the authoritative neural stream and compare direct `np.histogram` results against converted rates.

## 2-b. How is the `neural` data processed?

i. For each retained unit, one `searchsorted` over flattened edges obtains counts for all trials. Counts in half-open 50-ms bins are divided by 0.05 to produce float32 spikes/s. There is no smoothing or normalization.

ii.
```python
positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
```

iii. The agent states this matches the reference histogram/rate logic, with the task-required bin geometry, and validated it independently against raw spikes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and must be marked good on every represented trial. A session with no such unit is skipped. This extra all-trials rule removes 565 otherwise classifier-good units.

ii.
```python
class_good = classifier_mask(f)
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
```

iii. The notes justify the extra rule as necessary to maintain a fixed neuron population per session without fabricating values or varying dimensions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The unique go cue inside each trial is time zero. Relative edges from -2.5 to +1.5 s are added to its absolute timestamp, and session-absolute spikes are binned against those edges.

ii.
```python
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
edges_abs = go[:, None] + EDGES_REL[None, :]
rates = bin_selected_units(f, unit_indices, edges_abs)
```

iii. The notes emphasize that all NWB streams share the absolute clock, avoiding clock-offset mistakes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 50 ms: 80 contiguous bins spanning -2.5 to +1.5 s. Raw spike times are histogrammed into these bins; no further rebinning is applied.

ii.
```python
BIN_SIZE = 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

iii. These values directly implement the decoder instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial start/stop boundaries, the trial's go cue, and the shared bin centers. The selected tone is the latest sample event in that trial strictly before go.

ii.
```python
sample_events = events['sample_start_times/timestamps'][:]
tone = trial_event_mapping(trial_starts, trial_stops, sample_events,
                           'sample/tone onset', before=go, use_last=True)
```

iii. The notes say repeated task-state events can occur, so the latest pre-go sample onset is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each absolute go-aligned bin center has the selected tone timestamp subtracted, yielding a continuous seconds-since-tone ramp.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. Long values were investigated and attributed mainly to real early-lick state-machine delays rather than an alignment error.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact centers of the same go-aligned bins whose edges define neural rates.

ii.
```python
edges_abs = go[:, None] + EDGES_REL[None, :]
centers_abs = go[:, None] + CENTERS_REL[None, :]
tone_elapsed = centers_abs - tone[:, None]
```

iii. The notes report direct raw-NWB comparison via `np.allclose`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It comes from absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, evaluated at the absolute decoder bin centers.

ii.
```python
laser_starts = events['photostim_start_times/timestamps'][:]
laser_stops = events['photostim_stop_times/timestamps'][:]
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```

iii. The agent chose exact event intervals and notes that these agree with trial-table stimulation status.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary array starts at zero and is set to one wherever a bin center falls in any half-open laser interval `[start, stop)`.

ii.
```python
state = np.zeros(centers_abs.shape, dtype=bool)
for a, b in zip(starts, stops):
    state |= (centers_abs >= a) & (centers_abs < b)
return state.astype(np.float32)
```

iii. This produces the requested time-varying on/off input from exact timestamps.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Laser state is sampled at the same absolute go-aligned centers corresponding to each neural bin.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```

iii. Shared NWB timestamps and the common grid provide alignment without resampling offsets.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from the trials-table `outcome` and `trial_instruction`: ignore means no lick, hit means the instructed side, and miss means the opposite side.

ii.
```python
for i, (o, side) in enumerate(zip(outcome_str, instruction)):
    if o == 'ignore': choice[i] = 2
    elif o == 'hit': choice[i] = 0 if side == 'left' else 1
    else: choice[i] = 1 if side == 'left' else 0
```

iii. The notes call these authoritative labels and report 99.742% agreement with first post-go lick direction on responsive non-early trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Values are encoded left=0, right=1, no lick=2 and repeated across all 80 time bins.

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. Tiling lets per-trial and time-varying outputs coexist in one rectangular `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from `intervals/trials/outcome` for retained source-trial indices.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
```

iii. The source already supplies exactly the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map as ignore=0, miss=1, hit=2, then are repeated through 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_str], dtype=np.int64)
outputs[:, 1, :] = outcome[:, None]
```

iii. The mapping follows the requested category order; repetition provides a common output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` strings for retained trials.

ii.
```python
early_str = decode_array(tr['early_lick'][:])[trial_indices]
```

iii. The source explicitly labels the required property.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1, then the value is tiled over 80 bins.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int64)
outputs[:, 2, :] = early[:, None]
```

iii. The notes retain early-lick trials because this is a required decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (tracking likelihood) from `Camera0_side_TongueTracking`.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t = tongue['timestamps'][:]
video_data = tongue['data'][:]
y = np.asarray(data[:, 1], dtype=np.float64).copy()
likelihood = np.asarray(data[:, 2], dtype=np.float64)
```

iii. The notes identify the side-camera marker as the relevant tongue stream.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Visibility requires finite y/likelihood and likelihood at least 0.9. The AI computes frame-to-frame velocity for contiguous visible frames, flags destination frames over five standard deviations from mean velocity, and interpolates those outliers. Low-confidence frames remain invisible. For each decoder center, it takes the nearest video frame only if within 20 ms.

ii.
```python
visible = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_CUTOFF)
...
bad_pair = valid_pair & (np.abs(velocity - center) > threshold)
...
y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
...
video_idx = nearest_indices(video_t, target)
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
```

iii. The AI says this matches the method paper's five-sigma cleanup while preserving low-confidence samples as the task's explicit not-visible class. It argues 0.9 is robust because likelihood is highly bimodal.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th/60th percentiles are computed over cleaned, visible raw frames. A sampled visible y below q40 is 0, q40 through q60 inclusive is 1, and above q60 is 2; low-confidence, missing, or temporally distant samples are 3.

ii.
```python
q40, q60 = np.percentile(y[final_visible], [40, 60])
...
tongue_class[sampled_visible & (sampled_y < percentiles[0])] = 0
tongue_class[sampled_visible & (sampled_y >= percentiles[0]) & (sampled_y <= percentiles[1])] = 1
tongue_class[sampled_visible & (sampled_y > percentiles[1])] = 2
```

iii. The AI interprets “over the session” as all cleaned visible session frames and reserves class 3 for occlusion/missing samples.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The 80 absolute neural-bin centers are flattened; the closest camera timestamp to each is selected, subject to a 20-ms tolerance, and reshaped back to trials by time.

ii.
```python
target = centers_abs.ravel()
video_idx = nearest_indices(video_t, target)
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
...
tongue_class = tongue_class.reshape(n_trials, N_TIME)
```

iii. The notes report normal nearest-frame differences below 2 ms and direct reproduction in an independent sanity check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. An unclassified session is skipped; units not valid on every represented trial are removed; population-wide zero-spike trial windows are removed; event mapping failures raise errors; tongue velocity outliers are interpolated; and low-confidence/missing/distant tongue frames become class 3. Internal assertions validate all final shapes and values.

ii.
```python
if not mask.any():
    print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
...
trial_keep = np.any(rates != 0, axis=(1, 2))
...
tongue_class = np.full(target.shape, 3, dtype=np.int64)
```

iii. The notes describe these as fixes for unlabeled sessions, unit-specific invalidity, implausible acquisition gaps, tracker artifacts, and genuine occlusion, with reruns after each discovered issue.

## 10-a. What are the most time-consuming steps of the code?

i. Bulk HDF5 reads, per-unit spike `searchsorted`, accumulating/writing the roughly 11.94-GB pickle, and (when enabled) plotting are the largest costs. The full conversion took 164.37 s; neural binning was optimized to roughly fractions of a second per session.

ii.
```python
for j, unit in enumerate(unit_indices):
    spikes = spike_data[starts[unit]:endpoints[unit]]
    positions = np.searchsorted(spikes, flat_edges, side='left')
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify naive neuron-by-trial-by-bin work, repeated video scans, and repeated HDF5 reads as potential bottlenecks and report a sub-10-minute estimate before the faster full run.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The neural loop remains per unit because spike arrays are ragged, while all trials/edges are vectorized within it. Choice construction could be vectorized; event mapping remains a Python loop over trials; photostimulation loops over intervals; electrode decoding loops over rows. Nearest-frame tongue alignment is vectorized.

ii.
```python
for j, unit in enumerate(unit_indices):
    ...
for i, (a, b) in enumerate(zip(starts, stops)):
    ...
for i, (o, side) in enumerate(zip(outcome_str, instruction)):
    ...
for a, b in zip(starts, stops):
    state |= (centers_abs >= a) & (centers_abs < b)
```

iii. The notes specifically credit flattening all trial edges per unit and session-wise NumPy operations for the achieved speed; remaining small loops were not reported as limiting.

## 10-c. What processing does the code repeat multiple times?

i. Every retained NWB is opened once in `session_inventory` and again in `process_session`; classification, `is_good_trials`, masks, neuron counts, and electrode regions are consequently read/derived twice. The main conversion otherwise computes each stream once per session.

ii.
```python
def session_inventory(paths):
    for path in paths:
        with h5py.File(path, 'r') as f:
            class_good = classifier_mask(f)
            ...
            regions = electrode_regions(f, mask)
...
with h5py.File(path, 'r') as f:
    class_good = classifier_mask(f)
    ...
    labels = electrode_regions(f, mask)
```

iii. The AI calls the first pass a small metadata-only inventory that detects exclusions and establishes global regions without loading bulk spikes; it accepts the repeated metadata work for deterministic assembly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The inventory's per-session unit/trial counts are used only for progress messages; extensive per-session statistics are stored as metadata but are not decoder features. Tongue cleanup returns likelihood/outlier arrays mainly for optional plots/statistics. With `--show-processing`, figures are diagnostic and not consumed by the decoder. These are validation/documentation overhead, not core conversion needs.

ii.
```python
kept.append((path, int(mask.sum()), n_trials))
...
print(f'[{i+1}/{len(inventory)}] ... expected={expected_trials}x{expected_units}', flush=True)
...
stats = {'tongue_outliers': int(outliers.sum()), ...}
if make_plot:
    make_processing_plot(...)
```

iii. The notes justify this extra work as sanity checking, timing/progress reporting, diagnostics, and metadata documentation. Normal full conversion does not generate plots.
