# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files via `glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb'))`, sorts them, and processes each using `h5py.File` (not `pynwb`). A two-pass approach is used: first `session_inventory()` does a metadata prepass to identify usable sessions and collect brain regions, then `process_session()` loads and converts each session fully.

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
inventory, brain_regions = session_inventory(paths)
# ...
for i, (path, expected_units, expected_trials) in enumerate(inventory):
    n, x, y, subject, ridx, stats = process_session(path, region_lookup, ...)
```

```python
def process_session(path, region_lookup, make_plot=False):
    with h5py.File(path, 'r') as f:
        # ...reads units, trials, events, tongue tracking...
```

iii. The AI chose `h5py` for direct HDF5 access rather than `pynwb`. The two-pass design (inventory then processing) allows pre-computing which sessions to keep and the global brain region list. The CONVERSION_NOTES.md states this avoids loading bulk spike data for sessions that will be excluded.

## 1-b. How are the data split into subjects?

i. Subjects are derived from directory names: the `sub-<id>` prefix of each NWB file's parent directory. Unique subject IDs are sorted and indexed.

ii.
```python
subject = os.path.basename(os.path.dirname(path)).removeprefix('sub-')
subjects = sorted({os.path.basename(os.path.dirname(x[0])).removeprefix('sub-') for x in inventory})
subject_lookup = {x:i for i,x in enumerate(subjects)}
```

iii. Subject IDs are numeric strings from directory names (e.g., `440956`), which correspond to `nwb.subject.subject_id`. The CONVERSION_NOTES.md confirms 28 subjects were found, matching the reference papers.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each is identified by parsing the filename (removing the `_behavior+ecephys+ogen.nwb` suffix). Sessions are processed in sorted file-path order.

ii.
```python
session_id = os.path.basename(path).split('_behavior')[0]
```

iii. The AI identifies sessions from filenames. 173 of 174 sessions are kept (one dropped for having no classifier-good units). This matches the paper count of 173 sessions.

## 1-d. How are the data split into trials?

i. The AI determines trial count from the `is_good_trials` matrix shape: `n_trials = f['units/is_good_trials'].shape[1]`. Trial data is then read from `intervals/trials` for those indices. Go cue and tone onset events are mapped to trials using `trial_event_mapping()`, which searches for events within each trial's `[start_time, stop_time]` interval.

ii.
```python
n_trials = f['units/is_good_trials'].shape[1]
tr = f['intervals/trials']
trial_starts = tr['start_time'][:n_trials]
trial_stops = tr['stop_time'][:n_trials]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
```

iii. The AI uses `is_good_trials.shape[1]` as the number of "represented" trials, noting that 9 sessions have fewer columns in `is_good_trials` than total behavioral trials. This represents using a prefix of trials that have ephys coverage.

## 1-e. How are trials filtered based on quality controls?

i. Two filtering stages: (1) During `session_inventory()`, units must be `classification == 'good'` AND have all `is_good_trials` entries true. (2) After neural binning, trials with population-wide zero spikes are excluded (`np.any(rates != 0, axis=(1, 2))`). There is no explicit filtering based on `free_water` or `obs_intervals`. Sessions with fewer than 2 valid trials are rejected.

ii.
```python
# In session_inventory:
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid

# In process_session, after binning:
trial_keep = np.any(rates != 0, axis=(1, 2))
rates = rates[trial_keep]
```

iii. The AI explains that `is_good_trials` represents which trials are covered by curated-unit observation intervals. The zero-spike trial exclusion was added after the full verification revealed 2,576 all-zero trials (which the AI considered "physiologically impossible population-wide raw-spike gaps"). The AI does NOT filter `free_water` trials, retaining them because they are "structurally valid" trials with required labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time arrays) and `units/spike_times_index` (offsets into the ragged array). Only units passing the combined `classification == 'good'` AND `always_valid` filter are included. Go cue times provide the alignment reference.

ii.
```python
spike_data = f['units/spike_times']
endpoints = f['units/spike_times_index'][:]
starts = np.r_[0, endpoints[:-1]]
# ...
for j, unit in enumerate(unit_indices):
    spikes = spike_data[starts[unit]:endpoints[unit]]
```

iii. The AI reads spike times directly via h5py rather than through the pynwb API.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 contiguous 50ms bins spanning [-2.5, 1.5) s relative to the go cue. Bin edges are computed as `go[:, None] + EDGES_REL[None, :]`. For each unit, `np.searchsorted` finds spike positions at all edges, `np.diff` gives counts, and division by `BIN_SIZE` (0.05) converts to Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
edges_abs = go[:, None] + EDGES_REL[None, :]
rates = np.empty((n_trials, len(unit_indices), N_TIME), dtype=np.float32)
flat_edges = absolute_edges.ravel()
for j, unit in enumerate(unit_indices):
    spikes = spike_data[starts[unit]:endpoints[unit]]
    positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
    rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
```

iii. Same vectorized approach as the reference: one `searchsorted` per unit across all trial edges. The AI stores rates as `(n_trials, n_neurons, n_time)` and later transposes when splitting into per-trial arrays.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must satisfy two criteria: (1) `classification == 'good'` (the spike-sorting QC classifier result), and (2) `np.all(is_good_trials)` — the unit must be valid on every represented trial. This removes 565 additional units beyond the classifier filter alone (69,453 → 68,888). Sessions with no qualifying units are dropped.

ii.
```python
class_good = classifier_mask(f)  # classification == 'good'
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
unit_indices = np.flatnonzero(mask)
```

iii. The AI explains that four sessions contain some classifier-good units with false `is_good_trials` entries, and since the target format requires a fixed neuron population per session, those partially-valid units are excluded. The reference solution only uses `classification == 'good'` without the `is_good_trials` filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The go cue onset is found for each trial by searching `go_start_times/timestamps` events within the trial's `[start_time, stop_time]` interval using `trial_event_mapping()`. Bin edges are then computed as `go + EDGES_REL` in absolute time.

ii.
```python
go_events = events['go_start_times/timestamps'][:]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
edges_abs = go[:, None] + EDGES_REL[None, :]
```

```python
def trial_event_mapping(starts, stops, event_times, name, before=None, use_last=False):
    for i, (a, b) in enumerate(zip(starts, stops)):
        lo_idx = np.searchsorted(event_times, a, side='left')
        hi_idx = np.searchsorted(event_times, hi, side='right')
        # expects exactly one go event per trial
```

iii. The AI's approach explicitly searches for go events within trial boundaries rather than assuming positional alignment. This is functionally equivalent to the reference's direct indexing since there is one go event per trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins total, from -2.5 to +1.5 s relative to go cue. Bin edges are computed via `np.linspace(OFF_START, OFF_END, 81)`. No rebinning or smoothing is applied.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)  # 80
```

iii. The AI uses `np.linspace` for edge construction (vs. the reference's `T_START + BIN * np.arange(N_BINS + 1)`). Both produce identical edges for these values.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times/timestamps` (the tone/sample onset events) and the per-trial go cue. The tone for each trial is the last sample-start event within the trial interval that occurs before the go cue.

ii.
```python
sample_events = events['sample_start_times/timestamps'][:]
tone = trial_event_mapping(trial_starts, trial_stops, sample_events,
                           'sample/tone onset', before=go, use_last=True)
```

iii. The AI uses `trial_event_mapping` with `use_last=True` and `before=go` to find the last sample onset before the go cue within each trial. This handles early-lick trials where the sample epoch replays.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is computed as `(centers_abs - tone[:, None])`, where `centers_abs = go[:, None] + CENTERS_REL[None, :]`. This gives continuous, time-varying values in seconds.

ii.
```python
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
inputs = np.stack([tone_elapsed, photo], axis=1)
```

iii. Straightforward subtraction. The result is equivalent to `CENTERS + (go - tone)` as in the reference.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin center grid (`CENTERS_REL` relative to go cue), so each time-from-tone value corresponds 1:1 with the neural bins.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. Same grid alignment as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times/timestamps` and `photostim_stop_times/timestamps` in the `BehavioralEvents` acquisition group — these are event-level timestamps of laser on/off.

ii.
```python
laser_starts = events['photostim_start_times/timestamps'][:]
laser_stops = events['photostim_stop_times/timestamps'][:]
```

iii. The AI uses event-level timestamps rather than the per-trial columns (`photostim_onset` / `photostim_duration`) used by the reference. Both should represent the same information but from different data sources within the NWB file.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary array is built: for each laser start/stop pair, bin centers falling in `[start, stop)` are marked as 1. Multiple intervals are OR'd together via accumulation.

ii.
```python
def build_photostim(centers_abs, starts, stops):
    state = np.zeros(centers_abs.shape, dtype=bool)
    for a, b in zip(starts, stops):
        state |= (centers_abs >= a) & (centers_abs < b)
    return state.astype(np.float32)
```

iii. This loops over all photostim intervals in the session (not per-trial), applying each to the full `(n_trials, N_TIME)` centers array. The reference instead computes per-trial onset/offset from trial-table columns.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The absolute bin centers (`go + CENTERS_REL`) are compared directly against absolute photostim start/stop timestamps. Since both are on the same NWB clock, no offset correction is needed.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```

iii. Same clock, same grid as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trial_instruction` (left/right) and `outcome` (ignore/miss/hit) in the trials table. Choice is derived: ignore → no lick (2), hit → instructed side, miss → opposite side.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
instruction = decode_array(tr['trial_instruction'][:])[trial_indices]
choice = np.empty(n_trials, dtype=np.int64)
for i, (o, side) in enumerate(zip(outcome_str, instruction)):
    if o == 'ignore': choice[i] = 2
    elif o == 'hit': choice[i] = 0 if side == 'left' else 1
    else: choice[i] = 1 if side == 'left' else 0
```

iii. Same derivation logic as the reference. The AI verified choice against post-go lick timestamps (99.7% agreement).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as left=0, right=1, no lick=2. Per-trial values are tiled across all 80 time bins.

ii.
```python
outputs[:, 0, :] = choice[:, None]
```

iii. Same encoding and tiling as the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column in the trials table, which contains the strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_str], dtype=np.int64)
```

iii. Direct mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to ignore=0, miss=1, hit=2 and tiled across all 80 bins.

ii.
```python
outputs[:, 1, :] = outcome[:, None]
```

iii. Same encoding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column in the trials table, containing `'no early'` and `'early'`.

ii.
```python
early_str = decode_array(tr['early_lick'][:])[trial_indices]
early_map = {'no early': 0, 'early': 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int64)
```

iii. Direct mapping from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1 and tiled across all 80 bins.

ii.
```python
outputs[:, 2, :] = early[:, None]
```

iii. Same encoding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data (tongue_x, tongue_y, tongue_likelihood) with corresponding timestamps.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t = tongue['timestamps'][:]
video_data = tongue['data'][:]
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies a multi-step processing pipeline: (1) Mark frames with likelihood >= 0.9 as visible. (2) Apply a 5-sigma velocity outlier cleanup: compute frame-to-frame y velocity among visible frames, identify outliers > 5 standard deviations, and interpolate those frames from neighboring clean visible frames. (3) Compute 40th and 60th percentiles from all cleaned visible raw frames across the session. (4) For each bin center, find the nearest video frame; if within 20ms and visible, classify using percentiles; otherwise class 3 (not visible).

ii.
```python
LIKELIHOOD_CUTOFF = 0.9
MAX_VIDEO_DT = 0.020

def clean_tongue_y(timestamps, data):
    visible = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_CUTOFF)
    # 5-sigma velocity cleanup
    velocity[valid_pair] = np.diff(y)[valid_pair] / dt[valid_pair]
    threshold = 5.0 * np.nanstd(values)
    bad_pair = valid_pair & (np.abs(velocity - center) > threshold)
    y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
    q40, q60 = np.percentile(y[final_visible], [40, 60])
    return y, likelihood, final_visible, outlier, np.array([q40, q60])
```

iii. The AI justifies the 0.9 likelihood threshold by noting the bimodal distribution of likelihoods. The 5-sigma velocity cleanup is described as matching the reference method paper's marker preprocessing. Percentiles are computed on individual visible frames (not bin means as in the reference).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Class 0: y < q40; Class 1: q40 <= y <= q60; Class 2: y > q60; Class 3: not visible (low likelihood, no nearby frame, or frame > 20ms away).

ii.
```python
tongue_class[sampled_visible & (sampled_y < percentiles[0])] = 0
tongue_class[sampled_visible & (sampled_y >= percentiles[0]) & (sampled_y <= percentiles[1])] = 1
tongue_class[sampled_visible & (sampled_y > percentiles[1])] = 2
```

iii. The boundaries differ slightly from the reference's `np.digitize` in the treatment of exact boundary values, but this is unlikely to matter in practice.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses **nearest-frame sampling**: for each bin center, the nearest video frame is found via `nearest_indices()`. If the nearest frame is within 20ms and visible, its value is used; otherwise the bin is class 3.

ii.
```python
target = centers_abs.ravel()
video_idx = nearest_indices(video_t, target)
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
sampled_visible = visible[video_idx] & close
sampled_y = clean_y[video_idx]
tongue_class = np.full(target.shape, 3, dtype=np.int64)
```

iii. The reference instead bins all frames within each 50ms window and takes the mean (using `_bin_mean`). The AI's nearest-frame approach uses a single frame per bin center rather than averaging all frames in the bin window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Session with no classifier-good units (all classifications NaN): dropped during inventory. (2) Population-wide zero-spike trials: excluded after neural binning (2,576 trials removed). (3) Tongue frames with low likelihood or no nearby frame: assigned class 3 (not visible). (4) Velocity outliers in tongue y: interpolated from neighboring clean frames.

ii.
```python
# Drop sessions with no good units
if not mask.any():
    print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
    continue

# Exclude zero-spike trials
trial_keep = np.any(rates != 0, axis=(1, 2))

# Tongue: velocity outlier interpolation
y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
```

iii. The AI's zero-spike trial exclusion is a post-hoc filter not present in the reference, which instead uses `obs_intervals` and `free_water` filtering to avoid such trials proactively. The AI does not filter `free_water` trials explicitly.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports full conversion in 164.37 s for 173 sessions (~0.95 s/session). A two-pass approach is used: the inventory prepass reads metadata without loading bulk spike data, then the processing pass does the heavy I/O. Neural spike binning and tongue processing are the main per-session costs.

ii. N/A (timing information from conversion logs)

iii. The AI documented timing estimates in Step 7 and found the full conversion well under the 15-minute threshold.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) The per-unit spike binning loop (one `searchsorted` per unit, but vectorized across trials). (2) The `build_photostim` loop over laser intervals. (3) The `trial_event_mapping` loop over trials. The tongue processing does not have a per-trial loop but instead uses vectorized nearest-frame lookup.

ii.
```python
# Per-unit loop (vectorized across trials)
for j, unit in enumerate(unit_indices):
    spikes = spike_data[starts[unit]:endpoints[unit]]
    positions = np.searchsorted(spikes, flat_edges, side='left')

# Photostim loop over intervals
for a, b in zip(starts, stops):
    state |= (centers_abs >= a) & (centers_abs < b)

# Trial event mapping loop
for i, (a, b) in enumerate(zip(starts, stops)):
    # searchsort for each trial
```

iii. The per-unit loop cannot be easily vectorized due to ragged spike arrays. The trial_event_mapping loop could potentially be vectorized. The photostim loop is over a small number of intervals.

## 10-c. What processing does the code repeat multiple times?

i. The `session_inventory` prepass reads each file twice (once in inventory, once in processing), re-reading `classification`, `is_good_trials`, and region data. The classifier mask and unit validity are computed in both passes.

ii.
```python
# In session_inventory:
class_good = classifier_mask(f)
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)

# In process_session (same computation repeated):
class_good = classifier_mask(f)
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
```

iii. The duplicate computation in the two-pass approach trades efficiency for cleaner code organization. The metadata arrays are small relative to spike data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The 5-sigma velocity cleanup on tongue y is additional processing not present in the reference. The interpolated outlier values affect the percentile computation and potentially the per-bin classifications. Whether this extra processing improves or changes downstream results is unclear, but it is motivated by the method paper's marker preprocessing description.

ii.
```python
# Velocity outlier detection and interpolation
velocity[valid_pair] = np.diff(y)[valid_pair] / dt[valid_pair]
threshold = 5.0 * np.nanstd(values)
bad_pair = valid_pair & (np.abs(velocity - center) > threshold)
y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
```

iii. The AI also computes and stores various per-session statistics (tongue outlier counts, per-class counts, etc.) in the `stats` dict within `session_info` metadata, which is not used by the decoder.
