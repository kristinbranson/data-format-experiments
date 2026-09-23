# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is a DANDI download (DANDI 000363, Mesoscale Activity Map) laid out as one NWB/HDF5 file per session under `/app/data/sub-<subject>/`. The AI enumerates every session with a single sorted glob over that layout and reads the files **directly with `h5py`** rather than with `pynwb`, addressing raw HDF5 paths (`units/spike_times`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `general/extracellular_ephys/electrodes/location`).

Loading happens in **two passes over every file**: a metadata "inventory" prepass (`session_inventory`) that opens each of the 174 files to decide which sessions are usable and to build the global brain-region vocabulary, followed by the real conversion pass (`process_session`) that re-opens each kept file and reads the bulk arrays.

ii. Enumerating and the prepass:
```python
DATA_ROOT = '/app/data'
...
paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
inventory, brain_regions = session_inventory(paths)
print(f'Inventory: {len(paths)} source files, {len(inventory)} usable sessions, regions={brain_regions}', flush=True)
```

```python
def session_inventory(paths):
    """Small metadata prepass: reject sessions without classifier-good units."""
    kept, all_regions = [], set()
    for path in paths:
        with h5py.File(path, 'r') as f:
            class_good = classifier_mask(f)
            n_trials = f['units/is_good_trials'].shape[1]
            always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
            mask = class_good & always_valid
            if not mask.any():
                print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
                continue
            ...
            regions = electrode_regions(f, mask)
            all_regions.update(regions)
            kept.append((path, int(mask.sum()), n_trials))
    return kept, sorted(all_regions)
```

The conversion pass:
```python
for i, (path, expected_units, expected_trials) in enumerate(inventory):
    n, x, y, subject, ridx, stats = process_session(path, region_lookup, ...)
```

```python
def process_session(path, region_lookup, make_plot=False):
    tic = time.time()
    with h5py.File(path, 'r') as f:
        ...
```

iii. From CONVERSION_NOTES.md Step 2 and Step 10 Check 3: "`/app/data` is 50 GB and contains 174 NWB/HDF5 session files plus one DANDI metadata YAML… one directory per subject and one NWB per session." The reference MATLAB pipeline loaded DataJoint-exported `.mat` probe files, which do not exist here, so the AI reads the NWB equivalents instead: "Data loading | Direct HDF5/NWB datasets | Recursive MATLAB `loadmat` export loader | Same underlying MAP streams; direct NWB avoids export ambiguity." The prepass is justified in Step 7 as a speed-up: "Metadata-only inventory prepass | Detects exclusions/regions without loading bulk spikes", i.e. it establishes the global region vocabulary and session exclusions before any bulk spike I/O.

---

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is taken from the **parent directory name**, `sub-<id>`, with the `sub-` prefix stripped (e.g. `440956`). The set of subjects is the sorted unique set over the *kept* sessions, and `subject_idx` is each session's index into that list. The AI does not read `nwb.subject.subject_id`; it relies on the DANDI directory convention, which is derived from that field. The result is 28 subjects with 3–10 sessions each.

ii.
```python
subject = os.path.basename(os.path.dirname(path)).removeprefix('sub-')
```

```python
subjects = sorted({os.path.basename(os.path.dirname(x[0])).removeprefix('sub-') for x in inventory})
subject_lookup = {x: i for i, x in enumerate(subjects)}
...
subject_idx.append(subject_lookup[subject])
```

```python
'subjects': subjects,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "parent `sub-*` directory | `subjects`, `subject_idx` | Unique subject IDs and per-session index | filename parsing analogous to `process_one_sess` | 28 subjects expected." The reference code base itself parses subject/date/session/probe from filenames ("Subject/date/session/probe are parsed from filenames", Step 1 notes), so filename-derived subject IDs are the analogous operation. Step 9 records the check: papers say 28 mice, the data give 28, the conversion gives 28.

---

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is required. Session order is the sorted glob order, which (because the filename embeds `ses-<YYYYMMDD>T<HHMMSS>`) is chronological within each subject. A session identifier is derived by truncating the filename at `_behavior`, giving e.g. `sub-440956_ses-20190207T120657`. Sessions with no usable units are dropped at the inventory stage (see 2-c), leaving 173 of 174.

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
```

```python
session_id = os.path.basename(path).split('_behavior')[0]
```

```python
stats = {
    'session_id': session_id, 'subject': subject, 'n_trials': n_trials,
    'n_neurons': len(unit_indices), ...
}
...
'session_info': session_info,
```

iii. Step 2: "Files are organized as `/app/data/sub-<subject>/sub-<subject>_ses-<timestamp>_behavior+ecephys+ogen.nwb`: one directory per subject and one NWB per session." Step 4/Step 9: "Exclude `sub-440958_ses-20190216T162508`; it has no classifier-curated units. Remaining count exactly matches 173" — the paper's reported 173 behavioral sessions is used as the consistency check.

---

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` DynamicTable, one row per behavioural trial, with `start_time`/`stop_time` giving each trial's interval. Rather than assuming positional correspondence between the trial table and the event streams, the AI **maps events into trial intervals by timestamp**: `trial_event_mapping` requires exactly one `go_start_times` event inside `[start_time, stop_time]` for each trial and uses that as the trial's go cue. The same helper (with `use_last=True`) finds the tone.

Crucially, the AI does **not** use all trial-table rows. It sets `n_trials = f['units/is_good_trials'].shape[1]` and takes the **first** `n_trials` rows of the trial table, on the assumption that the recorded trials are always a leading prefix of the behavioural trials (see 1-e).

ii.
```python
n_trials = f['units/is_good_trials'].shape[1]
...
tr = f['intervals/trials']
trial_starts = tr['start_time'][:n_trials]
trial_stops = tr['stop_time'][:n_trials]
events = f['acquisition/BehavioralEvents']
go_events = events['go_start_times/timestamps'][:]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
```

```python
def trial_event_mapping(starts, stops, event_times, name, before=None, use_last=False):
    """Map events into trial intervals, optionally requiring times before another event."""
    result = np.empty(len(starts), dtype=np.float64)
    for i, (a, b) in enumerate(zip(starts, stops)):
        hi = b if before is None else min(b, before[i])
        lo_idx = np.searchsorted(event_times, a, side='left')
        hi_idx = np.searchsorted(event_times, hi, side='left' if before is not None else 'right')
        candidates = event_times[lo_idx:hi_idx]
        if len(candidates) == 0:
            raise ValueError(f'trial {i}: no {name} event in [{a}, {hi}]')
        if not use_last and len(candidates) != 1:
            raise ValueError(f'trial {i}: expected one {name}, found {len(candidates)}')
        result[i] = candidates[-1] if use_last else candidates[0]
    return result
```

iii. Step 2/Step 4: "Event series lengths may differ from trial count (e.g. sample/delay events), so trials must be aligned using timestamps and trial intervals rather than positional assumptions." Step 4 resolution for the go cue: "Find the unique `go_start_times/timestamps` value inside each trial interval and subtract it for binning/alignment." Step 10 Check 5: "Exactly one go event and at least one sample event map every represented trial."

---

## 1-e. How are trials filtered based on quality controls?

i. Two filters, applied in sequence, plus a deliberate decision **not** to apply the papers' behavioural exclusions.

1. **Recorded-trial prefix.** Nine sessions have fewer `is_good_trials` columns (and `obs_intervals` rows) than trial-table rows. The AI interprets the shortfall as "the ephys covers a *prefix* of the behavioural trials" and keeps only trial rows `[:n_trials]`, discarding the rest.
2. **Population-wide zero-spike trials.** After binning, any trial in which **every** curated unit fires zero spikes across the whole 4 s window is dropped, and the identical mask is applied to every other trial-aligned stream. This removed 2,576 of 93,310 trials (2.76%) across 94 sessions.
3. A session is rejected if fewer than 2 trials survive (raised as an error; never triggered — the minimum is 159).

Early-lick, `ignore`, `miss`, photostimulation, auto-water and free-water trials are all **kept**, contrary to the papers' analysis exclusions, because they are required decoder targets/inputs. Final: 90,734 trials.

ii. Prefix restriction:
```python
class_good = classifier_mask(f)
n_trials = f['units/is_good_trials'].shape[1]
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
unit_indices = np.flatnonzero(mask)
if n_trials < 2 or not f['units/is_good_trials'][mask, :].all():
    raise ValueError(f'{path}: invalid represented-trial matrix')

tr = f['intervals/trials']
trial_starts = tr['start_time'][:n_trials]
trial_stops = tr['stop_time'][:n_trials]
```

Explicit rejection of clipping to `obs_intervals`:
```python
# `is_good_trials` is the source validity mask. NWB obs_intervals rows
# mirror behavioral trial start/stop boundaries rather than continuous
# ephys availability; the required go-centered window may legitimately
# extend into adjacent ITI, so it must not be clipped to those rows.
```

Zero-spike filter:
```python
rates = bin_selected_units(f, unit_indices, edges_abs)
# Some source trials contain physiologically impossible population-wide
# raw-spike gaps despite true is_good_trials flags. Exclude these invalid
# data periods and apply the identical mask to every aligned stream.
trial_keep = np.any(rates != 0, axis=(1, 2))
trial_indices = np.flatnonzero(trial_keep)
n_zero = int((~trial_keep).sum())
if n_zero:
    print(f"  excluding {n_zero} population-wide zero-spike trials", flush=True)
rates = rates[trial_keep]
trial_starts = trial_starts[trial_keep]
trial_stops = trial_stops[trial_keep]
go = go[trial_keep]
tone = tone[trial_keep]
edges_abs = edges_abs[trial_keep]
centers_abs = centers_abs[trial_keep]
n_trials = len(trial_indices)
if n_trials < 2:
    raise ValueError(f'{path}: fewer than two valid nonzero neural trials')
```

iii. Step 4 (partial recordings): "`obs_intervals` and `is_good_trials` represent a prefix of recorded trials. All curated units in each session share the same count and all represented entries are true. Restrict each session to that prefix; never fabricate neural zeros for later behavioral-only trials." Step 37 of the trajectory records the supporting check: "All classifier-good units within a session have identical observation-interval lengths, and their `is_good_trials` entries are all true. Thus each usable session has a well-defined prefix of simultaneously recorded trials."

Step 10 Check 1 justifies the zero-spike filter: "Raw NWB checks proved these were exact population-wide source spike gaps, scattered through otherwise continuously covered sessions. A four-second absence of all spikes from 90–923 units is physiologically implausible, so these are invalid acquisition periods not marked by `is_good_trials`."

Step 3/Step 4 justify keeping behaviourally "bad" trials: "This decoder explicitly requires photostimulation input and early-lick, ignore/miss/hit, and no-lick output categories, so excluding those trials would destroy required targets."

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `units/spike_times` (ragged, session-absolute seconds) indexed via `units/spike_times_index`, restricted to the curated unit rows (`units/classification == 'good'` **and** `units/is_good_trials` true on every represented trial). The go-cue timestamps from `acquisition/BehavioralEvents/go_start_times/timestamps` supply the bin-edge placement. No other neural representation is used.

ii.
```python
def bin_selected_units(f, unit_indices, absolute_edges):
    """Vectorized ragged-spike binning, returning trials x neurons x time in Hz."""
    spike_data = f['units/spike_times']
    endpoints = f['units/spike_times_index'][:]
    starts = np.r_[0, endpoints[:-1]]
    ...
    for j, unit in enumerate(unit_indices):
        spikes = spike_data[starts[unit]:endpoints[unit]]
```

```python
go_events = events['go_start_times/timestamps'][:]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
edges_abs = go[:, None] + EDGES_REL[None, :]
```

iii. Step 5 mapping table: "`units/spike_times` ragged arrays | `neural` | Select `classification == good`; histogram absolute spikes into 80 half-open 50-ms bins at go + `[-2.5,1.5)`; divide counts by 0.05 for Hz; transpose to neurons x time | `sliding_histogram`, `process_one_area`". Step 2 confirms this is electrophysiology only: "Native data are electrophysiology; no calcium imaging/delta-F/F is involved."

---

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin **firing rates in Hz**. For each curated unit, the absolute bin edges of all trials are flattened into one globally ordered vector, a single `np.searchsorted` gives the running spike count at each edge, and `np.diff` along the bin axis yields the spike count per bin. Counts are divided by the 0.05 s bin width and stored as `float32`. Bins are half-open `[left, right)`. No smoothing, no normalisation, no baseline subtraction, no z-scoring.

ii.
```python
def bin_selected_units(f, unit_indices, absolute_edges):
    """Vectorized ragged-spike binning, returning trials x neurons x time in Hz."""
    spike_data = f['units/spike_times']
    endpoints = f['units/spike_times_index'][:]
    starts = np.r_[0, endpoints[:-1]]
    n_trials = absolute_edges.shape[0]
    rates = np.empty((n_trials, len(unit_indices), N_TIME), dtype=np.float32)
    flat_edges = absolute_edges.ravel()
    # Trial edges are globally time-ordered; search all trial edges in one C call/unit.
    for j, unit in enumerate(unit_indices):
        spikes = spike_data[starts[unit]:endpoints[unit]]
        positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
        rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
    return rates
```

iii. Step 1: "Reference `sliding_histogram` uses `[left,right)` bins and reports spikes/s." Step 10 Check 3: "Binning | 80 contiguous `[left,right)` 50-ms bins, rates in Hz | `sliding_histogram`, half-open 40-ms windows/3.4-ms stride | Same histogram/rate logic; task-required geometry overrides reference." Step 4: "Explicit task overrides reference bin geometry. Count spikes in 80 contiguous half-open 50-ms bins over [-2.5,1.5), divide by 0.05 s." Step 10 Check 2 verifies the result against an independent `np.histogram` reimplementation from raw NWB via `np.allclose`.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level criteria, ANDed:
1. `units/classification == 'good'` — the region-specific logistic-regression QC classifier of Chen, Liu et al. (2023). No individual metric thresholds are applied, and `units/unit_quality` (the Kilosort label) is explicitly **not** used.
2. `units/is_good_trials` true for **every** represented trial — i.e. the unit must be valid across the whole session, so the neuron population is fixed within a session.

Criterion 1 alone gives 69,453 units (paper: 69,943); criterion 2 removes a further 565 (0.81%), leaving **68,888** units, mean 398.2/session (min 90, max 923). A session with no surviving unit is dropped at the inventory stage — exactly one, `sub-440958_ses-20190216T162508`, whose `classification` is `nan` for all 1,852 units — leaving 173 sessions.

ii.
```python
def decode_array(x):
    """Decode an HDF5 string vector to a NumPy unicode array."""
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in x])


def classifier_mask(f):
    return decode_array(f['units/classification'][:]) == 'good'
```

```python
class_good = classifier_mask(f)
n_trials = f['units/is_good_trials'].shape[1]
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
unit_indices = np.flatnonzero(mask)
```

```python
if not mask.any():
    print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
    continue
```

iii. Step 3 curation rules: "Kilosort2 clusters were manually labeled on 28 penetrations across five major areas, and 15 quality metrics trained five region-specific logistic-regression classifiers… Use classifier-derived `good` units, not Kilosort `unit_quality` alone. NWB contains the classifier result in `units/classification`." Step 4 discrepancy table: "NWB has both Kilosort `unit_quality` (154,948 good) and final `classification` (69,453 good) | 69,943 classifier-good units | Use `classification == good`; this closely reproduces paper QC, unlike `unit_quality`."

The extra `always_valid` requirement is justified in Step 4: "Four sessions contain some classifier-good units with false `is_good_trials` entries; because target matrices require one fixed population per session, additionally require each retained unit to be valid on every represented trial. This removes 565 units (0.81%) while preserving all represented trials" — preferred over "varying neuron dimensions or fabricating zeros" (Step 6).

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go cue onset**. All NWB streams share one session-absolute clock, so no resampling or offset correction is needed. For each trial the unique `go_start_times` event falling inside `[trial_start, trial_stop]` is located, and the fixed relative edge grid `EDGES_REL` (-2.5 … +1.5 s) is added to it to give the trial's absolute bin edges; spikes are binned directly against those absolute edges.

ii.
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
```

```python
go_events = events['go_start_times/timestamps'][:]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
...
edges_abs = go[:, None] + EDGES_REL[None, :]
centers_abs = go[:, None] + CENTERS_REL[None, :]
```

```python
rates = bin_selected_units(f, unit_indices, edges_abs)
```

and the metadata that records it:
```python
'temporal_alignment_event': 'go cue onset',
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. Step 4: "NWB spikes/events are absolute; every trial contains exactly one go-start event… Find the unique `go_start_times/timestamps` value inside each trial interval and subtract it for binning/alignment." Step 5: "All streams use absolute NWB timestamps before subtraction/alignment, preventing clock-offset mistakes. The unique go event inside each trial is the alignment timestamp." Step 10 Check 3: "Alignment | unique absolute go event per trial | exported spikes already go-relative; lick/laser times subtract go | Equivalent alignment in source clock." The `--show-processing` plots draw a dashed line at t=0 to demonstrate the alignment visually.

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **50 ms**, 80 contiguous non-overlapping half-open bins spanning -2.5 s to +1.5 s relative to the go cue; bin centres -2.475 s … +1.475 s. The grid is defined once at module level and reused for every trial and every session, so every trial has exactly 80 timepoints. There is **no rebinning or resampling** of an intermediate representation: spikes are histogrammed once directly at the target resolution. The reference papers' 40 ms window / 3.4 ms stride sliding histogram is deliberately *not* reproduced, because the decoder task specifies the bin geometry. All non-neural streams (tone time, photostim, tongue) are placed on the same 80-bin grid.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
```

```python
rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
```

```python
'time_bin_size': 50.0,
'time_bin_units': 'ms',
'n_timepoints': N_TIME,
'time_bin_edges_seconds': EDGES_REL.astype(np.float32),
'time_bin_centers_seconds': CENTERS_REL.astype(np.float32),
```

iii. Step 4: "Firing-rate bins | Reference uses 40-ms width/3.4-ms stride | Decoder requires 50-ms bins | Explicit task overrides reference bin geometry." Step 5: "Window edges: `np.arange(-2.5, 1.5 + 0.05, 0.05)` (81 edges); centers are -2.475 through +1.475 s (80 points). Spike bins are left-closed/right-open." Step 10 Check 5: "Half-open bin edges avoid double counting. All sessions have exactly 80 bins." The verifier confirms `T: mean 80.00, min 80, max 80`.

---

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch/tone onsets) together with each trial's go cue. Because a lick during the sample or delay epoch replays that epoch, a trial can contain several `sample_start_times` events; the AI takes the **last one strictly before the go cue** and within the trial interval.

ii.
```python
sample_events = events['sample_start_times/timestamps'][:]
go = trial_event_mapping(trial_starts, trial_stops, go_events, 'go cue')
tone = trial_event_mapping(trial_starts, trial_stops, sample_events,
                           'sample/tone onset', before=go, use_last=True)
```
with the helper enforcing "inside the trial and before the go cue, take the last":
```python
hi = b if before is None else min(b, before[i])
lo_idx = np.searchsorted(event_times, a, side='left')
hi_idx = np.searchsorted(event_times, hi, side='left' if before is not None else 'right')
candidates = event_times[lo_idx:hi_idx]
...
result[i] = candidates[-1] if use_last else candidates[0]
```

iii. Step 4: "Some trials contain repeated sample-start events; every trial has at least one before go… Use the final sample-start event before the unique go cue. Common tone-to-go durations (1.85, 0.95, 2.45 s) support variable protocol timing. Preserve measured timing rather than assume a constant offset." Trajectory step 36: "the correct tone/sample onset is the last sample-start event before go (normally exactly 1.85 s before go), not the first event."

---

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For every bin the value is the absolute bin-centre time minus the trial's tone onset — i.e. a continuous, monotonically increasing ramp per trial with slope 0.05 s/bin, offset by the measured tone-to-go interval. Stored as `float32`, stacked as input channel 0. No clipping, rectification or normalisation; values are negative for bins that precede the tone. Observed range across the dataset: **[-1.525, 11.894] s**.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
...
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
...
inputs = np.stack([tone_elapsed, photo], axis=1)  # trials x 2 x time
```
```python
'input_names': ['time from tone onset (s)', 'photostimulation on'],
```

iii. Step 5 mapping: "final `sample_start_times/timestamps` before go | `input[0,:]` | For each bin center, seconds elapsed since tone onset: `(go + bin_center) - tone_onset` | Continuous, time-varying; values may be negative before tone."

The unusually long positive values were investigated rather than assumed to be a bug (Step 7): "Values >2.5 s occur overwhelmingly on early-lick trials (4,595 cases dataset-wide), where repeated delay-state transitions postpone go cue; raw sample onset is unique and unchanged. Thus these values are valid rather than alignment errors." Trajectory step 55 quantifies this: "4,595 of 4,724 intervals above 2.5 s are early-lick trials whose state machine delays go cue while tone onset remains unchanged."

---

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is constructed **from** the neural bin grid, so alignment is exact by construction: `centers_abs` is the same `go + CENTERS_REL` array that generated the spike bin edges (`edges_abs = go + EDGES_REL`). Bin *k* of the input therefore refers to the centre of exactly the same interval as bin *k* of the firing rates. No interpolation or resampling is involved.

ii.
```python
edges_abs = go[:, None] + EDGES_REL[None, :]
centers_abs = go[:, None] + CENTERS_REL[None, :]
```
```python
rates = bin_selected_units(f, unit_indices, edges_abs)
...
tone_elapsed = (centers_abs - tone[:, None]).astype(np.float32)
```
and the trial mask is applied identically to both streams:
```python
rates = rates[trial_keep]
...
go = go[trial_keep]
tone = tone[trial_keep]
edges_abs = edges_abs[trial_keep]
centers_abs = centers_abs[trial_keep]
```

iii. Step 5: "All streams use absolute NWB timestamps before subtraction/alignment, preventing clock-offset mistakes." Step 10 Check 2 verifies the input independently: "independently selected the unique go event, latest pre-go sample onset, and laser intervals; both time-from-tone and laser state matched via `np.allclose`." The `--show-processing` plot overlays the time-from-tone trace on the same t=0 go-cue axis as the rate heatmap.

---

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the **event streams** `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`, which give the absolute on/off times of every laser epoch in the session. The AI explicitly cross-checked these against the trials-table fields `photostim_onset` / `photostim_duration` (strings, `'N/A'` on control trials) and found them equivalent, then used the event streams because they are already absolute and need no string parsing.

ii.
```python
laser_starts = events['photostim_start_times/timestamps'][:]
laser_stops = events['photostim_stop_times/timestamps'][:]
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```

iii. Step 5 mapping: "`photostim_start_times` / `photostim_stop_times` | `input[1,:]` | 1 when bin center lies in any laser interval (`start <= t < stop`), else 0 | stimulation alignment in `process_one_sess` | Exact event intervals; trial table/event status agrees for all usable trials." Trajectory step 40: "Trial-table photostimulation status matches event start/stop streams for all 93,310 usable trials, so exact event intervals can construct the time-varying on/off input."

---

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A **binary time series**, not a per-trial flag: a bin is 1 if its centre falls in any laser interval `[start, stop)`, else 0, cast to `float32` and stacked as input channel 1. The state is ORed over *all* laser intervals of the session rather than only the trial's own, so a neighbouring trial's stimulation would also be registered if it fell inside the -2.5/+1.5 s window. Overall bin occupancy is 2.43%, and the per-session range is `[0, 1]` (a few sessions have no stimulation and are all-zero).

ii.
```python
def build_photostim(centers_abs, starts, stops):
    state = np.zeros(centers_abs.shape, dtype=bool)
    for a, b in zip(starts, stops):
        state |= (centers_abs >= a) & (centers_abs < b)
    return state.astype(np.float32)
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)" and "If an input is a time such as onset of some stimulus, represent it as a binary time series." Step 5: "Photostimulation state is sampled at bin centers from exact event start/stop intervals." Step 3 notes the biological context that makes near-go bins mostly zero: "We silenced ALM activity during the late delay epoch (last 0.5 s)… photoinhibition always ended before the 'Go' cue." Step 6 records a bug caught in review: "A photostimulation boolean-accumulator type issue found during code review was fixed before sample execution and covered by a direct self-check" (the `|=` accumulator had to be `bool`, not float).

---

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Via the same `centers_abs` array used for the neural bin edges — the laser on/off times are absolute NWB timestamps on the identical session clock, so membership is tested directly against the go-cue-relative bin centres. The same `trial_keep` mask is then applied. No interpolation or per-stream offset.

ii.
```python
centers_abs = go[:, None] + CENTERS_REL[None, :]
...
photo = build_photostim(centers_abs, laser_starts, laser_stops)
```
```python
state |= (centers_abs >= a) & (centers_abs < b)
```
```python
centers_abs = centers_abs[trial_keep]
```

iii. Step 5: "All streams use absolute NWB timestamps before subtraction/alignment." Step 10 Check 2: the laser state was independently reconstructed from raw NWB event timestamps and "matched via `np.allclose`." The `--show-processing` plot draws the photostim step function on the same go-cue-relative axis (`ax[1].step(CENTERS_REL, inputs[0, 1], where='mid', label='photostim on')`).

---

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the NWB trials table, so choice is **derived** from two trial-table string columns: `intervals/trials/trial_instruction` (`'left'` / `'right'`, the tone-instructed side) and `intervals/trials/outcome` (`'hit'` / `'miss'` / `'ignore'`). A hit means the animal licked the instructed side, a miss means it licked the other side, an ignore means it never licked.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
instruction = decode_array(tr['trial_instruction'][:])[trial_indices]
```

iii. Step 4: "MATLAB comments conflict on lick direction coding | NWB has instruction and outcome strings | Hit means instructed lick, miss means incorrect lick, ignore means no response | Derive choice: `ignore -> no lick`, `hit -> instruction`, `miss -> opposite instruction`; spot-check later against post-go left/right lick timestamps." Step 1 flagged the ambiguity in the reference code that motivated deriving rather than copying: "Lick-direction comments conflict in two locations (one says 0 left/1 right, another says 0 right/1 left), so raw data semantics and NWB documentation must resolve this."

The spot-check was carried out (Step 10 Check 2): "derived choice agreed with first raw post-go lick direction on 69,245/69,424 (99.742%) responsive non-early trials. Rare mismatches involve multiple/near-simultaneous events and do not invalidate authoritative trial outcome/instruction labels."

---

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, via an explicit per-trial loop over `(outcome, instruction)`. Because the output array must be a single rectangular `(4, 80)` per trial (to co-exist with the time-varying tongue channel), the per-trial value is **tiled across all 80 bins**. Written into row 0; `output_values[0] = ['left', 'right', 'no lick']`. Dataset fractions: left 0.428, right 0.422, no lick 0.149.

ii.
```python
choice = np.empty(n_trials, dtype=np.int64)
for i, (o, side) in enumerate(zip(outcome_str, instruction)):
    if o == 'ignore': choice[i] = 2
    elif o == 'hit': choice[i] = 0 if side == 'left' else 1
    else: choice[i] = 1 if side == 'left' else 0
```

```python
outputs = np.empty((n_trials, 4, N_TIME), dtype=np.int64)
outputs[:, 0, :] = choice[:, None]
```

```python
'output_names': ['lick direction choice', 'outcome', 'early lick', 'tongue y-position'],
'output_values': [
    ['left', 'right', 'no lick'],
    ...
],
```

iii. Step 5: "`trial_instruction` + `outcome` | `output[0,:]` choice | ignore -> no lick; hit -> instructed side; miss -> opposite side; tile over 80 bins | Classes: left=0, right=1, no lick=2." The tiling decision is justified in Step 5 item 6: "Use `(4,80)` for every trial; tile the three per-trial categories so they can coexist with time-varying tongue in one rectangular array", and in trajectory step 43: "The target output must be one rectangular array per trial. Because tongue position is time-varying, choice, outcome, and early-lick labels must be tiled across all 80 bins."

---

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` string column, which already holds exactly the three categories the instructions ask for: `'ignore'`, `'miss'`, `'hit'`. No derivation.

ii.
```python
outcome_str = decode_array(tr['outcome'][:])[trial_indices]
```

iii. Step 2: "`intervals/trials` is a DynamicTable. Every session has … `outcome` (ignore/miss/hit)". Step 2 Data Quality: "Direct trial-table fields already use exactly the requested outcome categories and early-lick labels." Source distribution recorded in Step 2: "hit 65,254 (68.70%), miss 15,641 (16.47%), ignore 14,095 (14.84%)".

---

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0 = ignore`, `1 = miss`, `2 = hit`, matching the order given in the instructions. The per-trial code is tiled across all 80 bins into row 1 of the output array. Final fractions: ignore 0.149, miss 0.166, hit 0.684 — consistent with the source table after trial filtering.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_str], dtype=np.int64)
```
```python
outputs[:, 1, :] = outcome[:, None]
```
```python
['ignore', 'miss', 'hit'],
```

iii. Step 5 mapping: "`outcome` | `output[1,:]` | Direct categorical mapping, tiled over 80 bins | correctness/report loading | Classes: ignore=0, miss=1, hit=2." The dictionary lookup will raise a `KeyError` on any unexpected string, which the AI treats as a deliberate fail-fast guard. Step 10 Check 2 confirms: "independently mapped raw outcome/instruction/early strings and matched all 80 tiled values via `np.allclose`."

---

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `intervals/trials/early_lick` string column, which holds `'no early'` / `'early'`. No derivation.

ii.
```python
early_str = decode_array(tr['early_lick'][:])[trial_indices]
```

iii. Step 2: trials table has "`early_lick` (early/no early)"; source counts "early 10,805 (11.37%); no early 84,185 (88.63%)". Step 3 notes the papers exclude these trials — "The data paper likewise excluded early-lick and no-response trials for behavioral analyses" — but Step 4 overrides that for this task because early lick is a required decoder output.

---

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early' -> 0`, `'early' -> 1`, and the per-trial code is tiled across all 80 bins into row 2 of the output array. Final fractions: no 0.884, yes 0.116. No further processing; the flag is not re-derived from lick timestamps.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int64)
```
```python
outputs[:, 2, :] = early[:, None]
```
```python
['no', 'yes'],
```

iii. Step 5 mapping: "`early_lick` | `output[2,:]` | Direct categorical mapping, tiled over 80 bins | early-report loading | Classes: no=0, yes=1." The choice to trust the table rather than re-derive is implicit in Step 2's "Direct trial-table fields already use exactly the requested outcome categories and early-lick labels." Note the early-lick event itself falls before the go cue and therefore inside the -2.5 s window, which is why the decoder can reach 0.75 balanced accuracy on it (Step 11).

---

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: a `(n_frames, 3)` DeepLabCut array whose columns are `(tongue_x, tongue_y, tongue_likelihood)` — confirmed from the series' own `description` attribute rather than assumed — with matching absolute `timestamps`. Column 1 (`tongue_y`) is the value; column 2 (`likelihood`) determines visibility. Column 0 (`tongue_x`) is read but not used. Present in all 174 sessions at ~300 Hz.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
video_t = tongue['timestamps'][:]
video_data = tongue['data'][:]
clean_y, likelihood, visible, outliers, percentiles = clean_tongue_y(video_t, video_data)
```
```python
def clean_tongue_y(timestamps, data):
    """Apply reference five-sigma velocity cleanup to high-confidence tongue y."""
    y = np.asarray(data[:, 1], dtype=np.float64).copy()
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
```

iii. Step 2: "`acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` exists in all 174 sessions. Data shape is `(n_video_frames,3)` with description `(tongue_x, tongue_y, tongue_likelihood)` and matching absolute timestamps; the representative session had 680,500 frames spanning 0 to 2493.305 s." Step 1 notes why this had to come from NWB and not the reference repo: "No tongue-position preprocessing was present in the inspected ephys MATLAB pipeline; tongue/video data must be mapped from the provided NWB data."

---

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. **Visibility** — a frame counts as visible only if `y` and `likelihood` are finite **and** `likelihood >= 0.9`. Low-likelihood frames still carry a finite coordinate (the tracker reports a position even when the tongue is retracted), so they must be excluded explicitly. This yields ~11.7% visible frames.
2. **Five-sigma velocity cleanup** (taken from the method paper) — frame-to-frame `dy/dt` is computed among contiguous visible frames; frames whose velocity deviates from the mean by more than 5 SD are flagged as the destination of an implausible jump and their `y` is linearly re-interpolated from the remaining clean visible frames. Outlier frames stay classified as *visible*.
3. **Session percentiles** — the 40th and 60th percentiles of `y` over **all visible frames of the session** (not over bin means) become the two class edges.
4. **Per-bin sampling** — each of the 80 bin centres takes the value of the single **nearest** video frame (see 8-d), and that frame's value/visibility is digitised.

Note that the quantity discretised is a *point sample at the bin centre*, and the percentiles are taken over the same per-frame quantity, so the two are self-consistent; but ~93% of the 300 Hz frames within each 50 ms bin are discarded rather than aggregated.

ii.
```python
LIKELIHOOD_CUTOFF = 0.9
MAX_VIDEO_DT = 0.020
```
```python
def clean_tongue_y(timestamps, data):
    """Apply reference five-sigma velocity cleanup to high-confidence tongue y."""
    y = np.asarray(data[:, 1], dtype=np.float64).copy()
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    visible = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_CUTOFF)
    pair = visible[1:] & visible[:-1]
    dt = np.diff(timestamps)
    velocity = np.full(len(y) - 1, np.nan)
    valid_pair = pair & np.isfinite(dt) & (dt > 0)
    velocity[valid_pair] = np.diff(y)[valid_pair] / dt[valid_pair]
    values = velocity[valid_pair]
    outlier = np.zeros(len(y), dtype=bool)
    if values.size:
        center = np.nanmean(values)
        threshold = 5.0 * np.nanstd(values)
        bad_pair = valid_pair & (np.abs(velocity - center) > threshold)
        # Mark the destination frame of each implausible jump. Iterative artifacts
        # are avoided by interpolation from all remaining high-confidence frames.
        outlier[1:] = bad_pair
    clean_visible = visible & ~outlier
    if clean_visible.sum() < 2:
        raise ValueError('Fewer than two clean visible tongue samples')
    if outlier.any():
        y[outlier] = np.interp(timestamps[outlier], timestamps[clean_visible], y[clean_visible])
    # Outliers were reference-imputed and remain visible; low-confidence samples do not.
    final_visible = visible
    q40, q60 = np.percentile(y[final_visible], [40, 60])
    return y, likelihood, final_visible, outlier, np.array([q40, q60])
```

iii. Step 5: "Define raw visibility as finite y and DeepLabCut likelihood >=0.9. Likelihood is extremely bimodal (median session 75th percentile about 0.00006, 90th percentile about 0.999996); 0.9 is insensitive to modest threshold changes and yields 11.68% visible frames on average." Step 3: "Method-paper marker cleanup identifies velocity outliers above five sigma and imputes them from nearby frames. When tongue is occluded in the mouth (typically before response), its position was set to its mean." Step 4 resolves the conflict with the required class 3: "Task explicitly requires class 3 'not visible'; use likelihood-based visibility and retain class 3 rather than mean-imputing it away. Apply reference five-sigma velocity cleanup to visible y traces before percentiles." Step 5: "Compute q40/q60 from all cleaned visible y frames over that session, exactly matching 'over the session.'" Step 10 Check 2 re-derived the whole chain from raw NWB and matched via `np.allclose`.

---

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as specified in the instructions, using the per-session `q40`/`q60` edges:
- `0` — visible and `y < q40`
- `1` — visible and `q40 <= y <= q60`
- `2` — visible and `y > q60`
- `3` — not visible (likelihood below cutoff, non-finite, or no video frame within 20 ms of the bin centre)

The array is initialised to 3 and the three visible classes are written over it. Final dataset fractions: 0.062 / 0.032 / 0.065 / 0.841. (Note that class 1 is closed on both ends, so a value exactly equal to `q60` falls in class 1 rather than class 2 — a measure-zero difference on continuous pixel coordinates.)

ii.
```python
tongue_class = np.full(target.shape, 3, dtype=np.int64)
tongue_class[sampled_visible & (sampled_y < percentiles[0])] = 0
tongue_class[sampled_visible & (sampled_y >= percentiles[0]) & (sampled_y <= percentiles[1])] = 1
tongue_class[sampled_visible & (sampled_y > percentiles[1])] = 2
tongue_class = tongue_class.reshape(n_trials, N_TIME)
```
```python
['below 40th percentile', '40th to 60th percentile', 'above 60th percentile', 'not visible'],
```
```python
'tongue_discretization': 'Per-session 40th/60th percentiles of cleaned visible tongue y; low-confidence or missing samples are class 3.',
```
and the internal validator:
```python
assert set(np.unique(y[3])).issubset({0,1,2,3})
```

iii. Step 5: "Boundaries: class 0 for y < q40; class 1 for q40 <= y <= q60; class 2 for y > q60." A representative-session sanity check is recorded: "q40=273.94, q60=288.70; sampled fractions low/middle/high/not-visible = 4.60%/2.51%/5.52%/87.37%, plausible because most of the -2.5 to +1.5-s window precedes overt licking." Step 12 reports the dominance of class 3 is expected: "Tongue validation accuracy of 0.5990 across four classes, despite 84% invisible samples and balanced scoring, supports the likelihood/percentile mapping."

---

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and events, so the AI takes the 80 absolute bin centres of each trial (`centers_abs`, the same array that defines the neural bins), finds the **nearest video frame** to each with a two-sided `searchsorted`, and requires that frame to be within **20 ms** of the bin centre (measured typical distance <2 ms at 300 Hz). Bins failing the 20 ms tolerance — e.g. during inter-trial gaps where the trial-gated video is off — are assigned class 3. No interpolation or offset correction.

ii.
```python
def nearest_indices(source_t, target_t):
    """Indices of nearest sorted source timestamp for each target."""
    idx = np.searchsorted(source_t, target_t)
    idx = np.clip(idx, 1, len(source_t) - 1)
    choose_previous = np.abs(source_t[idx - 1] - target_t) < np.abs(source_t[idx] - target_t)
    idx[choose_previous] -= 1
    return idx
```
```python
target = centers_abs.ravel()
video_idx = nearest_indices(video_t, target)
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
sampled_visible = visible[video_idx] & close
sampled_y = clean_y[video_idx]
```

iii. Step 5: "Tongue y uses the nearest 300-Hz video frame to each center; require nearest-frame distance <=20 ms (normal distance is <2 ms), otherwise class 3." Trajectory step 43: "Nearest video frames are within 1.7 ms of 50-ms bin centers." Step 10 Check 5: "Nearest video samples require <=20 ms distance (normally <2 ms); missing/low-confidence frames map to class 3." The `--show-processing` plot overlays the raw y trace, the q40/q60 lines, the likelihood trace and the final class series on a common go-cue-relative axis so that misalignment would be visible.

---

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct cases, handled differently depending on whether the datum is *absent* or *legitimately undefined*:

1. **Session never quality-controlled** — `sub-440958_ses-20190216T162508` has `classification = nan` for all 1,852 units. `decode_array` stringifies non-bytes entries, so nothing equals `'good'`, the mask is empty and the session is skipped at inventory with a printed `SKIP` message.
2. **Units invalid on some trials** — 565 classifier-good units in four sessions have false `is_good_trials` cells; those units are dropped so the population is fixed per session (rather than varying the neuron dimension or inserting zeros).
3. **Behavioural trials with no ephys coverage** — sessions where `is_good_trials` has fewer columns than the trial table are truncated to the leading `n_trials` rows.
4. **Trials with no spikes at all** — dropped after binning, with the identical mask applied to every other trial-aligned stream so no stream can desynchronise.
5. **Missing / untracked video** — frames below the likelihood cutoff, non-finite frames, and bins with no video frame within 20 ms become the explicit class `3 = not visible`; five-sigma velocity outliers are re-interpolated rather than dropped.

Anything unexpected beyond these (an outcome string not in the map, a trial with zero or multiple go cues, a session left with <2 trials, fewer than two clean tongue samples) raises an exception and aborts the run — a deliberate fail-fast posture rather than silent coercion. An internal `validate_result` pass re-checks every array's shape, dtype, finiteness and value set before pickling.

ii.
```python
def decode_array(x):
    """Decode an HDF5 string vector to a NumPy unicode array."""
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in x])
```
```python
if not mask.any():
    print(f"SKIP {os.path.basename(path)}: no classifier-good units", flush=True)
    continue
```
```python
always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
mask = class_good & always_valid
```
```python
trial_keep = np.any(rates != 0, axis=(1, 2))
...
if n_trials < 2:
    raise ValueError(f'{path}: fewer than two valid nonzero neural trials')
```
```python
visible = np.isfinite(y) & np.isfinite(likelihood) & (likelihood >= LIKELIHOOD_CUTOFF)
...
close = np.abs(video_t[video_idx] - target) <= MAX_VIDEO_DT
sampled_visible = visible[video_idx] & close
tongue_class = np.full(target.shape, 3, dtype=np.int64)
```
```python
def validate_result(data):
    ...
    assert n.shape == (nn, N_TIME) and n.dtype == np.float32 and np.isfinite(n).all() and (n >= 0).all()
    assert x.shape == (2, N_TIME) and np.isfinite(x).all()
    assert y.shape == (4, N_TIME) and np.isfinite(y).all()
```

iii. Step 4: the all-`nan` session "has no classifier-curated units… Remaining count exactly matches 173." Step 6: the unit-validity filter "is preferable to varying neuron dimensions or fabricating zeros." Step 4 on partial recordings: "never fabricate neural zeros for later behavioral-only trials." Step 10 Check 5: "Every final neural trial has at least one source spike; no fabricated zeros or NaNs remain." The video handling follows the task requirement for an explicit fourth class rather than the method paper's mean imputation (Step 4).

---

## 10-a. What are the most time-consuming steps of the code?

i. The script prints per-step timing, and the full run (`conversion_full_out.txt`) gives the breakdown:

| Step | Time | Share |
|---|---|---|
| Spike reading + binning (`bin_selected_units`, summed) | 136.3 s | 77% |
| Rest of per-session processing (video, inputs, outputs) | 17.2 s | 10% |
| Inventory prepass (174 file opens) + 11.9 GB pickle write | ~22.9 s | 13% |
| **Total** | **176.4 s** | |

So neural binning dominates: it does one HDF5 read plus one `np.searchsorted` over ~40,000 edges per unit, for 68,888 units. Per-session cost scales with `n_units × n_trials` (0.17 s to 2.05 s). The second cost centre is I/O — the ~680k×3 tongue array and the ragged spike buffer per session, plus writing the 11.9 GB pickle. The whole conversion finishes in under 3 minutes, well inside the instructions' 15-minute budget, so no further optimisation was pursued.

ii.
```python
    for j, unit in enumerate(unit_indices):
        spikes = spike_data[starts[unit]:endpoints[unit]]
        positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
        rates[:, j, :] = np.diff(positions, axis=1).astype(np.float32) / BIN_SIZE
```
```python
    print(f"  neural binned: {rates.shape} in {time.time()-tic:.2f}s", flush=True)
...
    print(f"  completed {session_id}: {n_trials} trials, {len(unit_indices)} neurons, "
          f"{time.time()-tic:.2f}s", flush=True)
```
```python
    size_gb = os.path.getsize(args.outpicklefile) / 1e9
    print(f'Wrote {args.outpicklefile}: {size_gb:.3f} GB; total time {time.time()-total_tic:.2f}s', flush=True)
```

iii. Step 6: "Naive nested neuron x trial x bin loops would be prohibitive for 69,453 units and 93,310 trials. Repeated per-trial video scans and repeated HDF5 spike reads would add unnecessary overhead." Step 7 records the measured split — "Vectorized searchsorted spike binning | Neural binning only 0.23–0.39 s/session"; "Session processing | about 0.96 s mean for sample | about 3 minutes for 173 sessions" — and Step 9 the actual: "Full conversion runtime: 164.37 s, substantially faster than the conservative estimate."

---

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five Python-level loops remain, of which four are avoidable:

1. **`bin_selected_units` per-unit loop** — *not* removable. Spike storage is ragged, so there is no single sorted array to search; the trial dimension is already vectorised by flattening all 81×n_trials edges into one `searchsorted` call. This is the same structure the reference uses.
2. **`trial_event_mapping` per-trial loop** — avoidable. It performs two `np.searchsorted` calls *per trial* and is invoked twice per session (go cue, tone). Both could be a single vectorised `searchsorted` over the whole trial-start/stop arrays, exactly as the human reference does with `sample[np.searchsorted(sample, go, 'left') - 1]`.
3. **`build_photostim` loop over laser intervals** — avoidable and the most wasteful in asymptotics: it allocates and ORs a full `(n_trials, 80)` boolean array once per stimulation epoch, i.e. O(n_stim × n_trials × 80) — roughly 10^8 boolean operations for a 500-trial, 250-stim session. A single `np.searchsorted` of the bin centres into the interleaved start/stop array and a parity test would be O(n_trials × 80 × log n_stim).
4. **The `choice` loop** — avoidable; a two-level `np.where` on the outcome/instruction arrays (as in the reference) replaces it.
5. **`decode_array` / `electrode_regions` list comprehensions** — avoidable; `decode_array` builds a Python list over every string entry of columns with up to 272k rows, and `electrode_regions` runs `json.loads` over every electrode row of the session — and is called **twice** per session (inventory and processing).

ii. The unavoidable one:
```python
    for j, unit in enumerate(unit_indices):
        spikes = spike_data[starts[unit]:endpoints[unit]]
        positions = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, N_TIME + 1)
```

The avoidable ones:
```python
    for i, (a, b) in enumerate(zip(starts, stops)):
        hi = b if before is None else min(b, before[i])
        lo_idx = np.searchsorted(event_times, a, side='left')
        hi_idx = np.searchsorted(event_times, hi, side='left' if before is not None else 'right')
```
```python
def build_photostim(centers_abs, starts, stops):
    state = np.zeros(centers_abs.shape, dtype=bool)
    for a, b in zip(starts, stops):
        state |= (centers_abs >= a) & (centers_abs < b)
    return state.astype(np.float32)
```
```python
for i, (o, side) in enumerate(zip(outcome_str, instruction)):
    if o == 'ignore': choice[i] = 2
    elif o == 'hit': choice[i] = 0 if side == 'left' else 1
    else: choice[i] = 1 if side == 'left' else 0
```
```python
    for value in raw:
        value = value.decode() if isinstance(value, bytes) else str(value)
        try:
            labels.append(json.loads(value).get('brain_regions', value))
        except (json.JSONDecodeError, TypeError):
            labels.append(value)
```

iii. Step 6 claims the vectorisation that was in fact only partly achieved: "For each unit, one vectorized `np.searchsorted` against all globally ordered trial-bin edges computes all counts. Go/tone mappings, tongue nearest frames, and output arrays are computed session-wise with NumPy." The tongue sampling *is* fully vectorised (`nearest_indices` over all 80×n_trials centres at once — notably more vectorised than the reference, which loops over trials), but the go/tone mapping is a per-trial Python loop. Step 7 records the practical conclusion: total runtime ~3 minutes, "below 15-minute optimization threshold", so the remaining loops were not revisited.

---

## 10-c. What processing does the code repeat multiple times?

i. Three repetitions:

1. **Every NWB file is opened and partially processed twice.** `session_inventory` opens all 174 files and computes `classifier_mask` (decoding the full `classification` string column), reads the whole `is_good_trials` matrix, computes `always_valid`, and runs `electrode_regions` (JSON-parsing every electrode row). `process_session` then re-opens the file and recomputes **all four** of those quantities from scratch. Nothing from the prepass is reused except the file path; `expected_units` and `expected_trials` are used only in a progress `print`.
2. **`is_good_trials` is read three times per session** in `process_session`-equivalent code paths: once for `.shape[1]`, once for `np.all(...)`, and once more for the `f['units/is_good_trials'][mask, :].all()` assertion (a fancy-indexed read of the same matrix).
3. **`nearest_indices` is recomputed in the plotting path** for the same session already computed in the main path.

This contradicts the Step 6 claim that "HDF5 datasets are read once per required stream/session". The duplication costs ~20 s of the 176 s run and is not asymptotically important, but it is genuine repeated work that the human reference avoids entirely by deferring the global region/subject vocabulary to an assembly step after the single pass.

ii. The prepass:
```python
def session_inventory(paths):
    """Small metadata prepass: reject sessions without classifier-good units."""
    kept, all_regions = [], set()
    for path in paths:
        with h5py.File(path, 'r') as f:
            class_good = classifier_mask(f)
            n_trials = f['units/is_good_trials'].shape[1]
            always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
            mask = class_good & always_valid
            ...
            regions = electrode_regions(f, mask)
            all_regions.update(regions)
            kept.append((path, int(mask.sum()), n_trials))
    return kept, sorted(all_regions)
```

The identical work repeated in the conversion pass:
```python
def process_session(path, region_lookup, make_plot=False):
    tic = time.time()
    with h5py.File(path, 'r') as f:
        class_good = classifier_mask(f)
        n_trials = f['units/is_good_trials'].shape[1]
        always_valid = np.all(f['units/is_good_trials'][:, :], axis=1)
        mask = class_good & always_valid
        unit_indices = np.flatnonzero(mask)
        if n_trials < 2 or not f['units/is_good_trials'][mask, :].all():
            raise ValueError(f'{path}: invalid represented-trial matrix')
        ...
        labels = electrode_regions(f, mask)
```
and the inventory's outputs going unused:
```python
    for i, (path, expected_units, expected_trials) in enumerate(inventory):
        print(f'[{i+1}/{len(inventory)}] {os.path.basename(path)} expected={expected_trials}x{expected_units}', flush=True)
```

iii. Step 7 frames the prepass as a speed-up rather than duplication: "Metadata-only inventory prepass | Detects exclusions/regions without loading bulk spikes." The underlying need is real — `brain_regions` must be a single global sorted vocabulary shared by all sessions, and `region_lookup` is passed into `process_session`, so the vocabulary has to exist before the first session is converted. Step 6 nevertheless asserts "HDF5 datasets are read once per required stream/session", which the code does not honour for `classification`, `is_good_trials`, `electrodes/location`, or the per-unit `spike_times` slices.

---

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, all diagnostic or vestigial rather than expensive:

1. **The whole inventory prepass output** beyond the file list and region vocabulary: `expected_units` / `expected_trials` feed only a `print`, and `mask` / `always_valid` are recomputed.
2. **`trial_stops`** is read, sliced, and then filtered by `trial_keep` — but after `trial_event_mapping` has run it is never used again.
3. **`tongue_x`** (column 0 of the tracking array) is read into memory as part of the `(n_frames, 3)` slab and never used.
4. **`likelihood` and `outliers`** are returned from `clean_tongue_y` and threaded through `process_session`, but are consumed only by the `--show-processing` plot and by two `stats` fields — i.e. computed on all 173 sessions but used on at most 2.
5. **The five-sigma velocity cleanup itself** is run over every frame of every session, yet only alters the rare flagged frames, and those frames remain class-eligible either way; its effect on the q40/q60 edges is marginal.
6. **The per-session `stats` dictionary** computes five `np.bincount` calls and a `photo.sum()` on every session; these land in `metadata['session_info']` and are not used by the decoder.
7. **`nearest_indices` is recomputed inside `make_processing_plot`** for a quantity already available.
8. **Output arrays are `int64`** where the values are in `{0,1,2,3}`; `int8` would be 8× smaller (the reference uses `int8`). The neural payload dominates the 11.9 GB file, so this is a modest waste rather than a decisive one.

ii.
```python
    for i, (path, expected_units, expected_trials) in enumerate(inventory):
        print(f'[{i+1}/{len(inventory)}] {os.path.basename(path)} expected={expected_trials}x{expected_units}', flush=True)
```
```python
        trial_stops = tr['stop_time'][:n_trials]
        ...
        trial_stops = trial_stops[trial_keep]     # never read again
```
```python
        video_data = tongue['data'][:]            # (n_frames, 3); column 0 unused
        clean_y, likelihood, visible, outliers, percentiles = clean_tongue_y(video_t, video_data)
```
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
    outputs = np.empty((n_trials, 4, N_TIME), dtype=np.int64)
```

iii. Most of these are deliberate and documented as diagnostics rather than oversights. Step 5 planned the plot-based verification that `likelihood`/`outliers` serve: "Plot raster/rates, tone-relative time, photostim intervals, tongue likelihood/y/classes, and percentile boundaries for up to two sessions." The `stats` dict is the substrate for the consistency tables in Steps 7/9/10 ("Compare aggregate subject/session/trial/neuron counts and categorical distributions"). Step 6 states the intended dtype economy — "rates use float32 and outputs int64" — so the `int64` width is an explicit, if unoptimised, choice. The AI's own efficiency review (Step 6) lists only two inefficiencies, both about loops and I/O, and does not identify any discarded computation.
