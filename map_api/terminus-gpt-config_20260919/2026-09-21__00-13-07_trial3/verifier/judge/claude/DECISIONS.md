# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all `.nwb` files under `/app/data` recursively, sorts them, and processes each file sequentially with `pynwb.NWBHDF5IO`. Each NWB file is one session. Trials come from `nwb.trials`, events from `nwb.acquisition['BehavioralEvents']`, units from `nwb.units`, and tongue tracking from `nwb.acquisition['BehavioralTimeSeries']`.

ii.
```python
DATA_ROOT = Path('/app/data')
files = sorted(DATA_ROOT.rglob('*.nwb'))
# ...
for p in files:
    res = process_session(p, make_plot)
```

```python
def process_session(path, make_plot=False):
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        unit_inds, classifier_good = selected_unit_indices(nwb.units)
        trials = nwb.trials
        events = nwb.acquisition['BehavioralEvents'].time_series
        go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. CONVERSION_NOTES.md documents that the dataset is organized as one NWB file per session under subject directories, with 174 files total. The AI uses `pynwb` as required by the instructions and processes all files in sorted order.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from `nwb.subject.subject_id` for each session (numeric strings like `'440956'`). At assembly, unique sorted subject IDs form the `subjects` list, and `subject_idx` maps each session to its position.

ii.
```python
subject = str(nwb.subject.subject_id)
# ...
unique_subjects = sorted(set(subjects))
subject_lookup = {x: i for i, x in enumerate(unique_subjects)}
'subject_idx': np.asarray([subject_lookup[r['subject']] for r in results], dtype=np.int64),
```

iii. The AI notes that numeric `subject_id` differs from mouse names in the papers (e.g., `440956` is mouse `SC015`), but uses the NWB field directly. 28 subjects are identified, matching the paper.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Sessions are identified by `nwb.identifier`. 174 files are found; one is dropped for having no classifier-good units, leaving 173 sessions matching the paper.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
# ...
identifier = str(nwb.identifier)
```

iii. The dandiset stores one session per file, so no splitting or grouping is needed. The AI documents that 173 sessions are retained after dropping the one file with no labeled units.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table (`nwb.trials`), one row per trial. The number of go-cue events is asserted to equal the number of trial rows.

ii.
```python
trials = nwb.trials
events = nwb.acquisition['BehavioralEvents'].time_series
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
assert len(go_all) == len(trials)
```

iii. The AI verifies the one-to-one mapping between trials and go events.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT use `obs_intervals` or filter `free_water` trials. Instead, it keeps ALL trial rows initially, bins spikes for all of them, then removes trials where every selected neuron has zero spikes across the entire 4-second window (detected as "simultaneous zero-total-spike acquisition gaps"). Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
# obs_intervals encode a paper-analysis subset that strongly excludes error
# trials, not general acquisition validity. Preserve all released trial rows;
# true simultaneous acquisition gaps are detected from zero total raw spikes.
trial_keep = np.ones(len(go_all), dtype=bool)
# ...
rates = bin_spikes(nwb.units, unit_inds, go)
activity_keep = np.any(rates != 0, axis=(1, 2))
excluded_zero_activity_trials = int((~activity_keep).sum())
if activity_keep.sum() < 2:
    return None
rates = rates[activity_keep]
```

iii. The AI's CONVERSION_NOTES explain that it initially tried using `obs_intervals` but found they "strongly exclude error trials" (reducing miss outcomes from 16.5% to 0.93%), so it rejected that approach. Instead it detects acquisition gaps empirically from zero activity. The final trial count (90,859) and outcome distributions closely match the reference's output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for selected units. Go-cue times from `BehavioralEvents/go_start_times` provide the alignment.

ii.
```python
spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
```

iii. Spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. For each selected unit, spikes are assigned to the latest trial-window start via `searchsorted`, then accumulated into 80 non-overlapping 50ms bins using `bincount`. Counts are divided by `BIN` (0.05) to convert to firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spikes(units, unit_inds, go):
    starts = go + OFF_START
    ntr, nneu = len(go), len(unit_inds)
    rates = np.zeros((ntr, nneu, N_TIME), dtype=np.float32)
    for k, unit_i in enumerate(unit_inds):
        spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
        trial_i = np.searchsorted(starts, spikes, side='right') - 1
        # ... validity checks ...
        bi = np.floor(rel[valid2] / BIN).astype(np.int64)
        flat = ti * N_TIME + bi
        counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
        rates[:, k, :] = counts.astype(np.float32) / BIN
    return rates
```

iii. The binning matches the 50ms non-overlapping bins specified in the instructions. Converting counts to Hz matches the reference code's `sliding_histogram(..., rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `classification == 'good'` from the QC classifier, and (2) `is_good_trials` must be `True` for every trial in the session. The second filter removes 565 additional units (from 69,453 to 68,888). A session with no surviving units is dropped.

ii.
```python
def selected_unit_indices(units):
    classification = np.asarray(units['classification'][:]).astype(str)
    candidates = np.flatnonzero(classification == 'good')
    keep = []
    for j in candidates:
        if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
            keep.append(int(j))
    return np.asarray(keep, dtype=np.int64), len(candidates)
```

iii. The AI's CONVERSION_NOTES justify this as preserving fixed neuron dimensions per session: removing the 565 units avoids the problem of some units being invalid on some trials while maintaining a fixed neuron-by-time matrix shape.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go-cue times share the same session-absolute clock. The bin window starts at `go + OFF_START` (-2.5s) for each trial. No additional alignment or interpolation is needed.

ii.
```python
go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
# In bin_spikes:
starts = go + OFF_START
```

iii. The common NWB timestamp base eliminates the need for cross-stream alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s relative to go cue, producing 80 timepoints per trial. The bin grid is defined once as 81 edges using `np.linspace`.

ii.
```python
BIN = 0.05
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
```

iii. Matches the instructions (50ms bins, -2.5 to +1.5s, go cue alignment).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` event timestamps and go-cue times. The tone onset for each trial is the last `sample_start_times` event before that trial's go cue.

ii.
```python
sample = np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64)
sample_i = np.searchsorted(sample, go, side='right') - 1
assert np.all(sample_i >= 0)
tone = sample[sample_i]
```

iii. The AI correctly handles early-lick replay (which creates multiple sample events per trial) by taking the latest sample onset before each go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. At each bin center, the value is the absolute time of the center minus the tone onset time. This is computed as `go + center_offset - tone` for each trial.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
```

iii. This produces a continuous time-varying ramp, consistent with the instruction's request for "time from tone onset in seconds."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers used for the time-from-tone computation are the same go-cue-relative centers used for the neural data, so they share the same time grid by construction.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
```

iii. Both neural and input use the same `CENTERS_REL` grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` event timestamps in `BehavioralEvents`.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64)
assert len(ps) == len(pe)
```

iii. The AI chose event timestamps over trial-table string fields (`photostim_onset` = `'N/A'` or float string). CONVERSION_NOTES state this avoids string parsing and captures the actual stimulation intervals.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its center falls within any photostim start/stop interval, 0 otherwise. For each bin center, the code finds the latest photostim_start before that time and checks if the center is before the corresponding stop.

ii.
```python
if len(ps):
    event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
    valid_event = event_i >= 0
    safe_i = np.maximum(event_i, 0)
    photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)
```

iii. This produces a binary time-varying input as specified. Non-stimulated trials have all-zero vectors.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim state is evaluated at `absolute_centers`, the same go-relative bin centers as the neural data, ensuring temporal alignment.

ii.
```python
absolute_centers = go[:, None] + CENTERS_REL[None, :]
```

iii. Same grid as neural and time-from-tone.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') in the trials table. No explicit choice column exists in the data.

ii.
```python
instruction = np.asarray(trials['trial_instruction'][:]).astype(str)[trial_keep]
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
choice = np.full(len(go), 2, dtype=np.int64)  # no lick
choice[(outcome_s == 'hit') & (instruction == 'left')] = 0
choice[(outcome_s == 'hit') & (instruction == 'right')] = 1
choice[(outcome_s == 'miss') & (instruction == 'left')] = 1
choice[(outcome_s == 'miss') & (instruction == 'right')] = 0
```

iii. Hit means licked the instructed side; miss means licked the opposite side; ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0=left, 1=right, 2=no lick. It is a per-trial value repeated across all 80 time bins.

ii.
```python
outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                     np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
           for j in range(len(go))]
```

iii. Per-trial outputs are tiled to match the (n_output, n_timepoints) format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, containing 'ignore', 'miss', 'hit'.

ii.
```python
outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_s], dtype=np.int64)
```

iii. The three categories match the instructions exactly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Per-trial value repeated across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.asarray([outcome_map[x] for x in outcome_s], dtype=np.int64)
# repeated in output array via np.full(N_TIME, outcome[j])
```

iii. Straightforward categorical encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing 'no early' and 'early'.

ii.
```python
early_s = np.asarray(trials['early_lick'][:]).astype(str)[trial_keep]
early = (early_s == 'early').astype(np.int64)
```

iii. The trials table provides this flag directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value repeated across all 80 bins.

ii.
```python
early = (early_s == 'early').astype(np.int64)
```

iii. Binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains (n_frames, 3) data = (tongue_x, tongue_y, tongue_likelihood) with timestamps. Column 1 is y-position; column 2 is likelihood.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_t = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue_ts.data[:], dtype=np.float64)
y_all, like_all = tongue_data[:, 1], tongue_data[:, 2]
```

iii. Camera0 side tongue tracking is present in all 174 sessions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a **nearest-frame approach**: for each bin center, it finds the nearest camera frame. If the nearest frame is within 10ms (`MAX_VIDEO_GAP`) and has likelihood >= 0.9 (`LIKELIHOOD_THRESHOLD`), the tongue y-value is used; otherwise the bin is class 3 ("not visible"). Session-wide percentiles (40th, 60th) are computed from all frames with likelihood >= 0.9 to define class edges.

ii.
```python
LIKELIHOOD_THRESHOLD = 0.9
MAX_VIDEO_GAP = 0.010

visible_all = np.isfinite(y_all) & np.isfinite(like_all) & (like_all >= LIKELIHOOD_THRESHOLD)
if visible_all.any():
    p40, p60 = np.percentile(y_all[visible_all], [40, 60])

y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
nearest_like = like_all[nearest_i]
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
tongue_class = np.full(query.shape, 3, dtype=np.int64)
tongue_class[visible & (y < p40)] = 0
tongue_class[visible & (y >= p40) & (y <= p60)] = 1
tongue_class[visible & (y > p60)] = 2
```

iii. The AI justifies the 0.9 likelihood threshold as "conservative" since no paper/code threshold is specified (CONVERSION_NOTES Step 5). The nearest-frame approach differs from the reference's bin-mean approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes: 0 (below 40th percentile), 1 (40th-60th percentile), 2 (above 60th percentile), 3 (not visible). Percentiles are computed per-session from all visible (likelihood >= 0.9) raw y-values.

ii.
```python
tongue_class[visible & (y < p40)] = 0
tongue_class[visible & (y >= p40) & (y <= p60)] = 1
tongue_class[visible & (y > p60)] = 2
```

iii. The categorization matches the instructions' specification. The percentiles are computed from raw visible samples rather than from bin means (as the reference does).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue y is queried at the same `absolute_centers` (go-relative bin centers) as the neural data. The nearest camera frame to each bin center is used, with a 10ms gap tolerance.

ii.
```python
query = absolute_centers  # go[:, None] + CENTERS_REL[None, :]
y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
```

iii. Same temporal grid as neural and input data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Session with no classifier labels**: `classification` values that aren't strings are cast via `.astype(str)` to `'nan'`, which doesn't match `'good'`, so the session has zero selected units and is dropped (returns `None`).
- **Trials with no spike data (acquisition gaps)**: Detected post-hoc by finding trials where all selected neurons have zero spikes across the full window. These trials are excluded.
- **Tongue frames with low confidence**: Nearest frames with likelihood < 0.9 or gap > 10ms are marked as class 3 ("not visible").
- **Truncated video stream**: One session with only 13,995 tongue samples is handled gracefully; uncovered bins become class 3.

ii.
```python
# Session with no good units
if len(unit_inds) == 0:
    return None

# Zero-activity trials
activity_keep = np.any(rates != 0, axis=(1, 2))

# Tongue visibility
visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
```

iii. The AI documents each case in CONVERSION_NOTES and handles them without fabricating data.

## 10-a. What are the most time-consuming steps of the code?

i. NWB file I/O dominates. The full conversion takes ~265 seconds for 173 sessions (~1.5s/session). Pickling the ~11.8 GB result adds ~12 seconds. Within a session, the per-unit spike binning loop is the main computation.

ii. N/A (timing reported in CONVERSION_NOTES)

iii. The AI profiled the conversion and found it well within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop (`for k, unit_i in enumerate(unit_inds)`) iterates over units, reading each unit's spike times individually. This could potentially be improved by reading the entire spike_times buffer at once (as the reference does) and processing all units' spikes in a single sorted-edge searchsorted call.

ii.
```python
for k, unit_i in enumerate(unit_inds):
    spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
    trial_i = np.searchsorted(starts, spikes, side='right') - 1
    # ...
```

iii. The AI's approach reads spike times per-unit rather than loading the full ragged buffer, adding per-unit I/O overhead. The reference reads `units['spike_times'].data` and `units['spike_times'].target.data` once for all units.

## 10-c. What processing does the code repeat multiple times?

i. No obvious repeated processing. Each session is processed once. The bin grid is computed once at module level and reused.

ii. N/A

iii. The AI's conversion is a single pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins ALL trials' spikes before filtering out zero-activity trials. This means spike binning is performed for trials that are later discarded. The reference avoids this by filtering trials before binning.

ii.
```python
rates = bin_spikes(nwb.units, unit_inds, go)  # bins ALL trials
activity_keep = np.any(rates != 0, axis=(1, 2))  # then filters
rates = rates[activity_keep]
```

iii. Additionally, the AI computes and stores extensive metadata per session (fine region labels, unit IDs, original trial indices, tongue percentiles, etc.) that go into session_info but are not used by the decoder.
