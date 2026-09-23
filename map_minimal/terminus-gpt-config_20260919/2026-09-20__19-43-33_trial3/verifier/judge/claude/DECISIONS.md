# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with `Path.rglob('*.nwb')` and sorted. Each file is opened with `pynwb.NWBHDF5IO`, and subjects, trials, units, and acquisition time series are read from within each file.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for si, path in enumerate(files):
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
```

iii. The agent recognized that NWB is the published format for this dataset and used `pynwb` as the standard reader. The agent inspected the data directory structure, counted 174 NWB files across 28 subjects, and iterated over all files.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `nwb.subject.subject_id`, a numeric string (e.g. `'440956'`). Subjects are extracted from the file paths by splitting on `_` and removing the `sub-` prefix. A mapping from subject to index is built once before processing sessions.

ii.
```python
subjects = sorted({p.name.split('_')[0].replace('sub-', '') for p in files})
subject_to_idx = {s:i for i,s in enumerate(subjects)}
...
sid = str(nwb.subject.subject_id)
...
subject_idx.append(subject_to_idx[sid])
```

iii. The agent identified subject IDs from the NWB file naming convention and the `subject.subject_id` field. The resulting 28 subjects match the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. No grouping or splitting is needed. Sessions are processed in sorted file-path order.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for si, path in enumerate(files):
```

iii. The agent identified that each NWB file is a complete session. 173 of 174 sessions are retained (one is dropped for having no good annotated units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The agent reads `start_time` and `stop_time` from `nwb.trials` and assigns go-cue and tone events to trials by finding the first event within each trial's `[start_time, stop_time)` interval.

ii.
```python
tr = nwb.trials
starts = np.asarray(tr['start_time'][:], float)
stops = np.asarray(tr['stop_time'][:], float)
ntr = len(starts)
go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
```

iii. The agent verified that go cue events exist for every trial and raises an error if any are missing.

## 1-e. How are trials filtered based on quality controls?

i. No trial filtering is applied. All trials that have a go cue and a tone onset are retained. There is no filtering based on `obs_intervals`, `free_water`, `early_lick`, or any other quality criterion.

ii.
```python
# No trial filtering code - all ntr trials are processed
for ti in range(ntr):
    bt = go[ti] + CENTERS
    sess_n.append(bin_spikes(spikes, go[ti]))
    ...
```

Metadata explicitly states:
```python
'trial_filter': 'all trials with go cue and tone onset'
```

iii. The agent inspected the `is_good_trials` mask for a sample session and found it was all-True, which may have contributed to the decision not to apply trial filtering. Early lick and ignore trials were deliberately retained as they are decoder output variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, the sorted spike times of each unit. Only units passing the quality filter contribute (see 2-c). The go-cue times are used to place bin edges.

ii.
```python
spike_col = nwb.units['spike_times']
spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
spike_ends = np.asarray(spike_col.data[:], dtype=np.int64)
spike_starts = np.r_[0, spike_ends[:-1]]
spikes = [spike_data[spike_starts[i]:spike_ends[i]] for i in kept_idx]
```

iii. `spike_times` is the only neural representation in the file, so firing rates must be computed from it directly.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning -2.5s to +1.5s relative to the go cue. Bin edges are added to each trial's go-cue time. `np.searchsorted` with `side='left'` gives insertion indices at each edge, and differencing gives spike counts per bin. Counts are divided by the bin width (0.05s) to yield firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
def bin_spikes(spike_times, go):
    out = np.empty((len(spike_times), 80), dtype=np.float32)
    abs_edges = go + EDGES
    for u, st in enumerate(spike_times):
        st = np.asarray(st)
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
    return out
```

iii. The agent noted this matches the reference code's `sliding_histogram(..., rate=True)` approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent uses `unit_quality == 'good'` (not `classification == 'good'`) AND requires a valid, non-empty histological annotation (`anno_name`). Units with empty or `'nan'` annotations are excluded. Sessions with no surviving units are dropped.

ii.
```python
quality = np.asarray(nwb.units['unit_quality'][:], dtype=str)
annotation = np.char.strip(np.asarray(nwb.units['anno_name'][:], dtype=str))
keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
kept_idx = np.flatnonzero(keep)
if len(kept_idx) == 0:
    print('  skipped: no good annotated units', flush=True)
    continue
```

iii. The agent searched methods.txt and the reference code for unit quality criteria and noted that the paper mentions classifier-labeled 'good' units. However, the agent chose `unit_quality` instead of `classification`. The agent also required histology (non-empty `anno_name`) following the reference code's pattern of intersecting ephys units with histology data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and event times are on the same session-absolute clock. Bin edges relative to the go cue are added to each trial's go-cue time. Spikes are binned against those absolute edges using `searchsorted`.

ii.
```python
abs_edges = go + EDGES
for u, st in enumerate(spike_times):
    st = np.asarray(st)
    out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
```

iii. Everything in the NWB file shares one global clock, so aligning to the go cue only requires adding the relative bin grid to the go-cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins, 80 bins total spanning -2.5s to +1.5s relative to go cue. Bin edges are defined using `np.linspace(START, END, 81)`. No rebinning is applied - spikes are directly binned at this resolution.

ii.
```python
START, END, DT = -2.5, 1.5, 0.05
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The 50ms bin width and -2.5 to +1.5s window follow the instructions directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents`, which are the tone onset timestamps. The agent uses the **first** `sample_start_times` event falling within each trial's `[start_time, stop_time)` interval.

ii.
```python
tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
```

```python
def first_event_in_trial(times, starts, stops):
    """First event in each NWB trial, NaN if absent."""
    times = np.asarray(times, dtype=np.float64)
    ans = np.full(len(starts), np.nan)
    j = 0
    for i, (a, b) in enumerate(zip(starts, stops)):
        j = np.searchsorted(times, a, side='left')
        if j < len(times) and times[j] < b:
            ans[i] = times[j]
    return ans
```

iii. The agent recognized that `sample_start_times` can have more entries than trials (due to early-lick replays) and chose the first event per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is computed as `bin_center_time - tone_onset_time`, where bin centers are relative to the go cue.

ii.
```python
bt = go[ti] + CENTERS
time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. Straightforward subtraction of the tone timestamp from each bin center's absolute time.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers used for the time-from-tone calculation are the same centers used for the neural data bins (go cue + CENTERS), ensuring alignment.

ii.
```python
bt = go[ti] + CENTERS
sess_n.append(bin_spikes(spikes, go[ti]))
time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. Both neural and input data share the same temporal grid defined by `EDGES`/`CENTERS` relative to the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`, which provide the absolute timestamps of photostimulation onset and offset across the session.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], float)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], float)
```

iii. The agent observed that the trials-table `photostim_onset` column contained `'N/A'` strings for non-stimulated trials and chose to use the BehavioralEvents timestamps instead, which provide actual numerical timestamps.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its center falls within any `[photostim_start, photostim_stop)` interval, and 0 otherwise. The comparison is done across all photostimulation epochs at once.

ii.
```python
photo = np.zeros(80, dtype=np.float32)
if len(ps):
    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. This creates a binary time-varying input as required by the instructions.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The bin centers (`bt = go[ti] + CENTERS`) used to determine photostimulation status are the same centers as the neural data bins, ensuring alignment.

ii.
```python
bt = go[ti] + CENTERS
photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. Same temporal grid as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') in the trials table.

ii.
```python
instruction = np.asarray(tr['trial_instruction'][:], dtype=str)
outcome = np.asarray(tr['outcome'][:], dtype=str)
...
if outcome[ti] == 'ignore':
    choice = 2                         # no lick/no response
elif outcome[ti] == 'hit':
    choice = 0 if instruction[ti] == 'left' else 1
else:                                  # miss: wrong direction
    choice = 1 if instruction[ti] == 'left' else 0
```

iii. The agent correctly identified that lick direction is not stored directly but can be inferred: hit means the animal licked the instructed side, miss means it licked the opposite side, ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0=left, 1=right, 2=no lick. It is a per-trial value repeated across all 80 time bins.

ii.
```python
o = np.empty((4,80), dtype=np.int8)
o[0] = choice
```

Output values:
```python
['left','right','no lick']
```

iii. Follows the instructions. Per-trial value broadcast across bins to maintain rectangular output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which holds strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = np.asarray(tr['outcome'][:], dtype=str)
...
out_map = {'ignore':0, 'miss':1, 'hit':2}
o[1] = out_map[outcome[ti]]
```

iii. The trials table stores the outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: 0=ignore, 1=miss, 2=hit. Per-trial value repeated across all 80 bins.

ii.
```python
out_map = {'ignore':0, 'miss':1, 'hit':2}
o[1] = out_map[outcome[ti]]
```

iii. Follows the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early = np.asarray(tr['early_lick'][:], dtype=str)
...
o[2] = 1 if early[ti] == 'early' else 0
```

iii. Direct from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value repeated across all 80 bins.

ii.
```python
o[2] = 1 if early[ti] == 'early' else 0
```

iii. Follows the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `tongue_x`, `tongue_y`, `tongue_likelihood`, with matching `timestamps`. Column 1 is the y-position; column 2 is the likelihood.

ii.
```python
tts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
cam_t = np.asarray(tts.timestamps[:], dtype=float)
cam = np.asarray(tts.data[:], dtype=float)
```

iii. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session-wide percentiles (40th, 60th) are computed over **raw camera frames** where `likelihood >= 0.9`. Per-trial, the nearest camera sample to each bin center is found, and if visible (likelihood >= 0.9), its y-value is classified against the percentile edges. Bins with no visible frame are assigned class 3 ("not visible").

ii.
```python
DLC_THRESHOLD = 0.9
...
visible_all = np.isfinite(cam[:,1]) & np.isfinite(cam[:,2]) & (cam[:,2] >= DLC_THRESHOLD)
if not np.any(visible_all):
    q40 = q60 = np.nan
else:
    q40, q60 = np.percentile(cam[visible_all,1], [40, 60])
...
idx = np.searchsorted(cam_t, bt)
idx = np.clip(idx, 1, len(cam_t)-1)
prev = idx - 1
idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
y = cam[idx,1]
vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
tongue_cat = np.full(80, 3, dtype=np.int8)
tongue_cat[vis & (y < q40)] = 0
tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
tongue_cat[vis & (y > q60)] = 2
```

iii. The agent chose a DLC likelihood threshold of 0.9 (standard for DLC) and computed percentiles over raw frames rather than bin-averaged values.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Categories are assigned using strict inequalities: 0 if `y < q40`, 1 if `q40 <= y <= q60`, 2 if `y > q60`, 3 if not visible. Note the middle category uses `<=` for both boundaries.

ii.
```python
tongue_cat[vis & (y < q40)] = 0
tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
tongue_cat[vis & (y > q60)] = 2
```

iii. This follows the instructions' specification of `< 40th`, `40th to 60th`, `> 60th` percentile boundaries.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center, the nearest camera sample (by timestamp) is found using `searchsorted` and compared against the adjacent sample. The nearest sample's y-value and visibility are used for that bin.

ii.
```python
idx = np.searchsorted(cam_t, bt)
idx = np.clip(idx, 1, len(cam_t)-1)
prev = idx - 1
idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
```

iii. This nearest-neighbor interpolation approach is simpler than averaging all frames within each bin. The camera runs at ~294 Hz (about 15 frames per 50ms bin), so using the nearest sample discards most of the available data within each bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Two cases:
- **Sessions with no good annotated units**: Dropped (1 session dropped).
- **Tongue frames with low likelihood**: Not visible frames (likelihood < 0.9) are excluded from percentile computation and assigned category 3.

Missing go cues or tone onsets raise a `ValueError`, halting processing rather than silently dropping trials.

ii.
```python
if len(kept_idx) == 0:
    print('  skipped: no good annotated units', flush=True)
    continue
...
if np.any(~np.isfinite(go)):
    raise ValueError(f'{path}: {np.sum(~np.isfinite(go))} trials lack go cue')
```

iii. The agent treats missing critical data (go cues, tones) as errors rather than silently handling them. Low-confidence tongue tracking is handled via the visibility threshold.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file (especially loading the full spike_times buffer) dominates. The per-unit spike binning loop also scales with unit count. The per-trial loop over all trials within a session is another bottleneck since spike binning is called once per trial rather than being vectorized across trials.

ii.
```python
for ti in range(ntr):
    bt = go[ti] + CENTERS
    sess_n.append(bin_spikes(spikes, go[ti]))
```

iii. The code processes one trial at a time within each session, calling `bin_spikes` for each trial separately.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop could be vectorized: instead of calling `bin_spikes` once per trial (each time iterating over all units), all trials' edges could be flattened into a single array and processed with one `searchsorted` call per unit (as the reference code does). The per-trial tongue alignment could also be vectorized.

ii.
```python
# Current: one bin_spikes call per trial
for ti in range(ntr):
    sess_n.append(bin_spikes(spikes, go[ti]))

# Inside bin_spikes: one searchsorted per unit per trial
def bin_spikes(spike_times, go):
    for u, st in enumerate(spike_times):
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
```

iii. The reference vectorizes the trial dimension by flattening all trial edges into one array, achieving `O(n_units)` searchsorted calls per session instead of `O(n_units * n_trials)`.

## 10-c. What processing does the code repeat multiple times?

i. The `bin_spikes` function is called once per trial, each time iterating over all units. This means each unit's spike train is searched `n_trials` times (once per trial) rather than once with all trial edges concatenated.

ii.
```python
for ti in range(ntr):
    sess_n.append(bin_spikes(spikes, go[ti]))
```

iii. The spike binning work scales as `n_units * n_trials` calls to `searchsorted`, whereas vectorizing across trials would require only `n_units` calls.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code retains trials that have no spike data (e.g., trials outside `obs_intervals`, free_water trials). These trials will have all-zero or near-zero neural activity, which adds noise and computational cost to decoder training without providing useful signal.

ii.
```python
# No trial filtering - all ntr trials are included
for ti in range(ntr):
    ...
```

iii. The reference code filters out trials without spike data (obs_intervals) and free_water trials, which removes ~3.7% of trials that would otherwise contribute uninformative data.
