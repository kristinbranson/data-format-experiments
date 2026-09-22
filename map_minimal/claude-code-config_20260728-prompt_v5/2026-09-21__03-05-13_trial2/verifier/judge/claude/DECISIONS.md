# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files with a glob over `sub-*/sub-*.nwb` under the data directory and processes each file individually using `h5py` (not `pynwb`). Each file is opened with `h5py.File` and the trials table, units table, behavioral events, and behavioral time series are accessed via HDF5 paths.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
```

```python
with h5py.File(fpath, 'r') as f:
    trials = f['intervals/trials']
    ...
    go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    sample_starts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    units = f['units']
```

iii. The agent explored the NWB file structure using h5py in preliminary investigation steps and chose to continue using h5py for the final script rather than switching to pynwb, which is the standard NWB reader.

## 1-b. How are the data split into subjects?

i. The subject ID is extracted from the NWB filename by splitting on `_` and taking the first part (e.g., `sub-440956` from the filename). Subjects are collected as a list during iteration, preserving first-seen order.

ii.
```python
basename = os.path.basename(fpath)
subject_id = basename.split('_')[0]
```

```python
if result['subject_id'] not in all_subjects:
    all_subjects.append(result['subject_id'])
```

iii. The agent parsed the filename to get subject IDs rather than reading the `nwb.subject.subject_id` field from the NWB file. This produces IDs like `sub-440956` rather than `440956`.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. Sessions are identified by the file basename. Session order follows the sorted file list.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
```

```python
'session_name': basename,
```

iii. The agent correctly recognized that one NWB file = one session. The sorted glob gives deterministic order.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per trial. Go cue times are read from `BehavioralEvents/go_start_times/timestamps`.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'][:])
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The agent used the trials table directly. No explicit assertion was made to check that the number of go cues matches the number of trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of filtering:

**Session-level filtering**: Sessions are excluded if (1) performance on control trials (hit rate among responded control non-early-lick, non-photostim, non-auto_water, non-free_water trials) is below 65%, or (2) there are fewer than 50 correct lick-left or 50 correct lick-right control trials.

**Trial-level filtering**: Trials where `auto_water == 1` or `free_water == 1` are excluded. Additionally, trials outside the neural recording range (based on min/max spike times with 1.0s tolerance) are excluded. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
# Session filtering
is_control = (photostim_onset_raw == 'N/A') & (auto_water == 0) & (free_water == 0)
is_not_early = (early_licks == 'no early')
control_regular = is_control & is_not_early
performance = control_hits / control_responded
if performance < MIN_PERFORMANCE:
    return None
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
```

```python
# Trial filtering
trial_mask = (auto_water == 0) & (free_water == 0)
```

```python
# Neural range filtering
valid_trial_mask = np.array([
    (go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and
    (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0)
    for ti in trial_indices
])
```

iii. The agent derived the session-level performance criteria from the methods text describing the original paper's analysis pipeline. The neural range filtering was added after discovering that some sessions had trials extending beyond the neural recording period, causing all-zero neural data. The agent used min/max spike times rather than the NWB `obs_intervals` field.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times of each unit) and `units/spike_times_index` (the ragged array index). Only units with `unit_quality == 'good'` are used.

ii.
```python
all_spike_times = units['spike_times'][:]
spike_times_index = units['spike_times_index'][:]
```

```python
unit_quality = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in units['unit_quality'][:]])
good_indices = np.where(unit_quality == 'good')[0]
```

iii. The agent correctly identified spike_times as the source of neural data.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins spanning [-2.5, 1.5] s relative to the go cue (80 bins). For each good unit, per-trial bin edges are computed as absolute times, `np.searchsorted` gives spike counts at each edge, and differencing gives counts per bin. Counts are divided by bin width (0.05 s) to give firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
```

```python
for t_idx in range(n_trials):
    go_t = go_subset[t_idx]
    abs_edges = go_t + BIN_EDGES
    edge_counts = np.searchsorted(st, abs_edges)
    rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
```

iii. The approach matches the standard spike-count-to-rate conversion. However, the AI loops over trials inside the unit loop (doubly nested), whereas the reference vectorizes the trial dimension by flattening all trial edges into one array per unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `unit_quality == 'good'` are retained. No thresholds on individual quality metrics are applied.

ii.
```python
unit_quality = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in units['unit_quality'][:]])
good_indices = np.where(unit_quality == 'good')[0]
if len(good_indices) == 0:
    return None
```

iii. The agent treated `unit_quality` as the QC classifier output, stating it "matches classifier-based QC from data paper." However, the NWB files contain a separate `classification` field which is the actual output of the QC classifier described in the spike sorting QC paper. `unit_quality` is an older, more permissive label that disagrees with the classifier on ~12% of units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed relative to the go cue for each trial. The absolute bin edges are `go_t + BIN_EDGES`, and spikes are binned against these edges directly. Since all times are on the same session-absolute clock, no separate alignment step is needed.

ii.
```python
abs_edges = go_t + BIN_EDGES
edge_counts = np.searchsorted(st, abs_edges)
rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
```

iii. The agent correctly aligned to the go cue as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50 ms as specified in the instructions. 80 non-overlapping bins span [-2.5, 1.5] s relative to the go cue. Bin edges are defined with `np.linspace`. No rebinning is applied; spikes are binned directly from raw spike times.

ii.
```python
TIME_BEFORE = 2.5
TIME_AFTER = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((TIME_BEFORE + TIME_AFTER) / BIN_SIZE))  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
```

iii. The grid matches the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onset timestamps) in BehavioralEvents, together with the go cue time for each trial.

ii.
```python
sample_starts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
```

```python
for i in range(n_trials):
    go_t = go_times[i]
    lower = go_times[i - 1] if i > 0 else 0.0
    mask_s = (sample_starts > lower) & (sample_starts < go_t)
    candidates = sample_starts[mask_s]
    if len(candidates) > 0:
        tone_onsets[i] = candidates[-1]
```

iii. The agent correctly identified that early lick replays create multiple sample_start events per trial and took the last one before the go cue as the relevant tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset relative to the go cue is computed, then the time from tone onset at each bin center is `BIN_CENTERS - tone_rel`. If no tone onset is found, a default offset of 1.85s (typical sample + delay duration) is used.

ii.
```python
tone_rel = tone_t - go_t  # negative
time_from_tone = BIN_CENTERS - tone_rel
```

Fallback:
```python
if np.isnan(tone_t):
    time_from_tone = BIN_CENTERS + 1.85
```

iii. The computation is equivalent to the reference's `CENTERS + (go - tone)` since `BIN_CENTERS - (tone - go) = BIN_CENTERS + (go - tone)`. The fallback for missing tone onsets is an extra handling not in the reference.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural data and the time-from-tone input use the same bin grid defined relative to the go cue (`BIN_EDGES` / `BIN_CENTERS`), ensuring alignment.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. Alignment is inherent in the shared bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`, which provide absolute session timestamps of photostimulation onset and offset.

ii.
```python
if 'photostim_start_times' in f['acquisition/BehavioralEvents']:
    ps_ts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps']
    if ps_ts.shape[0] > 0:
        ps_start_abs = ps_ts[:]
        ps_stop_abs = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
```

iii. The agent chose to use the BehavioralEvents timestamps rather than the trials table columns (`photostim_onset`, `photostim_duration`). Both approaches should yield the same result, but the reference uses the trials table approach.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, all photostim events are checked for overlap with the trial's time window. A bin is set to 1.0 if its center falls between any photostim start and stop time (relative to go cue).

ii.
```python
for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
    if ps_e < win_start or ps_s > win_end:
        continue
    ps_s_rel = ps_s - go_t
    ps_e_rel = ps_e - go_t
    on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
    photostim_vec[on_mask] = 1.0
```

iii. The result is a binary time series matching the reference approach, though derived from different source variables.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim start/stop times are converted to go-cue-relative times, then compared against `BIN_CENTERS` which are the same bin centers used for neural data.

ii.
```python
ps_s_rel = ps_s - go_t
ps_e_rel = ps_e - go_t
on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
```

iii. Alignment is ensured by the shared go-cue-relative time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `outcome` and `trial_instruction` columns of the trials table, same as the reference.

ii.
```python
outcome = outcomes[trial_idx]
instruction = instructions[trial_idx]
if outcome == 'ignore':
    choice = 2  # no lick
elif outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 2
```

iii. Choice is not stored directly, so it must be derived from instruction and outcome. A hit means the animal licked the instructed side; a miss means it licked the opposite side; ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no_lick, and broadcast across all 80 time bins.

ii.
```python
outputs[0, :] = choice
```

iii. The coding matches the reference (left=0, right=1, no lick=2).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table.

ii.
```python
outcomes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                    for x in trials['outcome'][:]])
```

iii. Outcome is stored explicitly in the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: hit=0, miss=1, ignore=2. This is broadcast across all 80 bins.

ii.
```python
outcome_val = {'hit': 0, 'miss': 1, 'ignore': 2}.get(outcome, 2)
outputs[1, :] = outcome_val
```

```python
'output_values': [
    ...
    ['hit', 'miss', 'ignore'],
    ...
]
```

iii. The numeric mapping differs from the reference (ignore=0, miss=1, hit=2) but is internally consistent with the `output_values` labels (`output_values[1][0] = 'hit'` matches code 0). Both are valid encodings since the output_values list defines the meaning of each integer.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` or `'early'`.

ii.
```python
early_licks = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in trials['early_lick'][:]])
```

iii. Early lick is stored explicitly in the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) and 1 (early), broadcast across all 80 bins.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
outputs[2, :] = early_val
```

iii. Matches the reference encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, which has columns (x, y, likelihood) with matching timestamps. Column 1 (y) is the position; column 2 (likelihood/confidence) determines visibility.

ii.
```python
tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
tongue_ts = tt['timestamps'][:]
tdata = tt['data'][:]
tongue_y = tdata[:, 1]
tongue_conf = tdata[:, 2]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with confidence <= 0.9 are considered not visible. The 40th and 60th percentiles are computed over **raw visible frames** across the whole session (not binned means). For each trial bin, the nearest tongue frame is found via `searchsorted`, and if it is within 5ms and above the confidence threshold, the y-position is discretized against the percentile edges into classes 0/1/2. Otherwise the bin is class 3 (not visible).

ii.
```python
DLC_CONFIDENCE_THRESHOLD = 0.9
vis = tongue_conf > DLC_CONFIDENCE_THRESHOLD
if np.any(vis):
    tongue_y_p40 = np.percentile(tongue_y[vis], 40)
    tongue_y_p60 = np.percentile(tongue_y[vis], 60)
```

```python
idx_all = np.searchsorted(tongue_ts, t_abs_all)
# ... nearest frame logic ...
tongue_y_disc[valid & (y_vals < tongue_y_p40)] = 0
tongue_y_disc[valid & (y_vals >= tongue_y_p40) & (y_vals < tongue_y_p60)] = 1
tongue_y_disc[valid & (y_vals >= tongue_y_p60)] = 2
```

iii. Several differences from the reference: (1) confidence threshold is 0.9 vs reference's 0.5, (2) percentiles are computed on raw frames rather than 50ms bin means, (3) per-trial values use nearest-frame interpolation rather than averaging all frames within each 50ms bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles of visible-frame y-positions define the class boundaries. Values below 40th percentile = 0, between 40th and 60th = 1, above 60th = 2, not visible = 3.

ii.
```python
tongue_y_disc[valid & (y_vals < tongue_y_p40)] = 0
tongue_y_disc[valid & (y_vals >= tongue_y_p40) & (y_vals < tongue_y_p60)] = 1
tongue_y_disc[valid & (y_vals >= tongue_y_p60)] = 2
```

iii. The thresholding logic matches the instructions' specification. The difference is in what the percentiles are computed over (raw frames vs bin means) and the confidence threshold used.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center, the nearest tongue tracking frame (by timestamp) is found using `searchsorted`. If the nearest frame is within 5ms and has sufficient confidence, its y-value is used; otherwise the bin is marked as not visible.

ii.
```python
t_abs_all = go_t + BIN_CENTERS
idx_all = np.searchsorted(tongue_ts, t_abs_all)
idx_all = np.clip(idx_all, 0, len(tongue_ts) - 1)
idx_prev = np.clip(idx_all - 1, 0, len(tongue_ts) - 1)
dist_cur = np.abs(tongue_ts[idx_all] - t_abs_all)
dist_prev = np.abs(tongue_ts[idx_prev] - t_abs_all)
best_idx = np.where(dist_prev < dist_cur, idx_prev, idx_all)
best_dist = np.minimum(dist_cur, dist_prev)
close_enough = best_dist < 0.005
```

iii. The nearest-frame approach differs from the reference, which averages all frames falling within each 50ms bin. At ~294 Hz, there are ~15 frames per bin, so nearest-frame discards most of the data within each bin. The reference's bin-averaging approach better summarizes the tongue position during each time bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing tone onset**: If no sample_start is found before a trial's go cue, a default offset of 1.85s is used.
- **Trials outside recording range**: Filtered using min/max spike times with 1.0s tolerance.
- **No tongue tracking data**: If `Camera0_side_TongueTracking` is absent or percentiles cannot be computed, all bins default to class 3 (not visible).
- **Sessions with no good units**: Skipped (returns None).
- **Low-performing sessions**: Filtered by the 65% performance threshold.

ii.
```python
if np.isnan(tone_t):
    time_from_tone = BIN_CENTERS + 1.85
```

```python
valid_trial_mask = np.array([
    (go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and
    (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0)
    for ti in trial_indices
])
```

iii. The agent handled missing data pragmatically but did not use `obs_intervals` (the NWB file's own record of which trials were observed), instead building a custom spike-range-based filter. The tone onset fallback is not present in the reference.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with h5py and loading all spike times into memory. The full conversion takes several minutes for 174 sessions. The per-unit, per-trial double loop in `compute_firing_rates_all_trials` is also expensive due to lack of trial-dimension vectorization.

ii.
```python
for i, st in enumerate(unit_spikes):
    for t_idx in range(n_trials):
        go_t = go_subset[t_idx]
        abs_edges = go_t + BIN_EDGES
        edge_counts = np.searchsorted(st, abs_edges)
        rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
```

iii. The agent initially had an even slower implementation and optimized it, but the final version still has a double loop (units x trials) rather than the reference's single loop (units only, with trials vectorized).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner trial loop in `compute_firing_rates_all_trials` could be vectorized by flattening all trial edges into one array per unit and using a single `searchsorted` call, as the reference does. The tone onset computation also has a per-trial loop that could use vectorized `searchsorted`.

ii.
```python
# Current: double loop
for i, st in enumerate(unit_spikes):
    for t_idx in range(n_trials):
        abs_edges = go_t + BIN_EDGES
        edge_counts = np.searchsorted(st, abs_edges)
```

Reference's vectorized approach:
```python
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
```

iii. The reference eliminates the trial loop by flattening all edges and doing one searchsorted per unit.

## 10-c. What processing does the code repeat multiple times?

i. The photostimulation computation iterates over ALL photostim events for EVERY trial, checking for overlap. This redundantly tests events from other trials. The tone onset computation similarly loops over all trials with per-trial masking.

ii.
```python
for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
    if ps_e < win_start or ps_s > win_end:
        continue
```

iii. The early-exit check mitigates the cost, but this is still O(n_trials * n_photostim_events) rather than O(n_trials).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes electrode location mappings (`get_electrode_target_region`) that parse JSON location strings for every electrode. This is used as a fallback for brain region mapping when `anno_name` is missing, but involves unnecessary JSON parsing overhead. Additionally, the large `ANNO_TO_REGION` and `TARGET_TO_REGION` mapping dictionaries group brain regions into coarse categories, which changes the granularity of the brain region information compared to what's available in the data.

ii.
```python
def get_electrode_target_region(location_json):
    d = json.loads(location_json)
    brain_regions = d['brain_regions']
    parts = brain_regions.split()
    side = parts[0]
    region_name = ' '.join(parts[1:])
    major = TARGET_TO_REGION.get(region_name, region_name)
    return side, major
```

iii. The coarse region grouping reduces the information content compared to the reference's fine-grained annotations.
