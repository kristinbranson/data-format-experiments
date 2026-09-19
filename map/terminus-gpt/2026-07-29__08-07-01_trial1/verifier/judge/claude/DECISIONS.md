# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing for `*.nwb` files under `data/` using `pathlib.Path.rglob`, sorting them, and processing each file with `h5py.File` (not `pynwb`). Each NWB file is one session; trials, units, and behavioral events are read from HDF5 groups within each file.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
files = choose_sessions(files, sample=args.sample)
# ...
for fp in files:
    sess = load_session(fp, brain_regions)
```

```python
def load_session(nwb_path, brain_regions):
    with h5py.File(nwb_path, 'r') as f:
        trials = f['intervals/trials']
        n_trials = trials['id'].shape[0]
        # ...
```

iii. The AI chose `h5py` over `pynwb` for direct HDF5 access. The trajectory shows the AI explored the data directory, found NWB files organized by subject directories, and iterated over all of them. No explicit justification was given for choosing `h5py` over `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). The full directory name (including `sub-` prefix) is used as the subject ID.

ii.
```python
def get_subject_id(nwb_path):
    return nwb_path.parent.name
```

iii. The AI used the filesystem directory structure to determine subject identity. This gives subject IDs like `sub-440956` rather than the numeric `440956` that would come from the NWB `subject.subject_id` field, but the mapping is equivalent.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is the filename stem (without extension).

ii.
```python
def get_session_id(nwb_path):
    return nwb_path.stem
```

iii. This is straightforward since the dataset stores one session per NWB file. The AI processes all 174 NWB files and doesn't drop any session based on QC status (unlike the reference which drops sessions with no `classification == 'good'` units).

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in the NWB file. The AI iterates over trials indexed by position, matching each trial index to the corresponding go cue time by position.

ii.
```python
trials = f['intervals/trials']
n_trials = trials['id'].shape[0]
# ...
n_match = min(n_trials, len(go_times))
for i in range(n_match):
    go = go_times[i]
```

iii. The AI assumes a 1:1 positional correspondence between trial table rows and go cue events, using `min(n_trials, len(go_times))` as a safety measure. No assertion is made to verify the counts match.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by several criteria: (1) go time must be finite, (2) the trial window `[go + T_START, go + T_END]` must fit within `[trial_start, trial_stop]`, (3) the trial's `choice`, `outcome`, and `early_lick` values must be recognized strings. No filtering by `obs_intervals` or `free_water` is applied.

ii.
```python
for i in range(n_match):
    go = go_times[i]
    if not np.isfinite(go):
        continue
    if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
        continue
    if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
        continue
```

iii. The AI's trajectory shows it was aware of the reference code's `get_regular_trial_mask` which excludes early lick, auto water, free water, and no-response trials, but the AI chose not to exclude early lick or ignore trials since they are required decoder outputs. However, the AI also does not use `obs_intervals` to identify trials without spike data, and does not filter `free_water` trials (which have no spikes). The `auto_water` and `free_water` columns are loaded but only `auto_water` is unused.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the flat spike time array) and `units/spike_times_index` (the ragged index). Only units with `unit_quality == 'good'` that belong to MAJOR_REGIONS are included.

ii.
```python
spikes_flat = f['units/spike_times'][:]
spikes_index = f['units/spike_times_index'][:]
starts = np.concatenate([[0], spikes_index[:-1]])
kept_spikes = [spikes_flat[starts[u]:spikes_index[u]] for u in kept_idx]
```

iii. The AI reads the full spike times buffer once and slices per unit, which is efficient. The spike times are on the same absolute session clock as the behavioral events.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning [-2.5 s, +1.5 s] relative to the go cue using `searchsorted` and `bincount`, then divided by bin size to convert to firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    a = np.searchsorted(spike_times, lo, side='left')
    b = np.searchsorted(spike_times, hi, side='left')
    if b <= a:
        return np.zeros(N_BINS, dtype=np.float32)
    rel = spike_times[a:b] - lo
    bins = np.floor(rel / BIN_SIZE).astype(np.int64)
    bins = bins[(bins >= 0) & (bins < N_BINS)]
    counts = np.bincount(bins, minlength=N_BINS).astype(np.float32)
    return counts / BIN_SIZE
```

iii. The AI uses a per-unit, per-trial approach (called inside a nested loop), converting spike times to relative times and using `floor` division for bin assignment. This is correct but less efficient than the reference's vectorized approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by two criteria: (1) `unit_quality == 'good'` and (2) the unit's brain region (derived from electrode location JSON) must be in `MAJOR_REGIONS` (left/right ALM, Striatum, Thalamus, Midbrain, Medulla).

ii.
```python
unit_quality = decode_arr(f['units/unit_quality'][:])
unit_electrodes = f['units/electrodes'][:]
elec_regions = parse_region_strings(f['general/extracellular_ephys/electrodes/location'][:])
unit_regions = elec_regions[unit_electrodes]
keep_units = (unit_quality == 'good') & np.isin(unit_regions, list(MAJOR_REGIONS))
```

iii. The AI's trajectory shows it struggled with the unit count discrepancy (154,948 `unit_quality == 'good'` vs. the paper's 69,943). It resolved this by adding a region filter to major broad regions. However, the AI used `unit_quality` instead of `classification`, which is the QC classifier verdict described in the spike sorting QC paper. These are different columns with different meanings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue by computing spike times relative to `go_time + T_START` and binning from there. The go cue times come from `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
# ...
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

```python
def bin_unit_spikes_fast(spike_times, go_time):
    lo = go_time + T_START
    hi = go_time + T_END
    # ...
    rel = spike_times[a:b] - lo
    bins = np.floor(rel / BIN_SIZE).astype(np.int64)
```

iii. The alignment is correct — all spikes are referenced to the go cue time, giving 80 bins from -2.5 s to +1.5 s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms, with 80 bins spanning [-2.5 s, +1.5 s] relative to the go cue. No rebinning is applied — spike times are binned directly into the final resolution.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. These values match the instructions exactly: 50 ms bins, -2.5 to +1.5 s around go cue, giving 80 time bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (from BehavioralEvents) and the go cue time. The AI finds the last sample event within the trial boundaries before the go cue.

ii.
```python
sample_event_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
# ...
sample_time = find_last_event_within_trial(sample_event_times, trial_start[i], trial_stop[i], before_time=go)
```

iii. The AI correctly identifies that `sample_start_times` can have multiple entries per trial (due to early lick replays) and takes the last one before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `BIN_CENTERS - tone_rel` where `tone_rel = sample_time - go_time`. Negative values (bins before the tone) are clipped to 0.

ii.
```python
def build_time_from_tone(sample_time, go_time):
    tone_rel = sample_time - go_time
    rel = BIN_CENTERS - tone_rel
    rel[rel < 0] = 0.0
    return rel.astype(np.float32)
```

iii. The clipping of negative values to 0 means that time bins before the tone onset all have value 0, losing the information about how long before the tone each bin is. The reference solution does not clip and allows negative values, preserving full temporal information.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same `BIN_CENTERS` array (relative to go cue) is used for both neural binning and computing time from tone, ensuring alignment.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
# ...
rel = BIN_CENTERS - tone_rel
```

iii. Both neural and input share the same time grid defined by `BIN_CENTERS`, so alignment is inherent.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the `photostim_start_times` and `photostim_stop_times` event streams in `BehavioralEvents`, filtered to events within the trial's `[start_time, stop_time]` boundaries.

ii.
```python
photo_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
photo_stops = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
# ...
ps = photo_starts[(photo_starts >= trial_start[i]) & (photo_starts <= trial_stop[i])]
pe = photo_stops[(photo_stops >= trial_start[i]) & (photo_stops <= trial_stop[i] + 1e-6)]
m = min(len(ps), len(pe))
inp1 = build_photostim_series(ps[:m], pe[:m], go)
```

iii. The AI uses the event streams rather than the trial table fields (`photostim_onset`, `photostim_duration`). It matches start/stop pairs within trial boundaries, using `min(len(ps), len(pe))` to handle potential mismatches.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is constructed: bins where the left edge is before the stop time and right edge is after the start time of any photostim event get value 1, else 0.

ii.
```python
def build_photostim_series(start_times, stop_times, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(start_times, stop_times):
        rs = s - go_time
        re = e - go_time
        overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
        x[overlap] = 1.0
    return x
```

iii. The AI uses bin edges for overlap detection rather than bin centers. This is slightly different from the reference which uses bin centers (`CENTERS >= stim_on` and `CENTERS < stim_off`). The overlap-based approach marks a bin as 1 if any part of the bin overlaps with the photostim period.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim start/stop times are converted to go-cue-relative times and compared against the same `BIN_EDGES` grid used for neural data.

ii.
```python
rs = s - go_time
re = e - go_time
overlap = (BIN_EDGES[:-1] < re) & (BIN_EDGES[1:] > rs)
```

iii. Alignment is achieved by expressing everything relative to the go cue, which is the same reference point used for neural binning.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from the `trial_instruction` column of the trials table, mapping `'left'` to 0 and `'right'` to 1.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
# ...
choice = decode_arr(trials['trial_instruction'][:])
# ...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
```

iii. The AI maps `trial_instruction` (the instructed lick direction) directly as the choice output. This is NOT the actual lick direction — `trial_instruction` tells the animal which side to lick, not which side it actually licked.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `trial_instruction` is mapped to integers (left=0, right=1) and broadcast across all 80 time bins. Only two categories are used — there is no "no lick" category for ignore trials. Ignore trials are filtered out entirely (since `'ignore'` is not in `CHOICE_MAP`... but wait, `CHOICE_MAP` maps `trial_instruction` values, not `outcome` values. The filter `choice[i] not in CHOICE_MAP` checks `trial_instruction[i]` against `{'left': 0, 'right': 1}`, so trials where `trial_instruction` is not 'left' or 'right' would be excluded. In practice, all trials have `trial_instruction` of 'left' or 'right', so this filter doesn't exclude ignore trials.)

ii.
```python
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
# ...
np.full(N_BINS, CHOICE_MAP[choice[i]], dtype=np.int64),
```

iii. The output has only 2 values (left=0, right=1) with `output_values = ['left', 'right']`. Ignore trials (no lick) are included but labeled with the instructed direction rather than a "no lick" category. The reference solution derives the actual lick direction from instruction x outcome and includes a third "no lick" category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, containing `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = decode_arr(trials['outcome'][:])
# ...
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
```

iii. The trials table stores the outcome explicitly; no derivation needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers (ignore=0, miss=1, hit=2) and broadcast across all 80 time bins.

ii.
```python
np.full(N_BINS, OUTCOME_MAP[outcome[i]], dtype=np.int64),
```

iii. Straightforward mapping matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing `'no early'` and `'early'`.

ii.
```python
EARLY_MAP = {'no early': 0, 'early': 1}
early = decode_arr(trials['early_lick'][:])
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to integers (no early=0, early=1) and broadcast across all 80 time bins.

ii.
```python
np.full(N_BINS, EARLY_MAP[early[i]], dtype=np.int64),
```

iii. Straightforward mapping matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data`, column index 1 (tongue y), and the corresponding timestamps.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_t = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_y_all = tongue[:, 1].astype(np.float32)
```

iii. The AI correctly identifies the tongue tracking time series and extracts the y-position column.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Tongue y-position is interpolated to bin centers using `np.interp` (linear interpolation), without any likelihood filtering. Session-wide 40th and 60th percentiles are computed on all finite tongue y values (raw frames, not bin means). Discretization assigns: 0 if below 40th percentile, 1 otherwise (default), 2 if above 60th percentile.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    if valid.sum() < 2:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan).astype(np.float32)
```

```python
all_tongue = np.concatenate([x[np.isfinite(x)] for x in sess_tongue_cont ...])
q40, q60 = np.percentile(all_tongue, [40, 60])
for ty, i in zip(sess_tongue_cont, valid_trial_inds):
    disc = np.full(N_BINS, 1, dtype=np.int64)
    disc[ty < q40] = 0
    disc[ty > q60] = 2
```

iii. The AI does NOT filter by tongue tracking likelihood (column 2). The reference filters frames with likelihood < 0.5, setting them to NaN, because the tracker reports positions even when the tongue is retracted. The AI also uses linear interpolation to resample to bin centers rather than computing bin means. Additionally, no "not visible" class (3) is assigned — bins where interpolation yields NaN default to class 1 (the middle category).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (between 40th and 60th percentile, the default), 2 (above 60th percentile). There is no "not visible" (class 3) category. The `output_values` lists these as `['low', 'mid', 'high']`.

ii.
```python
disc = np.full(N_BINS, 1, dtype=np.int64)
disc[ty < q40] = 0
disc[ty > q60] = 2
```

iii. The instructions specify a fourth category (3: "not visible") for bins where the tongue is not visible. The AI omits this category entirely. The reference solution assigns class 3 to bins with no visible tongue frames, and the tongue is not visible ~75% of the time.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue tracking timestamps are converted to go-cue-relative times and interpolated to the same `BIN_CENTERS` used for neural data.

ii.
```python
def interp_tracking_to_bins(track_t, track_y, go_time):
    rel_t = track_t - go_time
    valid = np.isfinite(track_y) & np.isfinite(rel_t)
    # ...
    return np.interp(BIN_CENTERS, rel_t[valid], track_y[valid], left=np.nan, right=np.nan)
```

iii. Alignment is via `BIN_CENTERS` (the same grid as neural data), but the method uses linear interpolation across all valid frames rather than binning frames into 50 ms windows. The reference uses bin means within each 50 ms window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with non-finite go times are skipped. (2) Trials where the window doesn't fit within trial start/stop are skipped. (3) Trials with unrecognized string values in choice/outcome/early_lick are skipped. (4) Sessions with fewer than 2 valid trials are dropped. (5) Tongue y NaN values from interpolation result in the default middle class (1) rather than a separate "not visible" class.

ii.
```python
if not np.isfinite(go):
    continue
if go + T_START < trial_start[i] or go + T_END > trial_stop[i]:
    continue
if choice[i] not in CHOICE_MAP or outcome[i] not in OUTCOME_MAP or early[i] not in EARLY_MAP:
    continue
# ...
if len(sess_neural) < 2:
    return None
```

iii. The AI handles missing data by skipping trials/sessions. However, it does not handle the case where a session was never quality-controlled (where `classification` and `anno_name` are NaN) — this doesn't apply since it uses `unit_quality` instead. It also doesn't handle `free_water` trials or use `obs_intervals`.

## 10-a. What are the most time-consuming steps of the code?

i. The per-unit, per-trial spike binning loop is the most expensive operation. Each unit's spikes are binned separately for each trial using `searchsorted` + `bincount`. Reading the full spike times array from HDF5 is also expensive. The AI's trajectory mentions optimizing from ~30.6 s to ~11.0 s for 2 sessions.

ii.
```python
neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The AI identified spike binning as a bottleneck and optimized it with `searchsorted`, but still loops over units and calls per-trial within each unit call.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop nests trial iteration (outer) with per-unit spike binning (inner via list comprehension). The reference solution vectorizes over all trials simultaneously for each unit by flattening all trial bin edges into one array. The tongue y interpolation loop (one `np.interp` call per trial) could also be vectorized.

ii.
```python
for i in range(n_match):
    # ...
    neural = np.stack([bin_unit_spikes_fast(sp, go) for sp in kept_spikes], axis=0)
```

iii. The per-trial outer loop means every unit's spikes are searched separately for every trial, rather than all trials at once per unit.

## 10-c. What processing does the code repeat multiple times?

i. The tongue tracking data (`tongue`, `tongue_t`) is loaded once per session but the `interp_tracking_to_bins` function is called per trial, recomputing `rel_t` and the `valid` mask each time on the full session data.

ii.
```python
for i in range(n_match):
    # ...
    ty = interp_tracking_to_bins(tongue_t, tongue_y_all, go)
```

iii. The validity check and relative time computation on the full session data are repeated for each trial unnecessarily.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `left_lick_times`, `right_lick_times`, and `auto_water` from the NWB file but never uses them in the conversion.

ii.
```python
left_lick = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_lick = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
# ...
auto_water = trials['auto_water'][:]
```

iii. These variables are loaded but never referenced in the output construction, wasting I/O time.
