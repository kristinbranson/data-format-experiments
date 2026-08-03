# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `data/sub-*/*.nwb` using `h5py` (not `pynwb`). Each file is opened once and processed via `process_session()`. Trials, events, units, and tongue tracking data are all extracted from within each file using direct HDF5 group access.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

Loading within a session:
```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    sample_event_times = load_event_times(f, 'sample_start_times')
    left_licks = load_event_times(f, 'left_lick_times')
    right_licks = load_event_times(f, 'right_lick_times')
    tongue_ts, tongue_data = load_tongue(f)
```

iii. The AI chose `h5py` over `pynwb` for direct HDF5 access. The CONVERSION_NOTES document that the NWB file structure was explored and all relevant groups identified (acquisition, units, intervals/trials, etc.).

## 1-b. How are the data split into subjects?

i. Each session's subject is read from `general/subject/subject_id` in the NWB file. Unique subjects are collected and sorted, and each session gets an index into that list.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```

Assembly:
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI follows the same approach as the reference, using the NWB subject ID field.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Session ID is derived from the file stem (filename without extension). Sessions are sorted by file path.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
session_id = path.stem
```

iii. The AI recognized the one-file-per-session structure of the dataset.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The number of go cue events is asserted to match the number of trials.

ii.
```python
trial = load_trial_table(f)
go_times = load_event_times(f, 'go_start_times')
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
```

iii. The AI correctly uses the trials table and validates against go cue event count.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in several ways: (1) it skips trials where `start < 0` (i.e., trial window extends before time 0), (2) it uses `is_good_trials` from the units table (if present) to check per-unit validity per trial, skipping trials where no units are valid, and (3) it skips trials that fall outside the global neural binning grid. It does NOT filter by `obs_intervals` or `free_water`. There is no explicit exclusion of free-water trials.

ii.
```python
if start < 0:
    continue

if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue

start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
```

iii. The CONVERSION_NOTES mention checking for trial validity via `is_good_trials` but do not discuss `obs_intervals` or `free_water` filtering, which the reference code uses. The AI uses `is_good_trials` as a per-unit, per-trial validity check, which is a different mechanism than `obs_intervals`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` for units with `classification == 'good'`. Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
```

```python
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. The AI creates a global session-wide binning grid from the earliest to latest go cue window, histograms all spikes for each good unit across this grid, then slices per-trial windows from the global rate matrix. Counts are converted to firing rates (Hz) by dividing by bin size. For trials with `is_good_trials` marking certain units as invalid, those units' rates are set to 0.0.

ii.
```python
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size

go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
trial_mat = global_rates[:, start_idx:end_idx].copy()
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The global-grid approach is a valid optimization. The zeroing of invalid units per trial is a unique choice not in the reference code; the reference instead filters entire trials based on `obs_intervals`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Additionally, if `is_good_trials` exists in the units table, it is used to zero out specific units on specific trials where they are not considered valid.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```

```python
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and ...) else None
...
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The `classification == 'good'` filter matches the reference. The additional `is_good_trials` zeroing is not in the reference code and could introduce artifacts (zeros where the reference would simply exclude the entire trial).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. A global session-wide binning grid is constructed, then per-trial windows are sliced using the go cue time to calculate the starting bin index.

ii.
```python
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The alignment to go cue is correct. The global-grid slicing approach requires that the global edges align exactly with per-trial edges, which depends on the `np.rint` rounding; this could introduce slight misalignment compared to the reference's approach of constructing per-trial edges directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins spanning -2.5 s to +1.5 s relative to go cue, giving 80 time bins. No rebinning is applied.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)

def build_edges(pre, post, bin_size):
    n_bins = int(round((pre + post) / bin_size))
    edges = np.linspace(-pre, post, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers
```

iii. Matches the instructions and the reference.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset events) and the go cue time. The AI selects the **first** `sample_start_times` event that falls within the trial's `[start_time, stop_time]` window.

ii.
```python
sample_event_times = load_event_times(f, 'sample_start_times')
...
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The AI uses the first sample event within the trial window. The reference instead uses the last sample event before the go cue (`searchsorted(..., side='left') - 1`), reasoning that an early lick replays the sample epoch, so the last tone before go is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_centers - tone_rel`, where `tone_rel = sample_time - go_time`. This is mathematically equivalent to `bin_centers + (go - sample_time)`, giving time since tone onset at each bin center.

ii.
```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)

def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

iii. The math is equivalent to the reference's `CENTERS + (go - tone)`. However, the tone selection differs (first vs last sample event), which would produce different values for trials with early licks.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers array, so the time-from-tone values are directly aligned with the neural bins.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
# same bin_centers used for both neural and input
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

iii. Correct alignment via shared bin centers.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary vector is built: 1 where `bin_center >= onset` and `bin_center < onset + duration`, 0 elsewhere. However, the AI uses `photostim_onset` directly as already relative to the go cue (passing it as `onset_rel` to `build_photostim_vector`), without adjusting for the fact that `photostim_onset` is stored relative to trial start, not relative to go cue.

ii.
```python
def build_photostim_vector(bin_centers, onset_rel, duration):
    x = np.zeros(bin_centers.shape[0], dtype=np.float32)
    if onset_rel is None or duration is None:
        return x
    if not np.isfinite(onset_rel) or not np.isfinite(duration):
        return x
    off = onset_rel + duration
    x[(bin_centers >= onset_rel) & (bin_centers < off)] = 1.0
    return x
```

iii. The AI treats `photostim_onset` as if it were relative to the go cue, but it is actually relative to trial start. The reference code correctly converts: `stim_on = trial_start + photostim_onset - go` to get the go-cue-relative onset time. This is a bug in the AI's code.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim vector uses the same `bin_centers` as the neural data, so the bins are aligned. However, due to the incorrect onset reference (see 4-b), the actual timing of the photostim signal is wrong.

ii.
```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
inp = np.stack([inp0, inp1], axis=0)
```

iii. Alignment mechanism is correct but the timing values are wrong due to the bug in 4-b.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from actual lick event times (`left_lick_times` and `right_lick_times`), checking which side was licked first in the response window `[go, trial_stop]`. If no lick is detected, it falls back to using the `trial_instruction` side.

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
...
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1

def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
    tl = l[0] if len(l) else np.inf
    tr = r[0] if len(r) else np.inf
    if tl == np.inf and tr == np.inf:
        return None
    return 0 if tl < tr else 1
```

iii. The AI uses actual lick events rather than deriving choice from instruction x outcome. When no lick occurs (ignore trials), it falls back to the instruction side rather than creating a "no lick" class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as left=0, right=1 (two classes). There is no "no lick" category. For ignore trials, the instruction side is used as a fallback. The value is repeated across all 80 time bins.

ii.
```python
'output_values': [
    ['left', 'right'],
    ...
]
...
out[0, :] = choice
```

iii. The reference has 3 output values: `['left', 'right', 'no lick']`, encoding ignore trials as class 2. The AI only has 2 classes, assigning ignore trials to the instruction side, which is incorrect since the animal made no choice on those trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which stores `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to ignore=0, miss=1, hit=2 and repeated across all time bins.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
...
out[1, :] = outcome
```

iii. Matches the reference exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which stores `'no early'` and `'early'`.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1 and repeated across all time bins.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
...
out[2, :] = early
```

iii. Matches the reference exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically the second column (y-position) of the data array, with timestamps.

ii.
```python
grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
data = np.asarray(grp['data'][()], dtype=float)
ts = np.asarray(grp['timestamps'][()], dtype=float)
...
tongue_y = tongue_data[:, ycol]  # ycol = 1
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI bins tongue y values into 50ms bins per trial by averaging frames within each bin. It does NOT filter frames by likelihood threshold. The 40th and 60th percentile thresholds are computed over all finite binned y values across the session. Bins are then classified: below 40th pct = 0, 40th-60th = 1, above 60th = 2. Bins with no tongue data default to class 1 (middle class).

ii.
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
yt = tongue_y[mask]
tt = tongue_ts[mask] - go
binned_y = np.full(len(bin_centers), np.nan, dtype=np.float32)
if len(tt):
    inds = np.digitize(tt, edges) - 1
    ok = (inds >= 0) & (inds < len(bin_centers)) & np.isfinite(yt)
    if np.any(ok):
        sums = np.zeros(len(bin_centers), dtype=np.float64)
        cnts = np.zeros(len(bin_centers), dtype=np.int64)
        np.add.at(sums, inds[ok], yt[ok])
        np.add.at(cnts, inds[ok], 1)
        nz = cnts > 0
        binned_y[nz] = (sums[nz] / cnts[nz]).astype(np.float32)

# Percentile computation
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y ...])
q40, q60 = np.percentile(all_y, [40, 60])

# Discretization
ycat = np.full(len(bin_centers), 1, dtype=np.int64)  # default to class 1
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

iii. Key differences from reference: (1) No likelihood filtering - the AI does not discard low-confidence frames (reference uses threshold of 0.5), (2) Bins with no data default to class 1 (middle) instead of a separate "not visible" class (reference uses class 3), (3) Only 3 output classes vs reference's 4.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of all finite binned tongue y values determine two thresholds. Values below 40th percentile get class 0, between 40th and 60th get class 1, above 60th get class 2. Bins with no data are defaulted to class 1. Only 3 classes total.

ii.
```python
q40, q60 = np.percentile(all_y, [40, 60])
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
```

iii. The reference uses `np.digitize` which gives 3 visible classes (0, 1, 2) plus a 4th "not visible" class (3). The AI lacks the "not visible" class and assigns those bins to the middle class instead.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue data window is defined by `go + edges[0]` to `go + edges[-1]`, matching the neural window. Frames are assigned to bins using `np.digitize` on the go-cue-relative timestamps against the bin edges.

ii.
```python
go = float(go_times[i])
start = go + edges[0]
stop = go + edges[-1]
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
inds = np.digitize(tt, edges) - 1
```

iii. Correct alignment via the same go-cue-relative bin grid as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units return None and are skipped. (2) Trials where `is_good_trials` marks all units as invalid are skipped; units marked invalid on specific trials have their rates zeroed. (3) Trials where the time window extends before time 0 or beyond the global grid are skipped. (4) Missing tongue data bins default to class 1. (5) Missing sample times result in NaN time-from-tone values.

ii.
```python
if len(good_idx) == 0:
    return None

if start < 0:
    continue

if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0

# tongue: default to class 1
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
```

iii. The AI does not use `obs_intervals` or `free_water` filtering. The zeroing of units per trial (rather than excluding trials) is a unique approach.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code constructs a global session-wide binning grid and histograms all spikes for each unit across it. Per the CONVERSION_NOTES, the estimated runtime was ~1.5 s/session, ~4.5 min for 174 sessions.

ii.
```python
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
```

iii. The CONVERSION_NOTES mention that the initial nested unit-by-trial approach was too slow and was optimized to a global-grid approach.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit histogram loop remains, as each unit has a different number of spikes (ragged storage). The per-trial tongue binning loop also remains.

ii.
```python
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
```

```python
for i in range(n_trials):
    ...
    mask = (tongue_ts >= start) & (tongue_ts < stop)
    ...
```

iii. The per-unit loop is inherent to ragged spike data. The per-trial loop for tongue and trial processing could potentially be vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The trial loop processes neural slicing, input construction, and output construction all within the same loop, so there is no redundant recomputation. However, the `out_outcome_map` and `out_early_map` dictionaries are recreated inside the loop for every trial.

ii.
```python
for i in range(n_trials):
    ...
    out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
    out_early_map = {'no early': 0, 'early': 1}
```

iii. The dictionary recreation is trivially cheap. No major repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `left_lick_times` and `right_lick_times` for every session to derive choice from actual lick events. This is unnecessary since choice can be derived from `trial_instruction` and `outcome` without loading the lick event streams.

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
```

iii. The lick event streams are loaded but could be avoided if choice were derived from instruction x outcome as the reference does.
