# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files with `Path('data').glob('sub-*/*.nwb')`, sorts them, and opens each NWB file directly with `h5py`. Within each file it reads the trial table from `intervals/trials`, event times from `acquisition/BehavioralEvents`, tongue tracking from `acquisition/BehavioralTimeSeries`, and unit information from `units`.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    sample_event_times = load_event_times(f, 'sample_start_times')
    left_licks = load_event_times(f, 'left_lick_times')
    right_licks = load_event_times(f, 'right_lick_times')
    tongue_ts, tongue_data = load_tongue(f)
```

iii. `CONVERSION_NOTES.md` says the raw data are organized as one NWB file per session under `data/sub-<subject_id>/`, and the trajectory records the decision to treat the raw NWB files as the full source universe and process each file once.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from `general/subject/subject_id` inside each NWB file. After processing all sessions, the AI forms a sorted unique subject list and builds `subject_idx` by indexing each session’s subject string into that list.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes explicitly say to use subject IDs from NWB subject metadata, and Step 2 documents that subject folders and `general/subject/subject_id` agree.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session order is the sorted file order, and session IDs are taken from `path.stem`, not from `nwb.identifier`.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
session_id = path.stem
...
'neural': [s['neural'] for s in sessions],
'input': [s['input'] for s in sessions],
'output': [s['output'] for s in sessions],
```

iii. `CONVERSION_NOTES.md` describes the data layout as one session per NWB file and records the plan to start from all 174 raw sessions, then exclude only sessions that fail integrity checks.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `intervals/trials`. The code sets `n_trials = len(trial['start_time'])` and requires the number of go cues to match that count. It then iterates over those trial indices.

ii.
```python
trial = load_trial_table(f)
go_times = load_event_times(f, 'go_start_times')
...
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
...
for i in range(n_trials):
    go = float(go_times[i])
```

iii. The trajectory shows the AI inspected the trial table and `go_start_times`, found all sessions had go cues, and used the trial table as the trial backbone.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference trial filter based on `obs_intervals` and `free_water`. Instead, it keeps almost all behavioral trials and only skips a trial if the extracted neural window would start before time 0, if the pre-binned slice would be out of bounds, or if `is_good_trials` exists and marks all good units invalid for that trial. It drops a session only if fewer than two trials remain.

ii.
```python
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if (
    'is_good_trials' in f['units']
    and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])
) else None
...
if start < 0:
    continue
...
if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
...
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
...
if len(session_trial_neural) < 2:
    return None
```

iii. The notes say the AI planned to “exclude only sessions that fail required data integrity checks,” and the trajectory shows it explored `is_good_trials` and `obs_intervals` but never implemented the reference `obs_intervals`/`free_water` trial filter. The final verification log still reports a large block of all-zero neural trials, consistent with that choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `units/spike_times` and `units/spike_times_index` for units whose `classification` is `'good'`. Trial go-cue times are used to define each trial window.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
...
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
...
go_times = load_event_times(f, 'go_start_times')
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `units/spike_times` plus `go_start_times` to `neural`, and Step 4 says unit inclusion should be restricted to `classification == 'good'` to match the curated dataset.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to firing rates by first histogramming each good unit once over a global session time grid at 50 ms resolution, then slicing the resulting rate matrix into per-trial windows around the go cue. Rates are in Hz.

ii.
```python
bin_size = edges[1] - edges[0]
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
...
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The Step 6 notes say the AI added a speedup by pre-binning spikes with NumPy histograms so full conversion would be fast enough, instead of repeatedly histogramming per unit per trial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered to `classification == 'good'`. In addition, when `units/is_good_trials` exists with a trial dimension matching the trial table, the AI zeroes out invalid units for that trial and skips trials where no good unit is valid.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
...
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if (
    'is_good_trials' in f['units']
    and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])
) else None
...
if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
...
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The notes justify the `classification == 'good'` rule from the QC white paper. The trajectory shows `is_good_trials` was investigated later as a possible explanation for all-zero trials; the final code retains a guarded version of that logic even though the agent discovered it was inconsistent across sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go-cue onset. The code builds bin edges from `-2.5` to `+1.5` seconds relative to go cue, computes each trial’s starting index on the global 50 ms grid from `go_times`, and slices exactly 80 bins for that trial.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
...
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The notes say the temporal alignment decision was to use `go_start_times`, and the metadata written by the script state the alignment event is “Go cue onset.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. Neural spikes are binned directly at that resolution; there is no second-stage temporal rebinning.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
...
'time_bin_size': 50.0,
```

iii. Step 5 in the notes records the explicit plan to bin spikes into 50 ms bins from `-2.5` to `+1.5` s relative to go cue, matching the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `go_start_times`, and the trial table’s `start_time`/`stop_time`. It searches for sample-start events that fall inside each trial and stores one sample time per trial.

ii.
```python
sample_event_times = load_event_times(f, 'sample_start_times')
...
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

iii. The notes say the AI treated `sample_start_times` as the likely tone proxy. The trajectory later records that `sample_start_times` is not one-per-trial and that the direct one-to-one assumption was incorrect, after which the code was changed to search within each trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI computes `tone_rel = sample_time - go_time` and then returns `bin_centers - tone_rel`, producing a continuous vector of time-from-tone values across the 80 go-aligned bins. However, if a trial has multiple `sample_start_times`, it uses the first one found within the trial rather than the last one before go cue.

ii.
```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
...
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

iii. The trajectory shows the AI noticed the one-event-per-trial assumption failed and adopted the “search within trial” workaround. There is no note justifying why the first within-trial sample event was chosen instead of the last pre-go sample event used by the reference.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns this input to neural data by evaluating the time-from-tone value at the same 80 go-aligned bin centers used for neural slicing, so `input[0]` and `neural` share a common time axis.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
...
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
inp = np.stack([inp0, inp1], axis=0)
```

iii. The notes describe the intended representation as a time-varying vector over the decoder bins, and the implementation uses the same `bin_centers` object that defines the neural windows.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trial-table fields `photostim_onset` and `photostim_duration`. It does not use `trial['start_time']`, `go_times`, or the session-level `photostim_start_times`/`photostim_stop_times` event streams to convert those values into go-relative time.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. The notes say photostimulation should be built from raw timing fields as a time-varying input, but the final code uses the per-trial onset and duration fields directly, without the extra timing reference that the trajectory had earlier identified as necessary.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses `photostim_onset` and `photostim_duration` as floats when present and fills a binary vector over the go-aligned bin centers wherever `bin_centers >= onset_rel` and `< onset_rel + duration`. Because it uses `photostim_onset` as if it were already go-relative, the resulting vectors are mis-timed; in the produced dataset they are all zero.

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

iii. `CONVERSION_NOTES.md` planned a binary photostim input. The trajectory earlier noted that photostim ended before go cue, but the final implementation does not convert the onset from trial-start-relative time into go-relative time.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI attempts to align photostimulation by expressing it on the same `bin_centers` axis as neural data, but it does not convert trial-relative photostim timing into go-relative timing first. So the array shape is aligned, but the timing is not.

ii.
```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
inp = np.stack([inp0, inp1], axis=0)
...
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The notes describe the intended alignment as go-cue aligned, but the final code never subtracts the go time from photostim onset. This mismatch is reflected in the final output, where `photostimulation_on` is zero for all retained trials.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the `left_lick_times` and `right_lick_times` event streams, plus `go_start_times` and `trial['stop_time']` to define the response window. If no lick is found, the code falls back to `trial_instruction`.

ii.
```python
left_licks = load_event_times(f, 'left_lick_times')
right_licks = load_event_times(f, 'right_lick_times')
...
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. Step 5 in the notes says the AI wanted to reconstruct “actual choice semantics” from lick events if possible. The trajectory shows it considered `trial_instruction` and lick times together when mapping choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code finds the first left and right lick after go cue and before trial stop. If neither exists, it assigns the instructed side as the choice. It writes only two classes, left `0` and right `1`, repeated across all bins; there is no explicit no-lick class.

ii.
```python
def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
    tl = l[0] if len(l) else np.inf
    tr = r[0] if len(r) else np.inf
    if tl == np.inf and tr == np.inf:
        return None
    return 0 if tl < tr else 1
...
'output_values': [
    ['left', 'right'],
```

iii. The notes and trajectory do not record a separate justification for collapsing ignore/no-lick trials into left or right. The final metadata and `output_values` confirm the AI decided on a two-class choice variable.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table field `outcome`.

ii.
```python
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. The notes explicitly mapped NWB `outcome` values `ignore`, `miss`, and `hit` to the required output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses a fixed dictionary `{'ignore': 0, 'miss': 1, 'hit': 2}` and repeats the resulting trial-level code across all 80 bins.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out[1, :] = outcome
```

iii. Step 5 of the notes states this direct categorical mapping decision and treats outcome as a per-trial variable.

## 6-c. How is `output` *Outcome* aligned with the neural data?

i. Outcome is aligned by trial index only. For each retained trial, the outcome code is repeated across the same 80 bins as the neural matrix.

ii.
```python
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[1, :] = outcome
```

iii. The notes describe per-trial categorical outputs stored alongside go-aligned neural windows; the code implements that as a repeated constant vector.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table field `early_lick`.

ii.
```python
early = out_early_map[str(trial['early_lick'][i])]
```

iii. The notes explicitly identify `early_lick` as a direct NWB trial annotation to map into the decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early'` to `0` and `'early'` to `1`, then repeats that trial-level code across all 80 bins.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
...
out[2, :] = early
```

iii. This is the direct mapping recorded in Step 5 of the notes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: the code loads the full data matrix and timestamps and uses column 1 as y-position.

ii.
```python
def load_tongue(f):
    grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
    data = np.asarray(grp['data'][()], dtype=float)
    ts = np.asarray(grp['timestamps'][()], dtype=float)
    return ts, data

def choose_tongue_y_column(data):
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('Unexpected tongue tracking shape')
    return 1
```

iii. The notes say the AI inferred that the data are likely `[x, y, likelihood]` and planned to use the second column as y-position. The final code hard-codes that choice.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, the AI takes frames in the go-aligned window, assigns frames to 50 ms bins with `np.digitize`, and averages all finite y-values per bin. It then concatenates all finite per-trial binned values across the session and computes the 40th and 60th percentiles from those values. It does not use the tracking likelihood column to filter low-confidence frames.

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
...
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))])
q40, q60 = np.percentile(all_y, [40, 60])
```

iii. The notes planned per-session percentile discretization of binned tongue y-values, but the trajectory shows the AI never fully resolved the confidence/visibility handling and simply implemented the y-column averaging path.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses only three categories. It initializes every bin to class `1` (middle), sets finite bins below the 40th percentile to `0`, above the 60th percentile to `2`, and between them to `1`. Missing bins are therefore silently assigned to the middle category rather than a separate “not visible” class.

ii.
```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
...
'output_values': [
    ['lt_40pct', '40to60pct', 'gt_60pct'],
],
```

iii. `CONVERSION_NOTES.md` records the plan to produce the three instructed percentile-based categories. The final code follows that plan literally, without adding the hidden-tongue category used by the human reference.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is aligned on the same go-relative `[-2.5, 1.5]` window used for neural data. The code selects tongue frames with absolute timestamps between `go + edges[0]` and `go + edges[-1]`, subtracts `go`, and bins those relative times using the same `edges` array used for neural windows.

ii.
```python
start = go + edges[0]
stop = go + edges[-1]
...
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
inds = np.digitize(tt, edges) - 1
```

iii. The notes state that go-cue alignment should be shared across neural, input, and video-derived outputs, and the implementation uses the same `edges` array for tongue and neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or inconsistent data in several ad hoc ways: sessions with no `classification == 'good'` units are dropped; `is_good_trials` is ignored unless its shape matches the trial count; trials are skipped if the requested neural window would be out of bounds or start before time 0; sessions with fewer than two kept trials are dropped; sessions with no finite tongue data after binning are dropped; and missing tongue bins are assigned category `1` indirectly by default. It does not implement the reference handling of behavior trials lacking spike observations via `obs_intervals` and `free_water`.

ii.
```python
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
...
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if (
    'is_good_trials' in f['units']
    and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])
) else None
...
if start < 0:
    continue
...
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
...
if len(session_trial_neural) < 2:
    return None
...
if len(all_y) == 0:
    return None
```

iii. The trajectory explicitly records that `is_good_trials` had inconsistent shapes across sessions and was guarded. The notes also say the AI would exclude only sessions failing integrity checks, but there is no comparable justification for leaving `obs_intervals`/`free_water` unhandled or for mapping missing tongue bins to the middle category.

## 10-a. What are the most time-consuming steps of the code?

i. The heaviest parts of the final code are loading large NWB arrays, histogramming spike times for every good unit across the full session grid, and trial-by-trial binning of tongue data. Optional plotting can also add cost when enabled.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    spike_times = f['units']['spike_times'][()]
    ...
    tongue_ts, tongue_data = load_tongue(f)
...
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
...
for i in range(n_trials):
    ...
    inds = np.digitize(tt, edges) - 1
```

iii. The Step 6 and Step 7 notes discuss runtime explicitly, first calling spike binning the likely bottleneck and later estimating roughly 1.5 seconds per session after optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the per-trial search over `sample_event_times`, the per-unit histogram loop for spikes, the per-trial `find_choice_from_licks` scans over session-wide lick arrays, and the per-trial tongue binning loop.

ii.
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
...
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
...
for i in range(n_trials):
    ...
    choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
    ...
    inds = np.digitize(tt, edges) - 1
```

iii. The Step 6 notes explicitly mention efficiency concerns and earlier versions had even more nested loops; the final code keeps a session-wide pre-binning optimization but leaves several repeated trial scans in place.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans session-wide event arrays inside per-trial loops. It rescans `sample_event_times` for each trial to find a sample onset, rescans all left and right licks for each trial inside `find_choice_from_licks`, and recomputes tongue bin assignments separately for each trial.

ii.
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
...
def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
...
for i in range(n_trials):
    ...
    inds = np.digitize(tt, edges) - 1
```

iii. The notes do not describe these as deliberate design choices; they follow from the AI’s local fixes during development, especially after it discovered that `sample_start_times` could not be treated as one value per trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps an unused `valid_trial_ids` list, computes `global_centers` only to size the session-wide spike-rate array, and can optionally render plots that are not part of the converted dataset. More importantly, it computes continuous `binned_y` traces only to discard them after discretization, although that intermediate is required for the final categories.

ii.
```python
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
...
valid_trial_ids = []
...
valid_trial_ids.append(i)
...
def maybe_plot(session_info, outdir='.'):
    ...
```

iii. There is no explicit justification in the notes for `valid_trial_ids`; it appears to be leftover scaffolding from development. The plotting path is justified only by the optional `--show-processing` inspection mode.
