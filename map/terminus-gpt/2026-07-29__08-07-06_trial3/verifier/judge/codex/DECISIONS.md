# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `data/sub-*/*.nwb`, sorting the paths, and opening each NWB file directly with `h5py`. Within each file it reads the trials table from `intervals/trials`, event timestamps from `acquisition/BehavioralEvents`, tongue tracking from `acquisition/BehavioralTimeSeries`, and unit metadata plus spike arrays from `units`.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```

```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    sample_event_times = load_event_times(f, 'sample_start_times')
    left_licks = load_event_times(f, 'left_lick_times')
    right_licks = load_event_times(f, 'right_lick_times')
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly decided to treat the raw NWB files in `data/sub-*` as the full source universe and to process every session file. The trajectory also shows it chose direct NWB inspection and direct reading with low-level HDF5 access rather than relying on the paper code's preprocessed pickles.

## 1-b. How are the data split into subjects?

i. Subjects are read from `general/subject/subject_id` in each NWB file. After processing all sessions, the AI builds a sorted unique `subjects` list and a per-session `subject_idx`.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes state that subject IDs should come from the NWB subject metadata, and Step 2 records that subject metadata live under `general/subject`. There is no evidence of any alternative grouping heuristic.

## 1-c. How are the data split into sessions?

i. The AI treats one NWB file as one session. Session order follows the sorted file list, and each session is labeled with `path.stem`.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
session_id = path.stem
```

iii. The notes repeatedly describe the dataset as one NWB file per session under each `sub-*` directory, and the trajectory shows the AI using that file boundary as the session boundary throughout.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trial table, using one row of `intervals/trials` per trial. The AI assumes the `go_start_times` event stream is one-to-one with trial rows and asserts that the lengths match.

ii.
```python
trial = load_trial_table(f)
go_times = load_event_times(f, 'go_start_times')
...
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
```

```python
for i in range(n_trials):
    go = float(go_times[i])
    ...
```

iii. In Step 5 of the notes, the AI planned to use behavioral event timestamps plus the trials table as the native trial structure. The trajectory shows it rejecting a naive one-event-per-trial assumption for `sample_start_times`, but not for `go_start_times`.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference trial QC. Instead, it skips trials whose pre-go window starts before time 0, optionally skips trials with no `is_good_trials` units when that matrix matches the trial count, skips trials whose extracted window would go out of the session-wide histogram bounds, requires at least 2 retained trials per session, and drops sessions with no good units or no finite tongue values. It does not filter by `obs_intervals` or `free_water`.

ii.
```python
if len(good_idx) == 0:
    return None
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
...
if len(all_y) == 0:
    return None
```

iii. The notes explicitly say the AI wanted a broader trial set than the paper's "regular trials" because early lick and ignore are decoder outputs, and the trajectory shows it investigating `is_good_trials` after verifier warnings. It never documented or implemented the reference `obs_intervals` plus `free_water` filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` for units whose `units/classification` equals `'good'`. Go-cue timestamps provide the alignment anchors.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
...
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
go_times = load_event_times(f, 'go_start_times')
```

iii. The notes say the raw neural source should be spike times from good units only, matching the white paper's quality classifier. The trajectory confirms the AI inspected `units/classification` and `spike_times` specifically for this purpose.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to 50 ms firing rates by first histogramming each good unit across one session-wide global time grid, then slicing out each trial's `[-2.5, 1.5]` window around go cue from that pre-binned matrix. Counts are divided by bin width to obtain Hz.

ii.
```python
session_t0 = float(np.min(go_times) + edges[0])
session_t1 = float(np.max(go_times) + edges[-1])
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The trajectory shows this was an explicit performance optimization after the first sample conversion was too slow. The notes mention that the original nested looping was too slow and that the AI added a session-wide prebinning speedup.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit-level QC uses `classification == 'good'`. In addition, if `units/is_good_trials` exists and its second dimension matches the behavioral trial count, the AI uses it as a per-trial mask and zeros out units marked invalid on that trial. A session with zero good units is dropped.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
...
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
```

```python
if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
...
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The notes justify keeping only classifier-good units to match the papers. The extra `is_good_trials` logic came later from the trajectory, where the AI tried to explain verifier warnings about all-zero neural trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go cue onset. The AI computes trial-window offsets from `go_start_times`, converts them to starting indices on a session-wide 50 ms grid, and slices each trial's neural matrix from that grid.

ii.
```python
go_times = load_event_times(f, 'go_start_times')
...
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
go = float(go_times[i])
start = go + edges[0]
stop = go + edges[-1]
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The notes identify go cue as the required alignment event, and the trajectory shows the AI verifying that `go_start_times/timestamps` were the correct event times. The optimization changed the mechanics of alignment but not the chosen anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins over a 4 s window from -2.5 s to +1.5 s. No additional temporal rebinning is applied after the histogramming step.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

```python
def build_edges(pre, post, bin_size):
    n_bins = int(round((pre + post) / bin_size))
    edges = np.linspace(-pre, post, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers
```

iii. The notes treat the decoder instructions as binding on the `[-2.5, 1.5]` window and 50 ms bins, and the script reflects that directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `trials.start_time`, `trials.stop_time`, and `go_start_times`. For each behavioral trial, it looks for `sample_start_times` events that fall inside that trial interval and uses the first match as the tone time.

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

iii. The trajectory shows the AI initially assuming one `sample_start_times` event per trial, then patching the code after finding extra events in early-lick trials. The notes say `sample_start_times` is the likely tone proxy.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After assigning one sample event to each trial, the AI computes `tone_rel = sample_time - go_time` and returns `bin_centers - tone_rel`, which is numerically the time since the chosen sample event at each go-aligned bin center. Missing sample times propagate as `NaN`.

ii.
```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

iii. The notes say this input should be a time-varying continuous vector derived from the event-time relationship between tone onset and the go-aligned bins. The trajectory does not show any deeper justification beyond fixing the multiple-event issue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-aligned bin centers used to extract neural trial windows, so the time vector shares the neural time axis exactly within the AI's format.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
...
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

```python
start_idx = int(go_bin_start[i])
end_idx = start_idx + len(bin_centers)
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The notes explicitly planned to put this input on the common `bin_centers` axis, and the implementation does that.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trial table fields `photostim_onset` and `photostim_duration`. It does not additionally use `trials.start_time` or `go_start_times` when constructing the binary vector.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
```

iii. In the notes, the AI recognized that photostimulation should come from the raw timing fields and be represented as a time-varying input rather than a trial-level flag. The shipped code kept only the onset and duration fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses `photostim_onset` and `photostim_duration` as floats when present, treats missing values as `NaN`, and marks bins as 1 when their go-aligned centers lie between `onset_rel` and `onset_rel + duration`. Nonstimulated trials remain all zeros.

ii.
```python
def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)
```

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

iii. The notes say the AI wanted a binary time series over the decoder bins and mention that photostimulation should end before go cue in this task. The trajectory does not show any later correction for the coordinate system mismatch.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Alignment is implicit: the AI compares `photostim_onset` and `photostim_duration` directly against the same `bin_centers` array used for the neural trial windows, effectively assuming those onset values are already expressed on the go-aligned axis.

ii.
```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
inp = np.stack([inp0, inp1], axis=0)
```

iii. The notes say photostimulation should be aligned to go cue, but the shipped implementation only reflects that indirectly by placing the binary vector on the shared bin-center axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice primarily from the behavioral event streams `left_lick_times` and `right_lick_times`, together with `go_start_times` and `trials.stop_time` to define the response window. If no lick is found, it falls back to `trial_instruction`.

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

iii. The notes say the AI preferred reconstructing actual choice semantics from lick events and trial annotations rather than assuming instruction equals choice. There is no note justifying the fallback that removes a no-lick class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the AI finds the earliest left and right lick in the response window `[go, stop_time]`; the earlier one sets the class (`0` left, `1` right). If neither side is licked, the instructed side is used instead. The result is repeated across all 80 bins, and only two category names are provided in `output_values`.

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
```

```python
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[0, :] = choice
...
'output_values': [
    ['left', 'right'],
```

iii. The notes describe the desired output as categorical choice and mention reconstructing behavior from raw events. The trajectory focused on getting decoder training to run and did not revisit the missing no-lick category after verification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial table field `outcome`.

ii.
```python
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. The notes identify `outcome` as an explicit trial-table column with values `hit`, `ignore`, and `miss`, so no extra derivation was intended.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps string labels to integers with `{'ignore': 0, 'miss': 1, 'hit': 2}` and repeats the per-trial code across all bins.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out[1, :] = outcome
```

iii. The notes say to use the direct NWB categorical values with a fixed mapping matching the decoder task specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trial table field `early_lick`.

ii.
```python
early = out_early_map[str(trial['early_lick'][i])]
```

iii. The notes identify `early_lick` as an explicit trial annotation and treat it as a direct decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early'` to `0` and `'early'` to `1`, then repeats that value across all bins of the output array.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
...
out[2, :] = early
```

iii. The notes explicitly planned a direct categorical mapping from the NWB strings.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its `timestamps`, using column 1 as the y coordinate. It does not use the tracking likelihood column when constructing the output.

ii.
```python
def load_tongue(f):
    grp = f['acquisition/BehavioralTimeSeries']['Camera0_side_TongueTracking']
    data = np.asarray(grp['data'][()], dtype=float)
    ts = np.asarray(grp['timestamps'][()], dtype=float)
    return ts, data
```

```python
def choose_tongue_y_column(data):
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('Unexpected tongue tracking shape')
    return 1
```

iii. The notes say the AI believed the tracking columns were "likely x/y/likelihood" and that column 1 was probably y. It planned to confirm this from the raw data but the shipped code hard-codes the column choice and omits likelihood handling.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI bins raw tongue y within each retained trial window using the same 50 ms edges as the decoder. It averages all finite y values in each bin, concatenates all finite trial-bin means across the session, computes the 40th and 60th percentiles, and later discretizes each binned y value against those two thresholds. If a session has no finite y values at all, the whole session is dropped.

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
```

```python
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in all_binned_y) else np.array([], dtype=float)
if len(all_y) == 0:
    return None
q40, q60 = np.percentile(all_y, [40, 60])
```

iii. The notes say session-wide percentile discretization was required by the task, but they only tentatively identified the y column and did not document any visibility threshold. The trajectory focuses on getting a plausible per-session discretization rather than matching the reference paper's likelihood-gated workflow.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three categories only: `0` for values below the 40th percentile, `1` for values between the 40th and 60th percentiles inclusive, and `2` for values above the 60th percentile. Nonfinite bins are left at the default class `1`; there is no separate "not visible" class in the stored outputs.

ii.
```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['lt_40pct', '40to60pct', 'gt_60pct'],
],
```

iii. The notes mention a planned 40/60 percentile discretization, but there is no justification in the notes or trajectory for dropping the required fourth "not visible" class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue frames are aligned by selecting the camera samples whose timestamps fall inside each trial's `go + [-2.5, 1.5]` window, subtracting `go` to get go-relative times, and binning those offsets with the same edges used for the neural data.

ii.
```python
start = go + edges[0]
stop = go + edges[-1]
mask = (tongue_ts >= start) & (tongue_ts < stop)
yt = tongue_y[mask]
tt = tongue_ts[mask] - go
...
inds = np.digitize(tt, edges) - 1
```

iii. The notes say behavioral time series should be aligned to go cue using the same common bin centers as the neural data, and the code follows that plan for tongue timing.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some missing values ad hoc. String placeholders such as `'N/A'`, `'nan'`, and `''` are converted to `NaN` for optional numeric fields; sessions with no good units are dropped; sessions with no finite tongue values are dropped; missing brain-region labels fall back to `'unknown'`; and `is_good_trials` is ignored unless its shape matches the trial count. It does not properly remove trials that have no spike coverage, which is why the verifier reports many all-zero neural trials.

ii.
```python
def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)
```

```python
if len(good_idx) == 0:
    return None
...
if len(all_y) == 0:
    return None
...
region = np.array(['unknown' if (x is None or x == 'nan' or x == '') else x for x in region], dtype=object)
```

iii. The notes say integrity failures should lead to exclusion rather than fabrication, but the trajectory shows the AI struggling with the all-zero neural trials and never implementing the reference `obs_intervals` fix. The shipped verification output still contains those warnings.

## 10-a. What are the most time-consuming steps of the code?

i. In the final script, the most time-consuming steps are the per-unit session-wide spike histogramming loop and the per-trial assembly loop that scans sample events, licks, and tongue frames. The notes also show that the earlier, slower version spent most of its time repeatedly binning spikes per trial, which the AI then optimized away.

ii.
```python
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
```

```python
for i in range(n_trials):
    ...
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    ...
    choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
    ...
```

iii. The notes explicitly discuss runtime, first flagging the nested neural binning as too slow and later reporting that the session-wide prebinning optimization reduced runtime to about 1.5 s per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the loop over trials that searches `sample_start_times` by interval membership, the loop over good units that histograms spikes one unit at a time, the per-trial lick-choice reconstruction, and the per-trial tongue binning/discretization loop.

ii.
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
```

```python
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
```

```python
for choice, outcome, early, binned_y in session_trial_output:
    ycat = np.full(len(bin_centers), 1, dtype=np.int64)
    ...
```

iii. The notes acknowledge performance concerns and document one large optimization, but the trajectory does not show similar effort spent on vectorizing the event-matching or tongue-processing loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rescans session-level arrays inside per-trial loops. It searches all sample events against each trial interval, refilters the full left/right lick timestamp arrays for every trial, and rebuilds tongue-frame masks and `np.add.at` accumulators for every trial separately.

ii.
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
```

```python
def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
```

```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
...
np.add.at(sums, inds[ok], yt[ok])
np.add.at(cnts, inds[ok], 1)
```

iii. The trajectory shows the AI optimizing only the most obvious spike-binning bottleneck. The remaining repeated scans were left in place without additional justification.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some work that is not needed for the final dataset: it tracks `valid_trial_ids` but never uses them, computes `global_centers` without consuming them later, supports optional plotting via `maybe_plot`, and stores/logs per-session timing fields that are discarded when assembling the final pickle. It also reconstructs choice from lick events even though outcome plus instruction would have sufficed for the reference label definition.

ii.
```python
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
...
valid_trial_ids = []
...
valid_trial_ids.append(i)
```

```python
def maybe_plot(session_info, outdir='.'):
    import matplotlib.pyplot as plt
    ...
```

```python
info = {
    ...
    'elapsed_sec': time.time() - t0,
}
print(f'processed {session_id}: trials={info["n_trials"]} good_units={info["n_good_units"]} time={info["elapsed_sec"]:.2f}s')
```

iii. The notes mention optional processing plots and runtime tracking for debugging. The trajectory also shows the AI focused on interactive validation and speed measurement, which explains these extra steps even though they do not contribute to the final decoder dataset.
