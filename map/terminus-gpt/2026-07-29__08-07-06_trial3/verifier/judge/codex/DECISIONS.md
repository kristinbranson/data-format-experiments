# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing `data/sub-*/*.nwb`, then opens each file with `h5py.File`. Within each session it separately loads the trial table, behavioral event timestamps, tongue tracking time series, and unit metadata/spike times.

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

iii. In `CONVERSION_NOTES.md` the AI explicitly decided that raw NWB files under subject folders are the source universe, with one session per file. The trajectory shows it viewed the raw NWB layout as the canonical dataset organization and chose direct HDF5 access rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. The AI uses `general/subject/subject_id` from each NWB file as the subject id, then forms the output `subjects` list from the sorted unique ids and maps sessions with `subject_idx`.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` says subject ids should come from NWB subject metadata. The trajectory also notes that raw data are organized by subject folders, but the AI treats the NWB subject field as the actual identifier carried into the output.

## 1-c. How are the data split into sessions?

i. The AI treats each `.nwb` file as one session. It uses the filename stem as the session id and preserves the sorted file order in the output.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
session_id = path.stem
...
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
    if info is not None:
        sessions.append(info)
```

iii. In `CONVERSION_NOTES.md` the AI wrote that each session is a single NWB file under `data/sub-<subject_id>/`. The trajectory repeatedly describes the dataset as “174 NWB session files,” so the file boundary is the session boundary in its reasoning.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table rows as trials. It asserts that `go_start_times` has the same length as `trial['start_time']`, then iterates over `range(n_trials)` and builds one neural/input/output item per kept trial.

ii.
```python
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
...
for i in range(n_trials):
    go = float(go_times[i])
    ...
    session_trial_neural.append(trial_mat)
    session_trial_input.append(inp)
    session_trial_output.append([choice, outcome, early, binned_y])
```

iii. The AI’s notes say trial metadata live in `intervals/trials` and that behavioral event streams should be aligned against trial structure. In the trajectory it treats the equality of trial-table length and `go_start_times` length as confirmation that one trial-table row corresponds to one behavioral trial.

## 1-e. How are trials filtered based on quality controls?

i. The final code does not reproduce the reference trial curation. It keeps most trials, only skipping trials whose aligned window begins before time 0, trials whose sliced neural window would fall outside the pre-binned session array, and trials with no valid units if `units/is_good_trials` exists with a matching shape. It does not filter by `obs_intervals` or `free_water`.

ii.
```python
is_good_trials = np.asarray(f['units']['is_good_trials'][()], dtype=bool) if ('is_good_trials' in f['units'] and f['units']['is_good_trials'].shape[1] == len(trial['start_time'])) else None
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
```

iii. The AI’s notes initially flagged a tension between the paper’s “regular trial” masks and the decoder task, and in the trajectory it later justified extra filtering by suspected neural-validity issues (`is_good_trials`, `obs_intervals`) after seeing all-zero neural warnings. The final code reflects that troubleshooting path, not the reference filtering rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/classification`, `units/spike_times`, `units/spike_times_index`, and the behavioral `go_start_times` event stream.

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

iii. In `CONVERSION_NOTES.md` the AI explicitly chose to use only units labeled `classification == 'good'`, matching the paper’s curation. The same notes map neural data to spike times binned around go-cue timestamps.

## 2-b. How is the `neural` data processed?

i. The AI pre-bins each good unit’s spikes over a session-wide global 50 ms grid using `np.histogram`, converts counts to Hz, and then slices 80-bin trial windows from that global firing-rate matrix.

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

iii. The AI’s notes say it added a speedup by “pre-bin[ning] all good-unit spikes once over the session timeline, then slice trial windows.” The trajectory also shows it changed to this approach after the first sample run looked too slow.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first filtered to `classification == 'good'`. In addition, if `units/is_good_trials` exists and matches the behavioral trial count, the AI zeroes out units deemed invalid on a given trial; sessions with no good units are dropped.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
...
if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
...
if is_good_trials is not None:
    trial_mat[~valid_units, :] = 0.0
```

iii. The AI’s notes justify `classification == 'good'` from the QC white paper. The extra `is_good_trials` handling came later from the trajectory, where the AI tried to explain verification warnings about all-zero neural trials and treated `is_good_trials` as a possible per-trial validity mask.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI intends to align to go-cue onset by using a common `edges` array spanning `[-2.5, 1.5]` around each trial’s `go_times[i]`. In practice it converts go times to rounded starting indices into the global session grid and slices a fixed-length window from there.

ii.
```python
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
...
for i in range(n_trials):
    go = float(go_times[i])
    start = go + edges[0]
    stop = go + edges[-1]
    ...
    start_idx = int(go_bin_start[i])
    end_idx = start_idx + len(bin_centers)
    ...
    trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. `CONVERSION_NOTES.md` states the conversion should use go-cue alignment with a `[-2.5, +1.5]` s window. The trajectory confirms the AI believed the global pre-binning optimization still preserved go-aligned trial windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins across a 4 s window, yielding 80 time bins per trial. It does not apply an additional downstream rebinning step after converting spikes to 50 ms firing-rate bins.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

iii. This is explicitly documented in `CONVERSION_NOTES.md` as one of the core mapping decisions: go-cue alignment, `[-2.5, +1.5]`, and 50 ms bins to satisfy the decoder task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `trial['start_time']`, `trial['stop_time']`, and `go_times`. It searches for sample events that fall inside each trial interval and stores one selected sample time per trial.

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

iii. The trajectory shows that the AI originally assumed `sample_start_times` was already one-per-trial, then changed to interval-based assignment after discovering the count mismatch. Its notes identify `sample_start_times` as the likely tone proxy.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI takes the first `sample_start_times` event that falls within the trial interval, subtracts the go-cue time to get `tone_rel`, then computes `bin_centers - tone_rel`, which equals bin-center time measured relative to that chosen sample event.

ii.
```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
...
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```

iii. In step 27 of the trajectory, the AI explicitly justifies this as a repair: because `sample_start_times` was not one-per-trial, it decided to “derive tone onset per trial by assigning event times to trials via interval membership.” No further justification is recorded for picking the first event in the interval.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns this input to the same per-trial 80-bin go-relative grid used for neural data, storing one continuous value per bin.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
...
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
inp = np.stack([inp0, inp1], axis=0)
```

iii. `CONVERSION_NOTES.md` says decoder inputs should be represented on the common trial-aligned time axis, and the AI consistently uses the same `bin_centers` array for both neural windows and input features.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trial-table fields `photostim_onset` and `photostim_duration`. It does not convert `photostim_onset` from trial-start-relative time into go-relative time using `trial['start_time']` and `go`.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i]) if 'photostim_onset' in trial else np.nan
ps_dur = parse_optional_float(trial['photostim_duration'][i]) if 'photostim_duration' in trial else np.nan
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. The AI’s notes correctly identified the raw photostim fields in the trial table. The trajectory does not show a later correction to rebase onset times, so the final code reflects the simplifying assumption that the stored onset could be used directly.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `photostim_onset` is parsed as an optional float and paired with `photostim_duration`. A binary vector is then created by marking bins whose centers fall between `onset_rel` and `onset_rel + duration`.

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

iii. In `CONVERSION_NOTES.md` the AI wrote that photostim should become a time-varying binary input. The trajectory supports that intent, but there is no explicit justification in the notes for omitting the trial-start-to-go conversion.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation by comparing the chosen onset and offset directly to the same `bin_centers` array used for neural data, effectively treating the stored onset as already expressed on the go-relative axis.

ii.
```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
inp = np.stack([inp0, inp1], axis=0)
```

iii. The notes say photostimulation should be “aligned to go cue” and represented as a binary time series, but the only explicit alignment mechanism in the final code is direct comparison to `bin_centers`.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice primarily from `left_lick_times`, `right_lick_times`, `go_times`, and `trial['stop_time']`. If no lick is found in the response window, it falls back to `trial['trial_instruction']`.

ii.
```python
def find_choice_from_licks(left_licks, right_licks, go_time, stop_time):
    l = left_licks[(left_licks >= go_time) & (left_licks <= stop_time)]
    r = right_licks[(right_licks >= go_time) & (right_licks <= stop_time)]
    ...
    if tl == np.inf and tr == np.inf:
        return None
    return 0 if tl < tr else 1
...
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. `CONVERSION_NOTES.md` records uncertainty about whether choice should mean instructed side or actual lick side and says to prefer “actual choice semantics” if possible. The final code follows that preference by using lick events, with instruction as a fallback when no lick is observed.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI assigns choice `0` if the first post-go left lick occurs before the first right lick and `1` otherwise. If there is no lick at all in the response window, it imputes the instructed side instead of using a separate “no lick” class. The resulting scalar is repeated across all 80 bins.

ii.
```python
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
...
out = np.zeros((4, len(bin_centers)), dtype=np.int64)
out[0, :] = choice
```

iii. The notes justify using actual lick direction as the preferred choice semantics. There is no explicit written justification for collapsing no-lick trials into the instructed side, but that is the behavior of the final code and of the two-value `output_values` it writes for choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table column `outcome`.

ii.
```python
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. The AI’s notes state that raw NWB `outcome` values are already `ignore`, `miss`, and `hit`, so outcome can be mapped directly from the trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the per-trial value across all 80 bins.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = out_outcome_map[str(trial['outcome'][i])]
...
out[1, :] = outcome
```

iii. This mapping is one of the explicit “Key Decisions” in `CONVERSION_NOTES.md`, where the AI says outcome should use the direct NWB categories and the requested 0/1/2 coding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table column `early_lick`.

ii.
```python
early = out_early_map[str(trial['early_lick'][i])]
```

iii. The notes explicitly state that `early_lick` can be mapped directly from NWB trial annotations.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early -> 0` and `early -> 1`, then repeats the per-trial label across all 80 bins.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
early = out_early_map[str(trial['early_lick'][i])]
...
out[2, :] = early
```

iii. This is also stated directly in the notes as a fixed categorical mapping from NWB string labels to the requested decoder codes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI uses `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically `timestamps` and column 1 of `data`, as tongue y-position. It assumes column 1 is y and does not use the likelihood column.

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
...
tongue_ts, tongue_data = load_tongue(f)
ycol = choose_tongue_y_column(tongue_data)
tongue_y = tongue_data[:, ycol]
```

iii. In `CONVERSION_NOTES.md` the AI wrote that the tongue series was “likely x/y/likelihood” and that the y-position was “likely second column.” The trajectory confirms this remained an inferred convention rather than a verified schema decision.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each kept trial, the AI selects camera frames in the `[-2.5, 1.5]` s go-aligned window, bins all finite y-values into the 50 ms bins by simple averaging, and stores the resulting continuous `binned_y`. It then pools all finite trial-bin values from the session to compute percentile thresholds.

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
all_binned_y.append(binned_y)
```

iii. The notes describe tongue y as something to “interpolate/assign” to decoder bins and discretize per session. There is no note justifying omission of the tracking-likelihood filter; the code reflects a simpler bin-and-average implementation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI computes the 40th and 60th percentiles over all finite `binned_y` values pooled from all kept trial bins in the session. Each trial/bin is then coded as `0` if below `q40`, `2` if above `q60`, and `1` otherwise. Bins with missing tongue data remain at the default class `1`; there is no separate “not visible” class.

ii.
```python
all_y = np.concatenate([x[np.isfinite(x)] for x in all_binned_y if np.any(np.isfinite(x))]) if any(np.any(np.isfinite(x)) for x in all_binned_y) else np.array([], dtype=float)
...
q40, q60 = np.percentile(all_y, [40, 60])
...
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
finite = np.isfinite(binned_y)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
ycat[finite & (binned_y >= q40) & (binned_y <= q60)] = 1
```

iii. `CONVERSION_NOTES.md` says the task requires 40th/60th percentile discretization per session. The final code follows that high-level instruction but, based on the trajectory and notes, does not record any explicit justification for the missing-data default class or for taking percentiles over pooled trial bins rather than session-wide bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue values are aligned to the same go-relative trial window and 50 ms bin edges used for neural data. Frame timestamps are shifted by `go` and digitized into the shared `edges`.

ii.
```python
start = go + edges[0]
stop = go + edges[-1]
...
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
...
inds = np.digitize(tt, edges) - 1
```

iii. The AI’s notes say video-derived outputs should be aligned to the common go-cue-centered decoder axis. The final code uses the same `edges` array for both neural and tongue binning.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some missing values defensively but not in the same way as the reference. It decodes byte strings, converts `'N/A'` photostim entries to `NaN` and then to all-zero photostim vectors, drops sessions with no good units or with no finite tongue values at all, skips a trial if its pre-go window starts before time 0 or falls outside the global pre-binned grid, and leaves `time_from_tone_onset` as `NaN` if no sample event is found. Missing per-bin tongue data is not given a separate category; it defaults to class `1`.

ii.
```python
def parse_optional_float(x):
    if isinstance(x, str):
        if x in ('N/A', 'nan', ''):
            return np.nan
        return float(x)
    return float(x)
...
if len(good_idx) == 0:
    return None
...
if start < 0:
    continue
...
if start_idx < 0 or end_idx > global_rates.shape[1]:
    continue
...
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
...
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
```

iii. The trajectory shows that many of these checks were added pragmatically while debugging format warnings and raw-data irregularities. The notes frame them as integrity checks or missing-data guards rather than as part of a paper-matched processing decision.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps in the AI code are opening every NWB/HDF5 session, reading the full spike-time and tongue-tracking arrays, pre-binning each good unit over the session-wide grid with `np.histogram`, and then looping over all trials again to slice windows and bin tongue data.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    spike_times = f['units']['spike_times'][()]
    spike_index = f['units']['spike_times_index'][()]
...
global_rates = np.zeros((len(good_idx), len(global_centers)), dtype=np.float32)
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
...
for i in range(n_trials):
    ...
```

iii. `CONVERSION_NOTES.md` first says the initial nested unit-by-trial histogram approach was too slow, then says the pre-binning speedup reduced runtime to about 1.5 s/session on the sample estimate. The trajectory also explicitly identifies the unit-binning path as the main performance bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorizable loops are: scanning all `sample_start_times` for each trial, histogramming spikes once per good unit, looping over trials to extract and bin tongue values, and looping again over trials to build final discrete output arrays.

ii.
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
    if len(hits):
        sample_times[i] = hits[0]
...
for jj, unit_i in enumerate(good_idx):
    st = get_ragged_row(spike_times, spike_index, int(unit_i))
    counts, _ = np.histogram(st, bins=global_edges)
...
for i in range(n_trials):
    ...
for choice, outcome, early, binned_y in session_trial_output:
    ...
```

iii. The AI’s notes explicitly mention unit loops as a speed concern. The final code still contains several additional Python-level loops that were not eliminated after that optimization pass.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several passes over related information: it scans the sample-event vector separately for every trial, makes one pass over trials to build continuous `binned_y` arrays and collect them into `all_binned_y`, then makes a second pass over stored trial outputs to discretize tongue y and assemble final output matrices. It also computes some trial bookkeeping such as `valid_trial_ids` that is never used later.

ii.
```python
sample_times = np.full(n_trials, np.nan, dtype=float)
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) & (sample_event_times <= trial['stop_time'][i])]
...
all_binned_y.append(binned_y)
...
final_outputs = []
for choice, outcome, early, binned_y in session_trial_output:
    ...
    final_outputs.append(out)
...
valid_trial_ids.append(i)
```

iii. The trajectory does not present this as an explicit design choice; it is simply how the final implementation ended up after iterative debugging and optimization.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and discards `global_centers` and `valid_trial_ids`, keeps an unused `show_processing` argument in `process_session`, and optionally generates debugging plots unrelated to the saved dataset. Those calculations do not contribute to `converted_data.pkl`.

ii.
```python
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
...
valid_trial_ids = []
...
valid_trial_ids.append(i)
...
def process_session(path, edges, bin_centers, show_processing=False):
    ...
```

iii. The trajectory shows these pieces arose during development and debugging. They were not called out in the notes as intended dataset features, and they do not affect the final saved analysis structure.
