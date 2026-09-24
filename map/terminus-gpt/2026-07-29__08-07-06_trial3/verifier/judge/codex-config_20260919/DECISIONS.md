# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every `data/sub-*/*.nwb` file, sorts the paths, and treats each as a session. It opens each file once with `h5py`, reads trials, events, tongue tracking, unit metadata, and ragged spike arrays, then appends surviving sessions.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
for p in files:
    info = process_session(p, edges, bin_centers, show_processing=args.show_processing)
```
```python
with h5py.File(path, 'r') as f:
    trial = load_trial_table(f)
    go_times = load_event_times(f, 'go_start_times')
    spike_times = f['units']['spike_times'][()]
```

iii. The notes state that the data contain 174 NWB files arranged by subject and session, and that raw NWB is the source universe. The agent chose a single-pass load to avoid repeated file opens.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each file's `general/subject/subject_id`. After conversion, unique IDs are sorted and each session is mapped to its subject index.

ii.
```python
subject = f['general']['subject']['subject_id'][()]
if isinstance(subject, bytes):
    subject = subject.decode()
```
```python
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes identify the NWB subject field as the source and report 28 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; the filename stem is its session ID. Sessions with no good units, fewer than two retained trials, or no finite binned tongue values are dropped.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    session_id = path.stem
```
```python
if len(good_idx) == 0:
    return None
if len(session_trial_neural) < 2:
    return None
if len(all_y) == 0:
    return None
```

iii. The notes expected the 174 raw files to reduce to the paper's 173 sessions through validity/curation checks; the full output indeed contains 173 sessions.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials and are positionally paired with `go_start_times`; the code asserts equal counts. It then loops through every trial row and builds one neural/input/output item for each retained row.

ii.
```python
n_trials = len(trial['start_time'])
assert len(go_times) == n_trials, (len(go_times), n_trials)
for i in range(n_trials):
    go = float(go_times[i])
```

iii. The notes say behavioral event timestamps, especially the go cue, should define alignment while the NWB trials table supplies trial annotations.

## 1-e. How are trials filtered based on quality controls?

i. A trial is skipped if its requested window begins before time zero, falls outside the precomputed global rate array, or has no good unit marked valid by `units/is_good_trials`. Invalid units within an otherwise retained trial are zero-filled. The code does not explicitly exclude `free_water` trials or match trials to `obs_intervals`.

ii.
```python
if start < 0:
    continue
if is_good_trials is not None:
    valid_units = is_good_trials[good_idx, i]
    if not np.any(valid_units):
        continue
...
trial_mat[~valid_units, :] = 0.0
```

iii. The notes planned to retain task-relevant early-lick, ignore, and stimulation trials, and to exclude only sessions/trials failing validity checks. They recognized the reference regular-trial filter but intentionally needed broader trials for requested outputs. They did not document why `is_good_trials` should replace `obs_intervals`/`free_water` filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from ragged `units/spike_times` and `units/spike_times_index`, restricted by `units/classification == 'good'`; go-cue times define trial extraction.

ii.
```python
cls = decode_arr(f['units']['classification'][()])
good_idx = np.flatnonzero(cls == 'good')
spike_times = f['units']['spike_times'][()]
spike_index = f['units']['spike_times_index'][()]
```

iii. The notes identify spike times plus the published classifier verdict as the correct curated neural source, matching the paper's roughly 69,943 good units.

## 2-b. How is the `neural` data processed?

i. Each good unit's spikes are histogrammed on one session-wide 50 ms grid, divided by 0.05 s to produce Hz, and trial matrices are sliced from that global array. Invalid unit/trial combinations are set to zero. There is no smoothing or normalization.

ii.
```python
global_edges = np.arange(session_t0, session_t1 + bin_size * 1.0001, bin_size)
for jj, unit_i in enumerate(good_idx):
    counts, _ = np.histogram(st, bins=global_edges)
    global_rates[jj] = counts.astype(np.float32) / bin_size
trial_mat = global_rates[:, start_idx:end_idx].copy()
```

iii. The agent says reference session dictionaries contain pre-binned firing rates and chose global pre-binning as a speed optimization, reducing runtime to about 1.5 seconds/session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled exactly `good` in `units/classification` are included. Additionally, `is_good_trials` can zero a unit for particular trials; trials with no valid good units are dropped. A session with no classifier-good units is dropped.

ii.
```python
good_mask = cls == 'good'
good_idx = np.flatnonzero(good_mask)
if len(good_idx) == 0:
    return None
```
```python
valid_units = is_good_trials[good_idx, i]
trial_mat[~valid_units, :] = 0.0
```

iii. The notes justify `classification == 'good'` as the region-specific 15-metric classifier used by the paper. They do not justify the extra per-trial zero filling.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The intended window is -2.5 to +1.5 seconds around each go cue. However, spikes are first put on a grid anchored to the earliest go cue, and each later trial start is rounded to the nearest grid index rather than binned using edges anchored exactly to that trial's go cue.

ii.
```python
session_t0 = float(np.min(go_times) + edges[0])
go_bin_start = np.rint((go_times + edges[0] - session_t0) / bin_size).astype(int)
start_idx = int(go_bin_start[i])
trial_mat = global_rates[:, start_idx:start_idx + len(bin_centers)].copy()
```

iii. The notes correctly state that go-cue timestamps are the anchor. The global-grid implementation was introduced solely as a runtime optimization and the notes claim sample verification preserved correctness, but format verification did not test exact temporal alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have 80 non-overlapping 50 ms bins across four seconds. Raw spike times are binned directly to this resolution; no later temporal rebinning is applied.

ii.
```python
pre = 2.5
post = 1.5
bin_size = 0.05
edges, bin_centers = build_edges(pre, post, bin_size)
```

iii. This directly follows the requested 50 ms bins and the notes report all sessions have 80 time bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial `start_time`/`stop_time`, each trial's go cue, and decoder bin centers. The first sample event within the trial boundaries is selected.

ii.
```python
hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                          (sample_event_times <= trial['stop_time'][i])]
if len(hits):
    sample_times[i] = hits[0]
```

iii. The notes identify `sample_start_times` as the tone/sample epoch but acknowledge that exact event mapping needed determination. The final code chose the first event in the trial without documenting the early-lick replay implication.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is converted to a go-relative offset, then subtracted from every go-relative bin center. If no sample event was found, the entire input is NaN.

ii.
```python
tone_rel = float(sample_times[i] - go) if np.isfinite(sample_times[i]) else np.nan
inp0 = build_time_from_tone(bin_centers, tone_rel).astype(np.float32)
```
```python
def build_time_from_tone(bin_centers, tone_time_rel):
    return bin_centers - tone_time_rel
```

iii. The notes describe a continuous time-varying vector computed as bin time minus tone onset, which the code implements.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the nominal go-relative centers of the 80 neural bins. Thus its array shape and nominal index alignment match neural, although the neural spike grid itself can be shifted by rounding.

ii.
```python
edges, bin_centers = build_edges(pre, post, bin_size)
inp0 = build_time_from_tone(bin_centers, tone_rel)
inp = np.stack([inp0, inp1], axis=0)
```

iii. The agent intended all streams to share the go-cue-aligned `bin_centers` axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset` and `photostim_duration`, parsed from strings/numbers, plus nominal go-relative bin centers.

ii.
```python
ps_on = parse_optional_float(trial['photostim_onset'][i])
ps_dur = parse_optional_float(trial['photostim_duration'][i])
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
```

iii. The notes correctly identify these raw fields and plan a binary time-varying stimulation input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Missing or nonfinite timing produces all zeros. Otherwise bins with centers in `[onset, onset + duration)` are marked one. The code does not convert the onset from trial-start-relative time to go-cue-relative time.

ii.
```python
off = onset_rel + duration
x[(bin_centers >= onset_rel) & (bin_centers < off)] = 1.0
```

iii. The notes say stimulation should be built from raw timing fields and aligned to the go cue, noting it should occur before go. They do not justify omitting the required trial-start-to-go conversion.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The binary vector has the same 80 nominal bin centers as neural, but raw onset is compared directly to go-relative centers even though it is stored relative to trial start. Therefore its temporal values are misaligned (and commonly all zero).

ii.
```python
inp1 = build_photostim_vector(bin_centers, ps_on, ps_dur)
inp = np.stack([inp0, inp1], axis=0)
```

iii. The planned decision was go-cue alignment, but sample validation only observed all-zero stimulation and deferred checking it on broader data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is primarily derived from `left_lick_times` and `right_lick_times`: the first lick between go cue and trial stop determines left/right. If neither exists, it falls back to `trial_instruction`.

ii.
```python
choice = find_choice_from_licks(left_licks, right_licks, go, float(trial['stop_time'][i]))
if choice is None:
    instr = str(trial['trial_instruction'][i])
    choice = 0 if instr == 'left' else 1
```

iii. The notes explicitly preferred actual lick direction over instructed side where possible, but did not address the required no-lick category.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earliest response-window left/right lick becomes 0/1. Ties favor right. No-lick trials are incorrectly assigned the instructed side. The scalar is repeated over all bins and only two output labels are declared.

ii.
```python
if tl == np.inf and tr == np.inf:
    return None
return 0 if tl < tr else 1
```
```python
out[0, :] = choice
...
['left', 'right'],
```

iii. The notes say choice should reflect actual choice semantics, but their sample checks only report two classes and treat that as acceptable.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` field.

ii.
```python
outcome = out_outcome_map[str(trial['outcome'][i])]
```

iii. The notes identify the explicit NWB outcome strings as the correct source.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped `ignore=0`, `miss=1`, `hit=2`, and the per-trial value is repeated across all 80 bins.

ii.
```python
out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out[1, :] = outcome
```

iii. This mapping is explicitly planned in the notes and follows the requested category order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` field.

ii.
```python
early = out_early_map[str(trial['early_lick'][i])]
```

iii. The notes identify the NWB early-lick annotation as the direct source.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Strings are mapped `no early=0`, `early=1`, and the value is repeated across all bins.

ii.
```python
out_early_map = {'no early': 0, 'early': 1}
out[2, :] = early
```

iii. This direct binary mapping is documented in the planning notes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and column 1 of `Camera0_side_TongueTracking/data`. Although the source has x, y, and likelihood, likelihood is not used.

ii.
```python
tongue_ts, tongue_data = load_tongue(f)
ycol = choose_tongue_y_column(tongue_data)
tongue_y = tongue_data[:, ycol]
```

iii. The notes infer the second column is y in `[x, y, likelihood]`, but do not decide to use likelihood to determine visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Raw y samples in each trial window are assigned to 50 ms bins and averaged. All finite y values are accepted regardless of tracking likelihood. Session percentiles are computed from finite binned values across retained trial windows. A session with no finite values is dropped.

ii.
```python
inds = np.digitize(tt, edges) - 1
ok = (inds >= 0) & (inds < len(bin_centers)) & np.isfinite(yt)
np.add.at(sums, inds[ok], yt[ok])
binned_y[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
```

iii. The notes planned per-session percentiles over valid y values and bin assignment/interpolation. They did not investigate confidence filtering or distinguish a retracted/invisible tongue from a valid coordinate.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles of all finite retained trial-bin means define classes 0, 1, and 2. Missing bins are initialized to class 1, so there is no required class 3 (`not visible`). Boundary values equal to either percentile are class 1.

ii.
```python
q40, q60 = np.percentile(all_y, [40, 60])
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
ycat[finite & (binned_y < q40)] = 0
ycat[finite & (binned_y > q60)] = 2
```

iii. The agent followed the per-session 40/60 percentile instruction for visible numeric values, but overlooked the explicit fourth `not visible` category; the validation summary therefore showed only three classes.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples from `[go-2.5, go+1.5)` are shifted by the go time and digitized using the nominal go-relative 50 ms edges. Index alignment matches the requested grid, though the separately global-binned neural data can be offset by rounding.

ii.
```python
mask = (tongue_ts >= start) & (tongue_ts < stop)
tt = tongue_ts[mask] - go
inds = np.digitize(tt, edges) - 1
```

iii. The notes intended event-timestamp alignment of all streams to the go cue and 50 ms bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Byte strings are decoded; optional numeric strings such as `N/A` become NaN; missing tone yields NaN input; no-good-unit, too-few-trial, and no-finite-tongue sessions are dropped. Missing tongue bins become middle class 1. Missing per-unit trial validity becomes zeros, and no-lick choice falls back to instruction.

ii.
```python
if x in ('N/A', 'nan', ''):
    return np.nan
```
```python
ycat = np.full(len(bin_centers), 1, dtype=np.int64)
trial_mat[~valid_units, :] = 0.0
```

iii. The notes frame these as integrity checks, but give no principled justification for imputing invisible tongue as middle position, invalid neural recordings as zero firing, or absent choice as instructed choice.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large spike/video arrays and histogramming every good unit over a session are the principal conversion costs. Full-dataset serialization and decoder training are also costly outside `process_session`.

ii.
```python
spike_times = f['units']['spike_times'][()]
for jj, unit_i in enumerate(good_idx):
    counts, _ = np.histogram(st, bins=global_edges)
```

iii. The notes measured an optimized runtime of about 1.5 seconds/session and describe avoidance of repeated file opens and global pre-binning as the main speedups.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial search for sample events repeatedly masks the full sample-event array and could use `searchsorted`. Trial tongue binning and the second per-trial output construction pass could be partially vectorized. The unit loop is difficult to remove because spike rows are ragged, although edge lookup can be vectorized across trials per unit.

ii.
```python
for i in range(n_trials):
    hits = sample_event_times[(sample_event_times >= trial['start_time'][i]) &
                              (sample_event_times <= trial['stop_time'][i])]
```
```python
for choice, outcome, early, binned_y in session_trial_output:
    ...
```

iii. The notes identify unit/trial loops as the initial bottleneck and claim NumPy histogramming/global pre-binning made runtime acceptable, but do not critically inventory the remaining loops.

## 10-c. What processing does the code repeat multiple times?

i. It performs two passes over retained trials: first to compute/store binned tongue values and provisional outputs, then to threshold tongue values and rebuild final output arrays. Outcome/early mapping dictionaries are recreated on every trial. Sample-event filtering scans the event array once per trial.

ii.
```python
for i in range(n_trials):
    out_outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
    out_early_map = {'no early': 0, 'early': 1}
```
```python
for choice, outcome, early, binned_y in session_trial_output:
    final_outputs.append(out)
```

iii. The notes do not discuss these repeated operations; they focus on eliminating repeated file opens and repeated spike histogramming.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It bins spikes across the entire interval between the earliest and latest requested window even though inter-trial portions and excluded trials are discarded. `global_centers` and `valid_trial_ids` are computed but never used. The `show_processing` argument to `process_session` is unused.

ii.
```python
global_centers = (global_edges[:-1] + global_edges[1:]) / 2
valid_trial_ids = []
...
valid_trial_ids.append(i)
```

iii. The global timeline was justified as a speed optimization, but the notes do not identify its unused bins/variables or discuss the tradeoff.
