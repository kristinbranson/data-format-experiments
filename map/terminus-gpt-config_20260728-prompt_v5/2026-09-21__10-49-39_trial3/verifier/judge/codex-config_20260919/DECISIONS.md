# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every NWB file under `/app/data`, sorts the paths, and opens each file once with `h5py`. It directly reads the trials and units HDF5 groups and relevant acquisition datasets. `--sample` restricts processing to the first two files; otherwise all 174 files are attempted.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
if args.sample:
    files = files[:2]
...
with h5py.File(path, 'r') as f:
    trial_table = load_trial_table(f)
    units = load_units_table(f)
```

iii. The notes justify direct `h5py` reads as faster than loading full pynwb objects and report 28 subjects, 174 NWB session files, and 94,990 raw trials. The agent says this covers the subject-folder/session-file organization of the dataset.

## 1-b. How are the data split into subjects?

i. A subject is inferred from the parent directory name of each NWB file (for example, `sub-440956`). Subjects are accumulated in first-encounter order, and each retained session receives an index into that list.

ii.
```python
subj = path.parent.name
...
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. The notes state that the directory layout has one folder per subject and that 28 such folders were detected. They planned to use the folder or NWB metadata; the implementation chose the folder name.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A session is appended only if at least two trials survive processing; its position in the output lists follows sorted file order.

ii.
```python
for f in files:
    sn, si, so, subj, regions = process_session(f, show_processing=args.show_processing)
    if len(sn) < 2:
        continue
    neural.append(sn)
    input_data.append(si)
    output_data.append(so)
```

iii. The notes identify the NWB files as session-level files. They observed 174 raw sessions versus 173 in the paper but retained all 174 because their QC fallback allowed the nominally uncurated session to survive.

## 1-d. How are the data split into trials?

i. Trial rows supply trial start/stop times, while `go_start_times` supplies one alignment time per trial. The code iterates over every go-cue index and constructs one neural/input/output item, unless its neural slice is out of bounds or entirely zero. It assumes trial rows and go-cue arrays correspond by position and does not assert equal length.

ii.
```python
go_times = infer_go_cue_times(trial_table, f)
...
for i, go in enumerate(go_times):
    ...
    if start_idx < 0 or end_idx > session_fr.shape[1]:
        continue
    ...
    if np.allclose(trial_mats, 0):
        continue
    session_neural.append(trial_mats)
```

iii. The notes say switching to the actual `go_start_times` fixed initial alignment problems. They regard zero-neural windows as invalid trials and report 90,999 retained trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed only when the requested neural slice is outside the pre-binned session array or every retained unit has zero activity throughout the window. Sessions with fewer than two surviving trials are removed. The code does not use `units/obs_intervals` or explicitly exclude `free_water` trials. The all-zero check is accidentally duplicated.

ii.
```python
if start_idx < 0 or end_idx > session_fr.shape[1]:
    continue
...
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
```

iii. The notes say all-zero trials caused verifier warnings and were therefore dropped. They deliberately retain early-lick and no-response trials because those are requested decoder targets, but do not document the reference solution's precise `obs_intervals` and `free_water` curation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged `units/spike_times` array and its `spike_times_index`, restricted by a selected good-unit mask. Trial placement uses trial start/stop times and go-cue timestamps.

ii.
```python
st = units_group['spike_times']
ind = units_group['spike_times_index'][:]
...
st = get_spike_times_for_unit(f['units'], int(u))
counts, _ = np.histogram(st[m], bins=session_edges)
```

iii. The notes identify spike times and curated unit labels as the proper raw sources and state that a manual spot-check against raw spike times matched for one neuron/trial.

## 2-b. How is the `neural` data processed?

i. For speed, each good unit's spikes are histogrammed once over a session-wide 50-ms grid and divided by 0.05 to produce Hz. Each trial is then extracted by rounding the desired go-relative start to the nearest session-grid index. No smoothing or normalization is applied.

ii.
```python
session_edges = np.arange(session_start, session_stop + BIN_SIZE, BIN_SIZE)
...
counts, _ = np.histogram(st[m], bins=session_edges)
session_fr[j] = counts.astype(np.float32) / BIN_SIZE
...
start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The notes justify pre-binning as a major speedup (about 38.7 s to 1.37 s on sample sessions). They treat it as equivalent to direct go-aligned binning, based on a single spot-check.

## 2-c. How is the `neural` data filtered based on quality controls?

i. `choose_good_units` prefers any recognized good-like value in `classification`, then tries several alternate label or boolean columns, then applies generic metric thresholds. If the metric mask is empty, it retains every unit. This keeps 70,654 units across 174 sessions rather than dropping the session without valid classifier labels.

ii.
```python
for key in ['classification', 'unit_quality', 'quality', 'label', 'cluster_quality']:
    ...
    if np.isin(sval, list(good_words)).any():
        return np.isin(sval, list(good_words))
...
if mask.sum() == 0:
    mask = np.ones(n, dtype=bool)
```

iii. The notes correctly identify paper QC as the classifier's `good` label, but the implementation adds defensive fallbacks. The notes call 70,654 “close” to the reported 69,943 and leave the one-session discrepancy unresolved.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The intended window is -2.5 to +1.5 seconds around the raw go cue. However, the code extracts from a session-wide grid by a rounded index, so trial edges can differ from exact go-relative edges by up to half a bin.

ii.
```python
start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
end_idx = start_idx + N_BINS
trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The agent says all streams share the actual go-cue clock and reports that replacing heuristic go-cue inference eliminated warnings. It did not acknowledge the residual phase error introduced by the session-grid extraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins spanning four seconds. Raw spikes are histogrammed directly into those bins (via a session-wide intermediate grid); there is no additional temporal rebinning.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
```

iii. This directly follows the decoder instructions. The notes and README consistently report 50 ms and 80 timepoints.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times/timestamps`, trial start/stop times, and go-cue timestamps. When multiple sample events fall within a trial, it selects the first; trial start is the fallback.

ii.
```python
sample_starts = np.asarray(trial_table['start_time'], dtype=float)
...
cand = sample_event_times[(sample_event_times >= start_times[i]) &
                          (sample_event_times <= stop_times[i])]
if len(cand):
    sample_starts[i] = cand[0]
```

iii. The notes say sample/tone event timestamps should be used. They mention that multiple sample tones may exist but do not justify choosing the first rather than the last sample onset before go after an early-lick replay.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each nominal go-relative bin center is shifted by `go_time - sample_start`, producing seconds elapsed since the chosen tone onset.

ii.
```python
def build_time_from_tone_vector(sample_start, go_time):
    return (BIN_CENTERS - (sample_start - go_time)).astype(np.float32)[None, :]
```

iii. The notes describe this as a continuous, time-varying input built from event alignment; no other transform is applied.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Its values use nominal bin centers at -2.475, -2.425, ..., +1.475 seconds relative to go. These have the intended index correspondence with neural bins, although the neural bins themselves may be shifted slightly by rounded extraction from the session grid.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
inp0 = build_time_from_tone_vector(sample_starts[i], go)
```

iii. The agent states all time-varying streams use the same go-aligned 50-ms grid, but does not account for the neural grid's rounding offset.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primarily it uses the trials-table `photostim_onset`, `photostim_duration`, and `start_time`. If those columns are absent, it falls back to behavioral-event photostim start/stop timestamps.

ii.
```python
onset = safe_float(trial_table['photostim_onset'][i])
dur = safe_float(trial_table['photostim_duration'][i])
abs_on = start_times[i] + onset
...
ps_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
```

iii. The notes say the initial event-only implementation yielded zeros and that interpreting trial-table onset as trial-relative fixed it. A spot-check then matched the raw fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. String values are safely converted to floats; valid onset and duration define an absolute interval. A nominal bin is 1 when its center is at or after onset and before offset, otherwise 0.

ii.
```python
on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
x[on] = 1.0
```

iii. The binary time series directly matches the requested decoder input. Invalid or `N/A` values produce an all-zero vector.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute photostimulation times are converted to go-relative times and compared with nominal go-relative bin centers. As with the tone input, nominal correspondence is correct but can be offset from the rounded session-grid neural bins.

ii.
```python
rs = s - go_time
re = e - go_time
on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
```

iii. The agent says photostimulation is represented across the same go-aligned full window and verified one trial against the trial-table timing.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from left- and right-lick event timestamps, each trial's go time, and its stop time. It does not use `trial_instruction` and `outcome` as the reference does.

ii.
```python
left_licks = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:]
right_licks = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:]
...
l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
r_post = np.any((right_licks >= go_times[i]) & (right_licks < min(stop[i], go_times[i] + 1.5)))
```

iii. The mapping plan calls for trial lick/no-response fields and behavioral-event logic. The agent does not document why direct lick events are preferable to the unambiguous instruction-plus-outcome derivation used by the reference.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left-only is class 0, right-only class 1, and neither or both is class 2 (`no lick`). The class is repeated across all 80 bins.

ii.
```python
if l_post and not r_post:
    choice[i] = 0
elif r_post and not l_post:
    choice[i] = 1
else:
    choice[i] = 2
...
np.full(N_BINS, choice[i], dtype=np.int64)
```

iii. The notes say the requested no-lick class must be retained. They do not address trials with both left and right lick events, which this logic incorrectly labels no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes from the trials-table `outcome` field when available, with choice used only as a fallback if the value is unrecognized or the column is absent.

ii.
```python
if 'outcome' in table:
    ov = np.array([... for v in table['outcome']], dtype=object)
else:
    outcome[:] = np.where(choice == 2, 0, 2)
```

iii. The notes identify the explicit trial outcome as the intended source and retain all three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Case-normalized strings map to 2 for hit/correct, 1 for miss/error/incorrect, and 0 for ignore/no. Unknown values become ignore for no choice and hit otherwise. The result is repeated across time.

ii.
```python
if s == 'hit' or 'correct' in s:
    outcome[i] = 2
elif s == 'miss' or 'error' in s or 'incorrect' in s:
    outcome[i] = 1
elif s == 'ignore' or 'no' in s:
    outcome[i] = 0
```

iii. For this dataset the explicit values are already `ignore`, `miss`, and `hit`, so the broader aliases are defensive and normally have no effect.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It uses the trials-table `early_lick` column. If absent, the initialized all-zero array is retained.

ii.
```python
if 'early_lick' in table:
    ev = np.asarray(table['early_lick'])
```

iii. The notes state that the trial table explicitly provides this flag and that these trials must be kept for the decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The exact normalized string `early` becomes 1; every other value becomes 0. The per-trial value is repeated over all 80 bins.

ii.
```python
early = np.array([
    1 if str(...).strip().lower() == 'early' else 0
    for v in ev
], dtype=np.int64)
```

iii. This implements the requested no/yes categorization and preserves early-lick trials rather than filtering them.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking/data` and the matching camera timestamps. It does not use column 2, the tracking likelihood/confidence.

ii.
```python
tongue_xy = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
...
y = np.asarray(data_xy[:, 1], dtype=float)
visible = np.isfinite(y)
```

iii. The notes correctly identify tongue y and mention missing/low-confidence samples should be not visible, but the code only treats non-finite y as invisible and ignores the confidence channel.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code computes session-wide percentiles over raw finite frame-level y values, categorizes every frame, then assigns each trial/bin the modal visible frame category. Empty/no-visible bins become class 3. It does not first average visible y within each 50-ms bin.

ii.
```python
q40, q60 = np.nanpercentile(y[visible], [40, 60])
...
vals = labels[m]
vals = vals[vals != 3]
out[b] = 3 if len(vals) == 0 else np.bincount(vals, minlength=3).argmax()
```

iii. The notes say percentiles are per-session over visible samples, but do not distinguish raw frames from the reference's per-bin means. Their reported tongue classes omit class 3 in the sample distribution, consistent with the visibility bug.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Finite y below the 40th percentile is 0; y from the 40th through the 60th percentile inclusive is 1; y above the 60th is 2; non-finite y is 3. Thresholds are session-wide raw-frame percentiles.

ii.
```python
out[visible & (y < q40)] = 0
out[visible & (y >= q40) & (y <= q60)] = 1
out[visible & (y > q60)] = 2
```

iii. The 40/60 cut points and four requested labels come directly from the task. The agent did not justify assigning exact equality at q60 to the middle class or ignoring likelihood when defining visibility.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in `[go-2.5, go+1.5)` are selected and assigned to nominal go-relative 50-ms bins. Each bin receives the modal frame label. This shares nominal go alignment with the inputs, but the neural extraction can have a small session-grid phase error.

ii.
```python
m = (tongue_ts >= go + T_START) & (tongue_ts < go + T_END)
tongue_trial = bin_tongue_for_trial(tongue_ts[m], tongue_disc[m], go)
...
rel = timestamps - go_time
m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
```

iii. The agent states camera and go timestamps share the session clock and uses the requested window and bin width.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The implementation contains broad fallbacks: inferred go cues, trial starts as tone times, absent lick streams as empty arrays, absent outcomes inferred from choice, absent tongue data mapped entirely to not-visible, alternate unit QC fields/metric thresholds, and unknown regions. Invalid numeric strings become NaN. Out-of-range or all-zero trials and sessions with fewer than two trials are dropped.

ii.
```python
return start + np.minimum(2.0, (stop - start) / 2)
...
except Exception:
    return np.nan
...
tongue_trial = np.full(N_BINS, 3, dtype=np.int64)
```

iii. The notes frame these as robust handling and specifically document dropping all-zero neural trials. Several fallbacks silently fabricate labels/times rather than failing on a schema mismatch, and the unit-QC fallback retains the uncurated session.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large NWB arrays, histogramming spikes for every retained unit, scanning camera data, and serializing the full pickle dominate. The agent specifically identified per-trial/per-unit spike binning as the initial bottleneck and optimized it to per-unit session-wide histograms.

ii.
```python
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    counts, _ = np.histogram(st[m], bins=session_edges)
```

iii. The notes report the spike-binning change reduced sample-session runtime from about 38.7 seconds to 1.37 seconds and estimate roughly four minutes for the full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining candidates include the per-unit histogram loop, per-trial conversion loop, per-bin tongue loop, per-trial lick-event scans, and repeated Python list/index searches when building subject and region mappings. The tongue bin loop could especially be replaced by precomputed bin indices and grouped reductions; lick directions could be computed with sorted-event `searchsorted` operations.

ii.
```python
for i in range(n):
    l_post = np.any(...)
    r_post = np.any(...)
...
for b in range(N_BINS):
    m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
```

iii. The notes acknowledge the original per-trial/per-unit approach as slow but only document the session pre-binning optimization; they do not discuss the other vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. Trial interval masks are repeatedly scanned for every trial when selecting sample events, lick events, fallback photostim events, and tongue frames. `spike_times_index` is reread inside every unit call. Subject and brain-region indices use repeated linear list membership and `.index` searches. The identical all-zero neural check runs twice.

ii.
```python
ind = units_group['spike_times_index'][:]
...
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
```

iii. The notes do not identify these repetitions. They emphasize the larger eliminated repetition: spikes are now binned once per unit/session rather than once per unit/trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads every dataset in the full units table although many are unused; loads fallback global photostim arrays even when trial photostim columns exist; computes and returns tongue thresholds but never uses them; builds `stop_times` for several fallback paths; and performs the duplicate all-zero check. With `--show-processing`, it also creates diagnostic figures that are not part of the dataset.

ii.
```python
units = load_units_table(f)
...
tongue_disc, tongue_thr = discretize_tongue_y(tongue_xy)
...
ps_starts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps'][:]
```

iii. The notes justify diagnostic plots and extensive fallbacks for validation, but do not call out the unused `tongue_thr`, eager table loading, or duplicate condition.
