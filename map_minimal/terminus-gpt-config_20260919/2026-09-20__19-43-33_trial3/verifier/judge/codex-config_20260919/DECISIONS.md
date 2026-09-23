# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every `.nwb` file under `/app/data`, sorts the paths, and opens each file once with `NWBHDF5IO`. It reads subject, trial, unit, behavioral-event, and camera data from the NWB file. It retains sessions having at least one selected unit.

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for si, path in enumerate(files):
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
```

iii. The trajectory says the dataset contains one large NWB collection across many mice and that streaming one NWB at a time controls memory. It treats recursive discovery as covering the full release and later reports 174 files, with 173 retained sessions.

## 1-b. How are the data split into subjects?

i. Subjects are initially inferred from the `sub-...` filename prefix. For each retained session the code reads `nwb.subject.subject_id` and maps that ID into the filename-derived, sorted subject list.

ii.
```python
subjects = sorted({p.name.split('_')[0].replace('sub-', '') for p in files})
subject_to_idx = {s:i for i,s in enumerate(subjects)}
...
sid = str(nwb.subject.subject_id)
...
subject_idx.append(subject_to_idx[sid])
```

iii. The trajectory identifies `nwb.subject.subject_id` as the session's subject and reports 28 subjects. It gives no separate justification for deriving the master list from filenames, apparently relying on the DANDI naming convention matching the NWB field.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Output session order is the sorted recursive file order; a file is skipped if no unit passes the agent's unit filter.

ii.
```python
for si, path in enumerate(files):
    ...
    if len(kept_idx) == 0:
        print('  skipped: no good annotated units', flush=True)
        continue
    ...
    neural.append(sess_n); inputs.append(sess_i); outputs.append(sess_o)
```

iii. The agent inferred the NWB file/session correspondence from the dataset layout. Its final checks report 173 sessions from 174 files; the omitted file had no units satisfying its filter.

## 1-d. How are the data split into trials?

i. NWB trial-table rows define trials. For each row, the code uses `[start_time, stop_time)` and assigns the first go cue and first sample/tone event in that interval. It raises an error if any trial lacks either event, then loops over all rows.

ii.
```python
starts = np.asarray(tr['start_time'][:], float)
stops = np.asarray(tr['stop_time'][:], float)
ntr = len(starts)
go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
...
for ti in range(ntr):
```

iii. The agent explicitly chose trial-interval assignment instead of assuming event-array position, describing this as safer. It expected one go cue and at least one tone in every retained behavioral trial.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered. Every trial-table row is retained if it has a go cue and tone; missing events abort conversion rather than excluding the row. The code does not use `units/obs_intervals` or remove `free_water` trials.

ii.
```python
if np.any(~np.isfinite(go)):
    raise ValueError(...)
...
if np.any(~np.isfinite(tone)):
    raise ValueError(...)
...
'trial_filter': 'all trials with go cue and tone onset',
```

iii. The module docstring says “all released behavioral sessions and trials are retained.” The trajectory does not investigate trials lacking spike observations or free-water trials; it considers retaining all behavioral trials consistent with the requested outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the ragged NWB `units['spike_times']` arrays for selected units. Trial go-cue timestamps determine absolute bin edges. Unit selection uses `unit_quality` and `anno_name`.

ii.
```python
quality = np.asarray(nwb.units['unit_quality'][:], dtype=str)
annotation = np.char.strip(np.asarray(nwb.units['anno_name'][:], dtype=str))
...
spike_col = nwb.units['spike_times']
spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
...
sess_n.append(bin_spikes(spikes, go[ti]))
```

iii. The trajectory identifies spike times as the source representation and the go cue as the required alignment event. It bulk-loads the ragged buffer after observing that per-unit HDF5 reads were slow.

## 2-b. How is the `neural` data processed?

i. For each trial and unit, sorted spike times are counted in 80 half-open bins using insertion-index differences and divided by 0.05 s to produce firing rates in Hz. No smoothing, normalization, or baseline correction is applied.

ii.
```python
abs_edges = go + EDGES
for u, st in enumerate(spike_times):
    st = np.asarray(st)
    out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
```

iii. The agent says `searchsorted` preserves the reference histogram's `[left,right)` behavior while avoiding repeated full spike-train scans. It chose rates because the supplied preprocessing uses `sliding_histogram(..., rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is retained only if `unit_quality == 'good'` and `anno_name` is nonempty and not the string `nan`. A session with zero such units is skipped. The newer `classification` field is not used.

ii.
```python
keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
kept_idx = np.flatnonzero(keep)
if len(kept_idx) == 0:
    ...
    continue
```

iii. The trajectory concludes that the paper's classifier-good units correspond to `unit_quality == 'good'`, and that the provided preprocessing also intersects ephys units with histology. After validation exposed a literal `nan` region, the agent tightened the annotation filter. This interpretation of the classifier field was mistaken.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The first go cue found inside each trial interval is used. Relative edges from -2.5 to +1.5 s are added to its absolute timestamp and spikes are histogrammed against those absolute edges.

ii.
```python
go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
...
abs_edges = go + EDGES
```

iii. The agent explicitly follows the requested go-cue alignment and notes that event and spike timestamps share the NWB session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50 ms bins over [-2.5, 1.5] s. Raw spike times are binned directly to this grid; there is no later rebinning.

ii.
```python
START, END, DT = -2.5, 1.5, 0.05
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The 50 ms resolution and four-second window come directly from the task. The trajectory also connects nonoverlapping 50 ms bins to the supplied reference histogram settings.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, trial start/stop times, the selected go cue, and the common bin centers. The first sample event within each trial is taken as tone onset.

ii.
```python
tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
...
bt = go[ti] + CENTERS
time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. The agent regarded the sample event as the tone and deliberately assigned events through trial intervals. It did not recognize that early licks can replay the sample epoch and that the final tone before go is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each neural-bin center in absolute time, the chosen tone timestamp is subtracted, producing a continuous float32 vector in seconds.

ii.
```python
bt = go[ti] + CENTERS
time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. The trajectory describes this as direct elapsed time from the trial's tone, with no additional transformation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 go-cue-relative bin centers used by the spike bins, so input element `k` corresponds to neural bin `k`.

ii.
```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
bt = go[ti] + CENTERS
```

iii. The agent uses one shared bin-center grid for time-varying inputs and outputs to ensure shape and temporal correspondence.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the session-wide `BehavioralEvents/photostim_start_times` and `photostim_stop_times` timestamp arrays plus each trial's absolute bin centers.

ii.
```python
ps = np.asarray(events['photostim_start_times'].timestamps[:], float)
pe = np.asarray(events['photostim_stop_times'].timestamps[:], float)
```

iii. The trajectory found explicit stimulation event streams and chose them as direct recorded on/off times, instead of reconstructing intervals from trial-table onset/duration fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Counts of starts and stops are checked for equality. Each bin center is classified 1 if it falls in any global `[start, stop)` stimulation interval and 0 otherwise.

ii.
```python
if len(ps) != len(pe):
    raise ValueError(...)
photo = np.zeros(80, dtype=np.float32)
if len(ps):
    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. The agent wanted a continuous binary trace sampled with the neural bins and adopted the same half-open interval convention.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute photostimulation intervals are tested at the same absolute bin centers, `go + CENTERS`, corresponding to the neural bins.

ii.
```python
bt = go[ti] + CENTERS
photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. The agent notes that all event streams share the NWB clock, so direct timestamp comparison aligns stimulation with neural activity.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial-table `trial_instruction` and `outcome`; it does not use raw lick events. Ignore means no lick, hit means the instructed direction, and miss means the opposite direction.

ii.
```python
instruction = np.asarray(tr['trial_instruction'][:], dtype=str)
outcome = np.asarray(tr['outcome'][:], dtype=str)
...
if outcome[ti] == 'ignore':
    choice = 2
elif outcome[ti] == 'hit':
    choice = 0 if instruction[ti] == 'left' else 1
else:
    choice = 1 if instruction[ti] == 'left' else 0
```

iii. The agent inferred that instruction plus correctness/outcome fully determines response direction and supplies an explicit third no-response class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded 0=left, 1=right, 2=no lick and repeated across all 80 time bins as a per-trial categorical label.

ii.
```python
o = np.empty((4,80), dtype=np.int8)
o[0] = choice
...
'output_values': [['left','right','no lick'], ...]
```

iii. The agent says repeating per-trial labels keeps a rectangular output alongside the time-varying tongue output and is permitted by the target format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the NWB trials-table `outcome` column.

ii.
```python
outcome = np.asarray(tr['outcome'][:], dtype=str)
```

iii. The requested categories already exist in the raw trial table, so the agent uses them directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0=ignore, 1=miss, 2=hit, cast into the int8 output array, and repeated for all bins.

ii.
```python
out_map = {'ignore':0, 'miss':1, 'hit':2}
...
o[1] = out_map[outcome[ti]]
```

iii. This fixed mapping exactly follows the requested category order; temporal repetition is justified as for choice.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the NWB trials-table `early_lick` column.

ii.
```python
early = np.asarray(tr['early_lick'][:], dtype=str)
```

iii. The agent found the explicit early-lick trial field and therefore did not reconstruct it from lick timestamps.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `'early'` becomes 1; every other value becomes 0. The result is repeated across 80 bins in the int8 output.

ii.
```python
o[2] = 1 if early[ti] == 'early' else 0
```

iii. The trajectory identifies the expected labels and uses a binary no/yes encoding. It does not validate unexpected strings.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking`: column 1 is y-position, column 2 is tracking likelihood, and `timestamps` provide camera times. Go-cue times and neural bin centers provide the target sampling times.

ii.
```python
tts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
cam_t = np.asarray(tts.timestamps[:], dtype=float)
cam = np.asarray(tts.data[:], dtype=float)
```

iii. The trajectory inspected the series description and confirmed `(tongue_x, tongue_y, tongue_likelihood)` at about 294 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Samples with finite y and likelihood at least 0.9 are considered visible for session thresholds. For each neural-bin center, the nearest camera sample is selected; it is visible only if y and likelihood are finite and likelihood is at least 0.9. Visible y is categorized and invisible samples get class 3. Camera samples are not averaged within 50 ms bins.

ii.
```python
visible_all = np.isfinite(cam[:,1]) & np.isfinite(cam[:,2]) & (cam[:,2] >= DLC_THRESHOLD)
...
idx = np.searchsorted(cam_t, bt)
idx = np.clip(idx, 1, len(cam_t)-1)
prev = idx - 1
idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
y = cam[idx,1]
vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
```

iii. The agent chose a conventional stringent DLC threshold of 0.9 and nearest-sample resampling. It states that thresholds should use confidently visible tongue samples, but provides no evidence that nearest-frame sampling matches the supplied method.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are calculated per session over all raw camera-frame y values passing the 0.9 likelihood test. At target times: y below q40 is class 0, q40 through q60 inclusive class 1, y above q60 class 2, and nonvisible class 3.

ii.
```python
q40, q60 = np.percentile(cam[visible_all,1], [40, 60])
...
tongue_cat[vis & (y < q40)] = 0
tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
tongue_cat[vis & (y > q60)] = 2
```

iii. The per-session 40/60 cutoffs and four requested labels come from the task. The agent chose percentiles of raw visible frames rather than percentiles of 50 ms bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. At each absolute neural-bin center, the nearest camera timestamp is chosen, regardless of whether the camera timestamp lies within that trial's recorded video interval.

ii.
```python
bt = go[ti] + CENTERS
idx = np.searchsorted(cam_t, bt)
...
idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
```

iii. The agent uses the shared session clock and common bin centers for alignment. It does not impose a maximum time distance or mark pre-video gaps missing, so nearest samples can be carried into uncovered bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing go cues or tones cause a `ValueError`. Unequal photostimulation start/stop counts also cause a `ValueError`. Sessions with no selected units are skipped. Missing/invalid anatomical annotations are excluded. If no visible tongue samples exist, percentile edges become NaN and every tongue bin remains class 3; individual nonvisible samples likewise become class 3. Trials without spike recording are not detected.

ii.
```python
if np.any(~np.isfinite(go)):
    raise ValueError(...)
...
if len(kept_idx) == 0:
    continue
...
if not np.any(visible_all):
    q40 = q60 = np.nan
...
tongue_cat = np.full(80, 3, dtype=np.int8)
```

iii. The agent favors failing on structurally required missing events, skipping unusable sessions, and representing missing tongue visibility explicitly. The trajectory also shows a post-validation correction to reject annotations converted to literal `'nan'`. It did not inspect `obs_intervals`, so missing neural observations appear as zero firing.

## 10-a. What are the most time-consuming steps of the code?

i. Full-session NWB I/O, loading spike and camera arrays, per-trial/per-unit spike searches, constructing roughly 94,000 trial objects, and serializing the approximately 10.8 GB pickle dominate runtime.

ii.
```python
spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
...
for ti in range(ntr):
    sess_n.append(bin_spikes(spikes, go[ti]))
...
cam = np.asarray(tts.data[:], dtype=float)
```

iii. The trajectory directly observed slow initial runs, attributing them first to repeated histograms and ragged HDF5 access and later to reading 174 NWBs, camera streams, object construction, and pickle serialization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested trial loop and per-unit loop in `bin_spikes` could be improved by forming all trial edges once per session and searching each unit once, as the reference does. The per-trial photostimulation comparison and nearest-camera lookup could also be vectorized across all trials.

ii.
```python
for ti in range(ntr):
    ...
    sess_n.append(bin_spikes(spikes, go[ti]))
```
```python
for u, st in enumerate(spike_times):
    out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
```

iii. The agent recognized the initial nested histogram implementation as a major bottleneck and replaced histograms with `searchsorted`, but it retained the outer per-trial calls. It also bulk-loaded ragged spike storage to remove thousands of small HDF5 reads.

## 10-c. What processing does the code repeat multiple times?

i. For every trial, it calls `bin_spikes`, and for every retained unit it searches that unit's spike array at 81 edges. Thus the same unit spike train is revisited once per trial. It also compares each trial's 80 centers against every session photostimulation interval and performs camera `searchsorted` per trial.

ii.
```python
for ti in range(ntr):
    ...
    sess_n.append(bin_spikes(spikes, go[ti]))
    ...
    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
    ...
    idx = np.searchsorted(cam_t, bt)
```

iii. The trajectory notes that the original implementation repeatedly scanned full spike trains and optimizes the inner operation, but does not eliminate repeated trial-level searches.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and retains every source subject name before session filtering, computes and stores tongue percentile/threshold metadata per session, runs `gc.collect()` after every file, and compares each trial with all session photostimulation epochs. More materially, it processes trials without spike observations into all-zero neural arrays; that work produces misleading downstream samples rather than useful data.

ii.
```python
subjects = sorted({p.name.split('_')[0].replace('sub-', '') for p in files})
...
session_info.append({... 'tongue_q40': float(q40), 'tongue_q60': float(q60), ...})
...
gc.collect()
```

iii. The trajectory gives no explicit justification for these minor costs. It treats full behavioral-trial retention as intentional, so it did not recognize processing unobserved/free-water trials as unnecessary. Most computed arrays do enter the pickle; the main waste is avoidable repeated work rather than a large discarded intermediate.
