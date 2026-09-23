# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It recursively finds all local `behavior_ophys_experiment_*.nwb` files under `/app/data`, reads `ophys_experiment_table.csv`, restricts metadata to those local experiment IDs, removes passive sessions and experiments without eye tracking, and then opens each retained NWB directly with `h5py` inside `process_experiment`.

ii. 
```python
def find_files():
    files = sorted(glob.glob(str(DATA_ROOT / '**' / 'behavior_ophys_experiment_*.nwb'), recursive=True))
    return {int(re.search(r'(\d+)\.nwb$', f).group(1)): f for f in files}

def select_experiments(files):
    meta = metadata_table('ophys_experiment_table.csv')
    meta = meta[meta.ophys_experiment_id.isin(files)].copy()
    meta = meta[~meta.session_type.str.contains('passive', case=False, na=False)].copy()
    eye_map = {eid: has_eye(files[eid]) for eid in meta.ophys_experiment_id.astype(int)}
    meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
    return meta

for k, row in meta.iterrows():
    eid = int(row.ophys_experiment_id)
    n, x, y, info = process_experiment(row, files[eid], imap, ...)
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the AI justifies direct NWB/HDF5 loading as a faster, lower-overhead substitute for full AllenSDK/PyNWB object construction. The notes and trajectory also state that passive experiments were excluded because the paper analyzed the active task, and experiments without eye tracking were excluded because pupil diameter was a required decoder output that “cannot be fabricated.”

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from the retained experiment metadata. The final `subjects` list is the sorted set of retained mouse IDs, stored as strings, and each session receives a `subject_idx` by looking its experiment’s `mouse_id` up in that set.

ii. 
```python
info = {'mouse_id': str(row.mouse_id), ...}
...
subjects = sorted({z['mouse_id'] for z in infos})
smap = {x: i for i, x in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.asarray([smap[z['mouse_id']] for z in infos], dtype=np.int64),
```

iii. The notes in Step 5 describe mouse ID as the source for `subjects`/`subject_idx` and say the retained experiments cover 38 mice after experiment-level curation.

## 1-c. How are the data split into sessions?

i. The AI treats each retained `ophys_experiment_id` file, i.e. each imaging plane, as a separate decoder session. It does not group multiple experiments from the same `ophys_session_id` together. The final metadata explicitly records `session_unit` as `ophys experiment / imaging plane`.

ii. 
```python
for k, row in meta.iterrows():
    eid = int(row.ophys_experiment_id)
    n, x, y, info = process_experiment(row, files[eid], imap, ...)
    neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)

info = {'ophys_experiment_id': eid, 'ophys_session_id': int(row.ophys_session_id), ...}
...
'metadata': {
    ...
    'session_unit': 'ophys experiment / imaging plane',
    'session_info': infos,
}
```

iii. In Step 4 and Step 5, the AI explicitly justifies this by saying each experiment file contains a distinct simultaneously recorded neuron population, while multi-plane sister files duplicate behavior. The trajectory says this matched the paper’s plane-level decoding better than acquisition-session-level grouping.

## 1-d. How are the data split into trials?

i. Trials are taken from each NWB file’s `intervals/trials` table. The AI keeps only `go` or `catch` trials, then defines each trial by its full `start_time` to `stop_time` interval. Instead of keeping native ophys frames, it converts each interval into a variable-length sequence of 100 ms bins using `floor((stop-start)/0.1)`.

ii. 
```python
tr = h['intervals/trials']
flags = {k: np.asarray(tr[k]).astype(bool) for k in
         ['go','catch','aborted','auto_rewarded','hit','miss','false_alarm','correct_reject']}
keep = (flags['go'] | flags['catch']) & ~flags['aborted'] & ~flags['auto_rewarded']
tids = np.flatnonzero(keep)
starts = np.asarray(tr['start_time'], float)
stops = np.asarray(tr['stop_time'], float)

for ti in tids:
    nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
    if nbin < 1:
        continue
    centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
```

iii. The notes in Step 5 say the AI intentionally “retain[s] full SDK/NWB start-to-stop intervals,” keeps variable-length trials, and uses trial start as the alignment event with a fixed 100 ms common bin size.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level curation keeps only `go` or `catch` rows and excludes `aborted` and `auto_rewarded` trials. Trials with fewer than one 100 ms bin are dropped. After interpolation, any trial with non-finite running or pupil values is also dropped. At the session/experiment level, passive experiments and experiments without eye tracking are removed before trial extraction, and a session must end with at least 2 usable trials.

ii. 
```python
meta = meta[~meta.session_type.str.contains('passive', case=False, na=False)].copy()
...
meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
...
keep = (flags['go'] | flags['catch']) & ~flags['aborted'] & ~flags['auto_rewarded']
...
if nbin < 1:
    continue
...
r = interp_valid(centers, run_t, run_x)
p = interp_valid(centers, eye_t, pupil, pupil_good)
if not (np.isfinite(r).all() and np.isfinite(p).all()):
    continue
...
if len(neural) < 2:
    raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. The notes say the AI followed the required `(go | catch) & ~aborted & ~auto_rewarded` trial curation, excluded passive experiments because their task labels were not behaviorally meaningful, and excluded no-eye experiments because pupil was mandatory. Step 5 also says missing behavior samples should not become a new category; instead a trial should be dropped if valid support is insufficient.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the NWB event-detection output, specifically `processing/ophys/event_detection/data` with its matching timestamps, not from dF/F traces.

ii. 
```python
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
if ev.shape[0] != len(ots):
    raise ValueError(f'{eid}: events/timestamps mismatch')
```

iii. In Step 3, Step 4, and Step 5, the AI repeatedly justifies this as matching the paper’s use of detected calcium events for neural analyses, even though the NWBs also contain dF/F.

## 2-b. How is the `neural` data processed?

i. The raw event-detection matrix is converted into per-trial neuron-by-time matrices by averaging event magnitudes within 100 ms bins across the trial interval. If a 100 ms bin contains no native samples, the code linearly interpolates between neighboring native frames. The output is stored as `float32` with shape `(n_neurons, n_bins)`.

ii. 
```python
def event_bin_means(events, ts, starts, centers):
    out = np.empty((events.shape[1], len(centers)), dtype=np.float32)
    edges = starts + np.arange(len(centers) + 1) * BIN_S
    lo = np.searchsorted(ts, edges[:-1], side='left')
    hi = np.searchsorted(ts, edges[1:], side='left')
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
        else:
            k = np.searchsorted(ts, centers[j])
            ...
            out[:, j] = (1-w)*events[k-1] + w*events[k]
    return out

n = event_bin_means(ev, ots, starts[ti], centers)
```

iii. Step 5 says the AI wanted a single common 100 ms bin width that could accommodate both ~11 Hz and ~31 Hz experiments while preserving 250 ms flashes and the paper’s 400 ms decoding window. The notes also say no additional smoothing or normalization was intended beyond binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra neuron-level quality filtering is applied inside `convert_data.py`. The AI assumes the released NWBs are already post-QC and only checks for an events/timestamps length mismatch.

ii. 
```python
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
if ev.shape[0] != len(ots):
    raise ValueError(f'{eid}: events/timestamps mismatch')
```

iii. Step 4 says the supplied NWBs are already “post-release-QC” and that adding ad hoc neuron thresholds would “double-filter” the data without justification.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to trial start. For each retained trial, the AI builds 100 ms bins beginning at `start_time`, uses the bin centers as the common trial time grid, and fills each bin using native event timestamps.

ii. 
```python
nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)
...
'metadata': {
    ...
    'temporal_alignment_event': 'trial start time on the ophys session clock',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes explicitly say “Alignment event is trial start (`off_start=0`, `off_end=None`).” The trajectory also records that the validator required homogeneous time-varying outputs, which reinforced trial-start-based binning over full trial windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms bin size (`BIN_S = 0.100`). Yes, temporal rebinning is applied: native 11 Hz or 31 Hz event traces are averaged into 100 ms bins for every trial.

ii. 
```python
BIN_S = 0.100
...
nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)
...
'metadata': {
    'time_bin_size': 100.0,
    'neural_signal': 'detected calcium event magnitude, mean in 100 ms bins',
}
```

iii. Step 5 states that the AI deliberately chose 100 ms as a common bin size across mixed acquisition rates, arguing that it preserved stimulus timing while satisfying the requirement for a uniform time bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation streams in the NWB file: `stimulus/presentation/*/{timestamps,data}` plus the corresponding template `control_description` strings in `stimulus/templates`. The AI does not derive image identity from the trial table’s `initial_image_name` / `change_image_name` columns.

ii. 
```python
def presentation_streams(h):
    ...
    for gname, g in h['stimulus/presentation'].items():
        if 'timestamps' not in g or 'data' not in g:
            continue
        t = np.asarray(g['timestamps'], float)
        idx = np.asarray(g['data']).astype(int)
        ...
        desc = decode_strings(templates[gname]['control_description'][:])
        ...
        onsets.extend(t[good]); names.extend(desc[idx[good]])
    ...
    return np.asarray(onsets)[order], np.asarray(names)[order]

flash_t, flash_name = presentation_streams(h)
```

iii. The notes say the AI wanted the actual stimulus presentation timing so it could distinguish image-on periods from gray intervals, because the task asked for “the image presented during the non-grey screen.”

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first builds a global image vocabulary from all retained experiments, with `gray` hard-coded as category 0. Then, within each trial, each 100 ms bin is labeled with the most recent flashed image only if the bin center falls within 250 ms after a flash onset; otherwise the label remains `gray`. Omitted or gray-like template labels are removed from the vocabulary and therefore collapse to `gray`.

ii. 
```python
def image_vocabulary(meta, files):
    names = set()
    ...
    return ['gray'] + sorted(n for n in names if n and n.lower() not in {'gray', 'omitted'})

image = np.zeros(nbin, dtype=np.int64)
# A natural image is visible for 250 ms after each flash onset.
left = np.searchsorted(flash_t, centers - 0.250, side='left')
right = np.searchsorted(flash_t, centers, side='right')
for j, (a, b) in enumerate(zip(left, right)):
    if b > a:
        ft = flash_t[b-1]
        if ft <= centers[j] < ft + 0.250:
            image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
```

iii. Step 5 says the AI intentionally added a gray category and treated “gray includes inter-flash gaps and omission intervals because requested identity is the image shown during non-gray screen.” The trajectory also records a deliberate emphasis on the 250 ms image plus 500 ms gray cadence.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned to the same 100 ms trial bin centers used for neural data. Each bin’s label is computed from the most recent flash onset relative to that bin center, so image identity and neural activity share identical trial lengths and time bins.

ii. 
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)
...
left = np.searchsorted(flash_t, centers - 0.250, side='left')
right = np.searchsorted(flash_t, centers, side='right')
...
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
```

iii. The notes state that all output streams should be sampled on the same common ophys-aligned bin grid as the neural data, which in this script means the 100 ms trial bin centers.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the trials table’s `change_time` values together with the `go` flag. Catch trials stay at zero throughout.

ii. 
```python
changes = np.asarray(tr['change_time'], float)
...
change = np.zeros(nbin, dtype=np.int64)
if flags['go'][ti] and np.isfinite(changes[ti]):
    j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
    if 0 <= j < nbin:
        change[j] = 1
```

iii. Step 5 says the AI used the true change onset and kept catch trials at zero because there is no actual image identity change on catch trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI converts each go trial’s scalar `change_time` into a one-bin impulse: it finds the first 100 ms bin at or after the change onset and sets only that bin to 1. All other bins stay 0.

ii. 
```python
change = np.zeros(nbin, dtype=np.int64)
if flags['go'][ti] and np.isfinite(changes[ti]):
    j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
    if 0 <= j < nbin:
        change[j] = 1
```

iii. The notes explicitly describe this as a “change impulse” and justify it by the task wording “right after” a change. The trajectory also says the AI preferred an impulse rather than a sustained post-change label.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary only: `0` for `no_change`, `1` for `change`.

ii. 
```python
'output_values': [
    vocab,
    ['no_change', 'change'],
    ...
]
...
change = np.zeros(nbin, dtype=np.int64)
...
change[j] = 1
```

iii. The AI’s notes in Step 5 call this a one-bin binary change event and do not describe any further thresholding beyond the 0/1 encoding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned to the same 100 ms trial bins as the neural data. The bin index is computed from `change_time - start_time`, and the resulting `change` vector is stacked directly into the per-trial output matrix that shares the neural trial length.

ii. 
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)
...
j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
...
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
```

iii. Step 5 says all outputs are represented on the common bin grid. The trial-start alignment and fixed bin width are the key justification recorded by the AI.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running-speed stream at `processing/running/speed/{timestamps,data}`.

ii. 
```python
run_t = np.asarray(h['processing/running/speed/timestamps'], float)
run_x = np.asarray(h['processing/running/speed/data'], float)
```

iii. The notes identify the NWB running-speed dataset as the released locomotion signal and say it should be interpolated onto the common trial time grid.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. For each trial, the AI linearly interpolates running speed to the 100 ms trial bin centers using `np.interp`. After all usable trials for one experiment/session are collected, it computes 20/40/60/80% quantile edges from all finite running samples in that experiment and labels each time point by which quintile bin it falls into.

ii. 
```python
def interp_valid(t_new, t, x, valid=None):
    ...
    return np.interp(t_new, t[good], x[good]).astype(np.float32)

r = interp_valid(centers, run_t, run_x)
...
def quintile_labels(values):
    allv = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
    labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
    return labels, edges

run_lab, run_edges = quintile_labels([q[2] for q in prelim])
```

iii. Step 5 says the AI wanted percentile bins rather than continuous values, but chose to compute quintiles independently within each experiment/session. The notes frame this as avoiding a missing-data category and keeping the bins balanced within each retained session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-frequency bins per retained experiment/session. The categories are integer labels `0` through `4`, named `0-20%`, `20-40%`, `40-60%`, `60-80%`, and `80-100%`.

ii. 
```python
edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
...
'output_values': [
    vocab,
    ['no_change', 'change'],
    ['0-20%','20-40%','40-60%','60-80%','80-100%'],
    ...
]
```

iii. Step 5 explicitly calls these “quintiles 0–4” and says the labels come from per-experiment valid aligned samples.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolating to the same 100 ms trial bin centers that are used to bin neural events. The resulting discretized labels have the same per-trial length as the neural matrices.

ii. 
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
r = interp_valid(centers, run_t, run_x)
n = event_bin_means(ev, ots, starts[ti], centers)
...
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
```

iii. The notes say all behavioral streams should be sampled on the same common bin grid as neural activity, which is why interpolation happens before discretization.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the eye-tracking stream at `acquisition/EyeTracking/pupil_tracking/{timestamps,data}` together with the blink mask at `acquisition/EyeTracking/likely_blink/data`. The AI computes a scalar pupil size from the two stored ellipse axes as `sqrt(width * height)`.

ii. 
```python
eye_t = np.asarray(h['acquisition/EyeTracking/pupil_tracking/timestamps'], float)
axes = np.asarray(h['acquisition/EyeTracking/pupil_tracking/data'], float)
blink = np.asarray(h['acquisition/EyeTracking/likely_blink/data']).astype(bool)
# EllipseSeries stores width and height (diameters); equivalent circular diameter.
pupil = np.sqrt(np.maximum(axes[:, 0] * axes[:, 1], 0.0))
pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
```

iii. The notes and trajectory say the AI inspected the NWB schema and decided the stored values were ellipse diameters, so it used an equivalent circular diameter instead of a single width column. Blink flags were treated as invalid samples.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI removes blink and non-positive/non-finite samples, linearly interpolates the derived pupil size to the 100 ms trial bin centers, and then computes per-experiment/session quintile bins from all finite aligned pupil samples.

ii. 
```python
pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
...
p = interp_valid(centers, eye_t, pupil, pupil_good)
...
pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
```

iii. Step 4 and Step 5 say blink and invalid samples should be treated as missing, valid samples should be interpolated onto the common trial grid, and no special missing-data category should be introduced. The same per-session quintile logic used for running speed is applied here.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-frequency bins per retained experiment/session, labeled `0` through `4` with the same percentile names used for running speed.

ii. 
```python
edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
...
'output_values': [
    ...,
    ['0-20%','20-40%','40-60%','60-80%','80-100%'],
    ['0-20%','20-40%','40-60%','60-80%','80-100%'],
    ...
]
```

iii. The notes explicitly describe pupil labels as per-experiment/session quintiles computed from valid aligned samples.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating the derived pupil-size signal to the same 100 ms trial bin centers used for neural binning. The discretized pupil vector is then stacked into the output matrix for each trial.

ii. 
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
p = interp_valid(centers, eye_t, pupil, pupil_good)
n = event_bin_means(ev, ots, starts[ti], centers)
...
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
```

iii. The notes state that running, pupil, and stimulus outputs all share the common trial bin grid with the neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table flags `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
flags = {k: np.asarray(tr[k]).astype(bool) for k in
         ['go','catch','aborted','auto_rewarded','hit','miss','false_alarm','correct_reject']}
...
if flags['hit'][ti]: outcome = 0
elif flags['miss'][ti]: outcome = 1
elif flags['false_alarm'][ti]: outcome = 2
elif flags['correct_reject'][ti]: outcome = 3
else: raise ValueError(f'{eid} trial {ti}: retained trial lacks outcome')
```

iii. The notes say the trial outcome identities are exact in the active sessions and correspond to the standard go/catch outcomes in the dataset.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps outcomes to the fixed integer order `hit=0`, `miss=1`, `false_alarm=2`, `correct_reject=3`, then broadcasts the chosen code across every time bin of that trial so it can coexist with the time-varying outputs in one homogeneous output matrix.

ii. 
```python
if flags['hit'][ti]: outcome = 0
elif flags['miss'][ti]: outcome = 1
elif flags['false_alarm'][ti]: outcome = 2
elif flags['correct_reject'][ti]: outcome = 3
...
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
...
'output_values': [
    ...,
    ['hit','miss','false_alarm','correct_reject'],
]
```

iii. The trajectory explicitly says the validator could not handle mixed static and time-varying outputs inside one trial array, so the AI chose to broadcast the static trial outcome over time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are handled mostly by exclusion rather than imputation. Entire experiments are removed if eye tracking is absent. Within a retained experiment, a trial is dropped if it has fewer than one 100 ms bin or if interpolated running/pupil still contains any non-finite values. `interp_valid` returns all-NaN if there are fewer than two valid source samples. Sessions with fewer than 2 usable trials raise an error and are not kept. For values outside the valid source support, `np.interp` uses the nearest valid endpoint rather than creating NaNs.

ii. 
```python
def has_eye(path):
    with h5py.File(path, 'r') as h:
        return 'acquisition/EyeTracking/pupil_tracking/data' in h

def interp_valid(t_new, t, x, valid=None):
    ...
    if good.sum() < 2:
        return np.full(len(t_new), np.nan, dtype=np.float32)
    return np.interp(t_new, t[good], x[good]).astype(np.float32)

if nbin < 1:
    continue
...
if not (np.isfinite(r).all() and np.isfinite(p).all()):
    continue
...
if len(neural) < 2:
    raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. The notes justify this by saying pupil is mandatory, so no-eye sessions cannot be reconstructed; blink or invalid pupil samples should be treated as missing; and missing behavior should not become a new output category. The trajectory also describes the no-eye exclusions as necessary because there were no sister planes from which pupil could be recovered.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are opening and reading each NWB file, parsing the stimulus presentation streams, and per-trial neural binning. Inside each trial, `event_bin_means` loops over every 100 ms bin and averages across neurons, and image identity is assigned with another per-bin loop.

ii. 
```python
with h5py.File(path, 'r') as h:
    ...
    flash_t, flash_name = presentation_streams(h)
    ...
    for ti in tids:
        ...
        n = event_bin_means(ev, ots, starts[ti], centers)
        ...
        for j, (a, b) in enumerate(zip(left, right)):
            ...
```

iii. In Step 6, the AI explicitly records that full PyNWB/AllenSDK object construction would be too slow and that the conversion should “stream one NWB at a time with h5py.” The code structure reflects that I/O and per-trial processing were expected bottlenecks.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial loop over `tids`, the per-bin loop inside `event_bin_means`, and the per-bin loop that checks whether each 100 ms center falls inside a 250 ms flash window. The final assembly loop over trials and outputs is also straightforward but still serial.

ii. 
```python
for ti in tids:
    ...

for j, (a, b) in enumerate(zip(lo, hi)):
    if b > a:
        out[:, j] = ...
    else:
        ...

for j, (a, b) in enumerate(zip(left, right)):
    if b > a:
        ...

for q, rl, pl in zip(prelim, run_lab, pup_lab):
    outputs.append(...)
```

iii. The notes mention avoiding “per-bin boolean masks over all native timestamps” and emphasize readability and streaming, but they do not claim these loops were fully optimized.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work. Every retained NWB is opened once in `image_vocabulary` to scan template names and then opened again in `process_experiment` for actual conversion. Within each experiment, the script first stores continuous running/pupil arrays in `prelim` and then makes a second pass to compute quintile labels and assemble outputs. It also traverses trial bins once for neural averaging and again for image labeling.

ii. 
```python
def image_vocabulary(meta, files):
    for eid in meta.ophys_experiment_id.astype(int):
        with h5py.File(files[eid], 'r') as h:
            ...

for k, row in meta.iterrows():
    eid = int(row.ophys_experiment_id)
    n, x, y, info = process_experiment(row, files[eid], imap, ...)

prelim.append((image, change, r, p, outcome, centers))
...
run_lab, run_edges = quintile_labels([q[2] for q in prelim])
pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
for q, rl, pl in zip(prelim, run_lab, pup_lab):
    outputs.append(...)
```

iii. There is no explicit note defending these repeated passes. The notes instead emphasize keeping memory bounded and streaming one file at a time, which explains why some recomputation was tolerated.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script constructs an empty input array for every trial even though the decoder input dimensionality is always zero. It also stores `centers` inside `prelim` mainly so later code can size the output and plots, but those centers are not saved in the final pickle. More broadly, `plot_processing` and some session metadata fields (`source_trial_indices`, per-session quintile edges, timing information) are useful for debugging but are not required by downstream decoder training.

ii. 
```python
neural.append(n); inputs.append(np.empty((0, nbin), dtype=np.float32))
prelim.append((image, change, r, p, outcome, centers)); kept_ids.append(int(ti))
...
info = {
    ...,
    'running_quintile_edges': run_edges.tolist(),
    'pupil_quintile_edges': pup_edges.tolist(),
    'source_trial_indices': kept_ids,
    'seconds': time.time()-t0
}
if show_plot:
    plot_processing(eid, prelim, neural, outputs, run_edges, pup_edges)
```

iii. The trajectory says these choices were made to satisfy the decoder/validator interface and to support processing diagnostics. There is no evidence that the AI considered them analytically necessary downstream.
