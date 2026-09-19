# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the local Visual Behavior Ophys release under `data/visual-behavior-ophys-1.1.0/` rather than via the AllenSDK project cache. It reads `ophys_experiment_table.csv`, intersects that table with the NWB files actually present on disk, filters to `passive == False`, and then loads each remaining `ophys_experiment_id` one at a time from its NWB file with `BehaviorOphysExperiment.from_nwb(...)`.

ii.
```python
def get_experiment_list(sample=False):
    et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0]) 
                  for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
    m = et[et['ophys_experiment_id'].isin(nwb_ids)]
    a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
    return a

def load_experiment(eid):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
    path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

iii. The justification in `CONVERSION_NOTES.md` and trajectory is that the provided dataset consists of local NWB files plus metadata CSVs, and that only active experiments should be used. The code also uses a preliminary `h5py` pass (`fast_collect_stats`) to gather global image names and running/pupil distributions before the full load/process pass.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the experiment table, then converted to strings and indexed in sorted order.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
...
asi.append(s2i[str(row['mouse_id'])])
```

iii. This follows the standard Allen metadata convention that `mouse_id` identifies the animal. No alternative subject grouping logic is used.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one decoder “session”, not each `ophys_session_id`. Multi-plane MESO sessions are therefore split into multiple separate decoder sessions, one per experiment / plane.

ii.
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ed = load_experiment(eid)
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
    asi.append(s2i[str(row['mouse_id'])])
```

iii. The trajectory gives the explicit rationale: in step 43 the AI states that “each experiment should be treated as a separate ‘session’ since each has its own set of neurons,” even though it also recognized that MESO experiments from one `ophys_session_id` share the same behavioral data.

## 1-d. How are the data split into trials?

i. Trials are taken from the AllenSDK `trials` table. For each kept trial, the AI uses the trial’s `start_time` and `stop_time` to select a variable-length subset of a session-wide resampled timeline, producing one neural matrix and one output matrix per trial.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
    ntl.append(nr[:, ti])
    otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The AI’s notes say the trial window is `start_time` to `stop_time` and that both Go and Catch trials should be included while Aborted and Auto-rewarded trials are excluded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch) & ~aborted & ~auto_rewarded`. Trials producing fewer than two resampled bins are skipped. Experiments with fewer than two surviving trials are skipped from the final dataset.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
if len(vt) == 0: return [], [], nn
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
    skipped += 1; continue
```

iii. The justification in `CONVERSION_NOTES.md` is “Include Go + Catch, exclude Aborted + Auto-rewarded.” The code does not add the reference solution’s explicit `change_time.notna()` filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `ds.events.events`, i.e. deconvolved calcium-event traces from the AllenSDK object, not from `dff_traces`.

ii.
```python
return {
    'ophys_timestamps': ds.ophys_timestamps.copy(),
    'events': np.vstack(ds.events.events.values).astype(np.float32),
    ...
}
...
ev = ed['events']
```

iii. The stated justification is in `CONVERSION_NOTES.md` Step 1 (“Paper uses ‘events’”) and trajectory step 43 (“Use events (calcium events), not dff_traces, as the paper specifies”).

## 2-b. How is the `neural` data processed?

i. Neural event traces are resampled from native ophys timestamps onto a fixed 93 ms session-wide bin grid. Each bin contains the mean event value over original frames falling inside that bin; empty bins are forward-filled from the previous bin.

ii.
```python
TARGET_BIN_SIZE = 0.093

def resample_session(events, ophys_ts, bc):
    h = TARGET_BIN_SIZE / 2
    nn, nb = events.shape[0], len(bc)
    out = np.zeros((nn, nb), dtype=np.float32)
    li = np.searchsorted(ophys_ts, bc - h, side='left')
    ri = np.searchsorted(ophys_ts, bc + h, side='left')
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
    return out
...
nr = resample_session(ev, ots, bc)
```

iii. The trajectory explains that the AI wanted a common time bin across CAM2P (~31 Hz) and MESO (~11 Hz) data, and chose ~93 ms so higher-rate data could be downsampled to the lower-rate regime.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no additional neuron-level filtering in the conversion code. Whatever cells are exposed in `ds.events` are used directly.

ii.
```python
'events': np.vstack(ds.events.events.values).astype(np.float32),
...
nn = ev.shape[0]
...
ntl.append(nr[:, ti])
```

iii. The notes say ROI filtering was already applied in the NWB/SDK release (`valid_roi` handling upstream), so the AI did not add another neuron-quality filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to trial boundaries derived from ophys time: a session-wide bin-center vector `bc` is built from trial start/stop times, resampled neural data are indexed by that vector, and each trial keeps bins with `start_time <= bc < stop_time`. Metadata labels the alignment event as trial start.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
'temporal_alignment_event': 'Trial start time',
'off_start': 0.0,
```

iii. The AI’s notes repeatedly say to align all streams to ophys timestamps and to use trial `start_time` to `stop_time` windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 93 ms bin size (`TARGET_BIN_SIZE = 0.093`). Yes: all experiments are rebinned onto that common resolution.

ii.
```python
TARGET_BIN_SIZE = 0.093
...
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
'time_bin_size': TARGET_BIN_SIZE * 1000,
'bin_size_seconds': TARGET_BIN_SIZE,
```

iii. The trajectory explicitly justifies this as a compromise between ~32 ms CAM2P and ~93 ms MESO data, and `CONVERSION_NOTES.md` Step 4 says “Resample all to ~93ms bins.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `stimulus_presentations`, specifically the `image_name` and `start_time` columns in the change-detection stimulus block, with omitted flashes removed.

ii.
```python
sp = ed['stimulus_presentations']
...
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
```

iii. The notes map `stimulus_presentations.image_name` to `output[0]: image_identity`, and exclude omitted stimuli because the task asked for the image shown during the non-grey screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first builds a global sorted image-name vocabulary across all experiments. Within each experiment, it maps stimulus names to integer codes, then for each resampled time bin uses the most recent non-omitted stimulus presentation start time to assign the current image identity.

ii.
```python
all_img = sorted(img_set)
...
n2i = {n: i for i, n in enumerate(imn)}
...
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. The AI’s justification is that `stimulus_presentations` gives the actual image stream over time, and the global mapping keeps labels consistent across experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same resampled bin-center timeline `bc` as the neural data, and per-trial output slices use the same `ti` indices as the neural trial matrices.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    ntl.append(nr[:, ti])
    otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The alignment justification is simply that all outputs are generated on the same ophys-derived common bin grid before trial slicing.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations` in the change-detection block, specifically the `is_change` flag and corresponding `start_time`.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
...
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The notes map `stimulus_presentations.is_change` to `output[1]: image_change`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary vector over the common bin grid and sets bins to 1 for 750 ms after each detected change event.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The rationale in the notes is that 750 ms is the image presentation interval used in the task and paper.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary with categories `no_change` and `change`.

ii.
```python
ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
      [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
```

iii. No additional thresholding beyond the binary windowing is used.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same resampled `bc` timeline as neural activity, then trial-sliced with the same `ti` indices.

ii.
```python
cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The AI’s justification is the same as for image identity: everything is aligned on one ophys-derived timebase before per-trial extraction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
'running_speed': ds.running_speed.copy(),
...
run = ed['running_speed']
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. This matches the AllenSDK’s standard running-wheel data stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the common 93 ms bin centers and then discretized into 5 global percentile bins. The percentile edges are computed in a first pass across all experiments using subsampled raw running traces from NWB files.

ii.
```python
rs = f['processing']['running']['speed']['data'][::10]
...
rbe = pct_bins(rs_all, 5)
...
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
rb = dig(ri, rbe)
```

iii. `CONVERSION_NOTES.md` says running speed should be “Interpolate + 5 percentile bins,” and the trajectory discusses using a common timebase first and consistent global bins second.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five percentile-based bins.

ii.
```python
def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e
...
rb = dig(ri, rbe)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes explicitly say “5 percentile bins.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same common resampled timebase `bc` that is used for neural activity and all other outputs.

ii.
```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
...
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The justification is consistency: once all streams are on `bc`, trial extraction uses the same indices for neural and behavioral outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI uses `ds.eye_tracking`, but specifically takes the `pupil_area` field rather than `pupil_width` / diameter.

ii.
```python
'eye_tracking': ds.eye_tracking.copy(),
...
eye = ed['eye_tracking']
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. The notes say the output is mapped from `eye_tracking.pupil_area`. The trajectory also mentions fixing the `h5py` stats pass to use the NWB `area` key.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is linearly interpolated onto the common bin grid and discretized into 5 global percentile bins. No blink filtering is applied in the final code.

ii.
```python
try:
    pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
    pa_all.append(np.array(pa, dtype=np.float64))
except:
    pass
...
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
```

iii. The justification in the notes is simply “Interpolate + 5 percentile bins.” There is no explicit justification for using area rather than width or for omitting blink removal.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five percentile-based bins.

ii.
```python
pbe = pct_bins(pa_all, 5)
...
pb = dig(pi, pbe)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes explicitly specify five percentile bins for the pupil output.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated onto the same common `bc` timebase as the neural data and are sliced into trials with the same indices.

ii.
```python
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The alignment logic is the same as for running speed and image identity.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. The notes map “trials outcome” to `output[4]: trial_outcome`, and the code assumes the Allen trial booleans are mutually exclusive after filtering.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI encodes the outcome into integers `0..3` in hit/miss/false-alarm/correct-reject order, then repeats that integer across every time bin of the trial so it can be stacked with the time-varying outputs.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
...
['hit','miss','false_alarm','correct_reject']
```

iii. In trajectory step 59 the AI explicitly reasons that a static per-trial output needs to be repeated across time so it can live inside a `(n_output, n_timepoints)` array alongside the time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing running or pupil samples lead to `NaN` after interpolation; these are later assigned to the middle discrete bin by `dig`. If there are fewer than two valid samples for interpolation, the entire interpolated series is filled with `NaN`. Missing pupil data during the first-pass stats collection are silently ignored by `try/except`. Trials with fewer than two bins are dropped, and experiments with fewer than two surviving trials are skipped. The final code does not wrap experiment loading/processing in a `try/except`.

ii.
```python
def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
...
ri = interpolate.interp1d(...)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
pi = interpolate.interp1d(...)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
...
try:
    pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
    pa_all.append(np.array(pa, dtype=np.float64))
except:
    pass
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    skipped += 1; continue
```

iii. The notes mention a few fixes during development (output dtype, pupil field), but the final code’s missing-data behavior is mostly implicit rather than explicitly justified.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the per-experiment NWB loads via `pynwb` / `BehaviorOphysExperiment.from_nwb(...)`. The AI also added a separate first pass over all NWB files for `fast_collect_stats`, but its notes estimate the main load step is still the dominant cost.

ii.
```python
def load_experiment(eid):
    ...
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
...
imn, rs_all, pa_all = fast_collect_stats(el)
...
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
```

iii. `CONVERSION_NOTES.md` Step 7 reports roughly 5.5 s average load time per experiment and much smaller processing/stat-collection costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization candidates are:
- the per-bin loop in `resample_session`
- the per-change loop that writes `image_change`
- the per-trial `iterrows()` loop in `process_experiment`
- the full-dataset `iterrows()` loops in `fast_collect_stats` and the main processing pass

ii.
```python
for b in range(nb):
    if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
    elif b > 0: out[:, b] = out[:, b-1]
...
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
for _, t in vt.iterrows():
    ...
for i, (_, row) in enumerate(exp_list.iterrows()):
    ...
for i, (_, row) in enumerate(el.iterrows()):
```

iii. The AI did not explicitly justify keeping these loops, but `CONVERSION_NOTES.md` frames the implementation as “fast enough” because expensive I/O dominates runtime and the first-pass `h5py` optimization already reduced some overhead.

## 9-c. What processing does the code repeat multiple times?

i. The code performs two passes over every experiment file. First, `fast_collect_stats` opens every NWB with `h5py` to gather image names and global running/pupil distributions. Then the main loop loads every experiment again with `pynwb` / AllenSDK for actual conversion. The image/stimulus information is therefore read twice.

ii.
```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
for i, (_, row) in enumerate(exp_list.iterrows()):
    with h5py.File(path, 'r') as f:
        ...
...
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
```

iii. The notes present this as an optimization tradeoff: `h5py` is used for a fast statistics pass, then AllenSDK loading is used for the richer conversion pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several full-session arrays on the common bin grid (`nr`, `rb`, `pb`, `ib`, `cb`) and then only keeps per-trial subsets. It also loads `metadata` from each experiment but does not use it downstream, and the first-pass `fast_collect_stats` scans all NWB interval groups for image names even though later processing only uses the change-detection block.

ii.
```python
return {
    ...
    'metadata': dict(ds.metadata),
}
...
nr = resample_session(ev, ots, bc)
...
rb = dig(ri, rbe)
pb = dig(pi, pbe)
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    ntl.append(nr[:, ti])
    otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
...
for k in f['intervals']:
    if 'image_name' in f['intervals'][k]:
        ...
```

iii. There is no explicit justification beyond convenience and speed of implementation.
