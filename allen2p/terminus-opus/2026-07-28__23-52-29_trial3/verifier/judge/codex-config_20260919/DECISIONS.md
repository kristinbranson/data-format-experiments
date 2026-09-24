# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local experiment metadata CSV, intersects it with locally present NWB experiment IDs, excludes passive experiments, and loads each selected NWB independently through `pynwb` and `BehaviorOphysExperiment.from_nwb`. Full mode selected 202 of 284 local experiments.

ii.
```python
et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0])
              for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
m = et[et['ophys_experiment_id'].isin(nwb_ids)]
a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
```

iii. The notes say the available release contains 284 NWBs and 202 active experiments. The AI chose active experiments because the requested task is Visual Behavior and used the SDK object to expose synchronized tables and traces.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique string-valued `mouse_id`s from the selected experiment rows; each retained experiment receives the corresponding subject index.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
asi.append(s2i[str(row['mouse_id'])])
```

iii. The notes identify 38 mice and use the metadata's unique mouse identifier.

## 1-c. How are the data split into sessions?

i. Every `ophys_experiment_id` is emitted as a separate decoder session. The code does not group simultaneous imaging planes by `ophys_session_id`; thus 202 experiments become 202 output sessions although the notes report 174 unique ophys sessions.

ii.
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ed = load_experiment(eid)
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
```

iii. The AI consistently calls an experiment a session and emphasizes processing “all active experiments.” It also notes that paper analyses used individual imaging planes, which appears to motivate plane-level units.

## 1-d. How are the data split into trials?

i. The SDK `trials` table defines trials. For each accepted row, fixed 93-ms bin centers satisfying `start_time <= center < stop_time` are extracted, so trial lengths remain variable.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
    ntl.append(nr[:, ti])
```

iii. The notes say Go and Catch trials are included and trials span their full experimental start/stop window.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be Go or Catch, must not be aborted or auto-rewarded, and must contain at least two resampled bins. Entire experiments with fewer than two retained trials are skipped.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
if len(ti) < 2: continue
...
if len(ntl) < 2:
    skipped += 1; continue
```

iii. This directly follows the requested trial curation and the decoder's minimum-two-trials requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the SDK's deconvolved calcium `events`, not dF/F.

ii.
```python
'events': np.vstack(ds.events.events.values).astype(np.float32),
```

iii. The notes state that the paper uses deconvolved calcium events and that valid-ROI filtering is already represented in the NWB/SDK object.

## 2-b. How is the `neural` data processed?

i. Events from one imaging experiment are averaged into uniform 93-ms windows centered on a session-wide time grid. Empty bins copy the prior bin (the first stays zero). No normalization is applied and planes are not combined.

ii.
```python
li = np.searchsorted(ophys_ts, bc - h, side='left')
ri = np.searchsorted(ophys_ts, bc + h, side='left')
for b in range(nb):
    if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
    elif b > 0: out[:, b] = out[:, b-1]
```

iii. The AI wanted a common resolution for 11-Hz MESO and 31-Hz CAM2P recordings and chose approximately the slower rig's native 93-ms period. It reports sanity-checking means and frame-count ratios after rebinning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit per-cell filtering in the conversion. All cells exposed by `ds.events` are retained.

ii.
```python
ev = ed['events']
nn = ev.shape[0]
```

iii. The notes say NWB cells already passed `valid_roi`/multi-label ROI filtering, so no second filter was applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is placed on an ophys-time grid, then each trial is sliced from trial start (inclusive) to stop (exclusive). Metadata calls the alignment event “Trial start time.”

ii.
```python
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ntl.append(nr[:, np.where(tm)[0]])
```

iii. The notes say all streams are synchronized on ophys time and confirm that converted trial duration scales appropriately after resampling.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 93 ms. Native event frames are averaged within each 93-ms interval.

ii.
```python
TARGET_BIN_SIZE = 0.093
...
'time_bin_size': TARGET_BIN_SIZE * 1000,
```

iii. The AI chose 93 ms to reconcile 11-Hz and 31-Hz equipment and match the MESO frame rate described in the whitepaper.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `stimulus_presentations.image_name`, restricted during processing to the `change_detection` stimulus block and excluding omitted presentations. The global vocabulary is collected more broadly from every NWB interval table containing `image_name`.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
```

iii. The mapping plan explicitly assigns `stimulus_presentations.image_name` to time-varying image identity, preserving actual presentation timing.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Unique names are sorted into global integer categories. For each bin center, the code assigns the most recently started non-omitted change-detection image; bins preceding the first such image default to category 0.

ii.
```python
n2i = {n: i for i, n in enumerate(imn)}
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. Global sorted codes give consistent labels across image sets and sessions. The full run found 16 image identities.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels are evaluated at the same 93-ms bin centers as neural activity and then sliced with the identical per-trial indices.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The justification is shared ophys-clock timing and a single session-wide grid for every stream.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses `stimulus_presentations.is_change` and presentation `start_time` within the change-detection block.

ii.
```python
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The notes map stimulus presentation change flags directly to the requested binary output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes zeros and marks the 750 ms following every real change presentation as one.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The notes identify 250-ms images plus 500-ms inter-stimulus periods, hence one 750-ms presentation cycle.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: category 1 inside `[change_start, change_start + 0.750)` and category 0 otherwise.

ii.
```python
ov = [imn, ['no_change','change'], ...]
```

iii. No continuous thresholding is needed because the source is a boolean change flag.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change windows are evaluated on the common bin-center grid and indexed by the same trial mask as neural activity.

ii.
```python
cb[(bc>=ct)&(bc<ct+0.750)] = 1
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. This is justified by the common synchronized timestamps and grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `dataset.running_speed.timestamps` and `dataset.running_speed.speed`; preliminary percentile statistics are read directly from NWB `processing/running/speed/data`.

ii.
```python
run = ed['running_speed']
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. The SDK running-speed table is the standard processed wheel signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Every tenth raw running sample from every selected experiment estimates global quintile edges. Non-NaN speed samples are linearly interpolated to neural bin centers and digitized.

ii.
```python
rs = f['processing']['running']['speed']['data'][::10]
...
rbe = pct_bins(rs_all, 5)
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear',
    bounds_error=False, fill_value=np.nan)(bc)
rb = dig(ri, rbe)
```

iii. The AI cites faster statistics collection and percentile bins for balanced classes, with one global set of thresholds for consistency.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Percentiles 0, 20, 40, 60, 80, and 100 define five categories; endpoints become infinities. `np.digitize` plus clipping produces labels 0–4, and missing values become middle category 2.

ii.
```python
e = np.percentile(v2, np.linspace(0, 100, n+1))
e[0] = -np.inf; e[-1] = np.inf
...
b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
b[np.isnan(v)] = (len(e) - 1) // 2
```

iii. Quintiles were selected because the instruction requires five equal-percentile bins; middle-bin imputation avoids an invalid category.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated at the exact neural bin centers and sliced with the same trial index vector.

ii.
```python
ri = interpolate.interp1d(...)(bc)
...
np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...])
```

iii. The AI relies on the dataset's hardware-synchronized timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the target name, the AI uses `eye_tracking.pupil_area`; preliminary percentile values come from the NWB pupil-tracking `area` dataset.

ii.
```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
...
pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
```

iii. The notes explicitly map pupil area to pupil diameter and record fixing the raw HDF5 key to use area rather than a width/height proxy.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Every tenth area sample estimates global quintiles. Non-NaN areas are linearly interpolated to bin centers and digitized. Blink flags are not used.

ii.
```python
pbe = pct_bins(pa_all, 5)
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear',
    bounds_error=False, fill_value=np.nan)(bc)
pb = dig(pi, pbe)
```

iii. The AI applies the same global, balanced discretization scheme as running and treats area as the pupil-size signal.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global area percentiles define five bins exactly as for running; NaNs become category 2.

ii.
```python
pbe = pct_bins(pa_all, 5)
pb = dig(pi, pbe)
```

iii. The instruction requests five equal-percentile categories, and global edges keep codes consistent across experiments.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated to the same 93-ms centers and sliced by the same trial indices.

ii.
```python
pi = interpolate.interp1d(...)(bc)
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The notes rely on synchronized eye and ophys clocks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table booleans `hit`, `miss`, and `false_alarm`; all remaining retained trials are assigned correct reject.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else
     (2 if t['false_alarm'] else 3))
```

iii. The notes identify the four canonical outcomes hit, miss, false alarm, and correct reject.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes are encoded 0–3 in the documented order and repeated at every time bin of the trial.

ii.
```python
np.full(len(ti), oc, dtype=np.int64)
...
['hit','miss','false_alarm','correct_reject']
```

iii. Repetition makes the static trial label compatible with the common time-varying output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. NaNs are removed before constructing interpolators and percentile edges; interpolation outside coverage yields NaN, later mapped to middle bin 2. Missing/empty pupil statistics fall back to synthetic edges. Trials with fewer than two bins and experiments with fewer than two trials are skipped. Empty neural bins copy the previous value. There is no per-experiment exception handler, and absent expected NWB structures can stop conversion (except missing pupil area during the statistics pass, which is ignored).

ii.
```python
vm = ~np.isnan(rsp)
...
b[np.isnan(v)] = (len(e) - 1) // 2
if len(v2) == 0: return np.linspace(-1, 1, n+1)
if len(ti) < 2: continue
...
except:
    pass
```

iii. The stated goal is robust categorical output and memory-efficient processing. The notes do not justify the middle-bin imputation or broad exception during pupil statistics specifically.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each NWB through pynwb/AllenSDK dominates (estimated 5.5 s per experiment, about 18 minutes total); processing is about 0.5 s per experiment. Pickling the roughly 3-GB result is also substantial but was not highlighted.

ii.
```python
ed = load_experiment(eid)
ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
```

iii. The notes provide measured estimates of 20 s for fast stats, 18 min for SDK loads, and 1.5 min for processing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The 93-ms neural-bin loop could use reduce-at/cumulative sums; the per-change boolean assignment could be expressed through vectorized interval searches; trial `iterrows` and global experiment/statistics loops could be batched or parallelized. Some trial-boundary and image operations are already vectorized.

ii.
```python
for b in range(nb):
    if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
...
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
for _, t in vt.iterrows():
```

iii. The AI did not explicitly discuss remaining vectorization opportunities, but it says session-wide resampling was chosen over repeated per-trial resampling for speed.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened first with h5py for image/running/pupil statistics and again with pynwb/SDK for conversion. Percentile statistics include all raw samples, then the same behavioral streams are interpolated during processing. Image names are scanned from all interval tables before change-detection presentations are scanned again.

ii.
```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
ed = load_experiment(eid)
```

iii. The AI accepts this two-pass design because direct h5py statistics are reported as about 0.1 s rather than roughly 5 s per SDK load and allow streaming one full experiment at a time.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_experiment` copies `metadata` but no downstream function reads it. `fast_collect_stats` scans image names in every interval group although processing only uses change-detection presentations. In optional plotting, figures are diagnostic and discarded by decoding. Session-wide arrays outside retained trial windows are computed and then discarded, though computing them once simplifies extraction.

ii.
```python
'metadata': dict(ds.metadata),
...
for k in f['intervals']:
    if 'image_name' in f['intervals'][k]:
```

iii. These costs are undocumented as necessary; the AI's stated optimization is mainly minimizing expensive SDK loads and releasing each experiment promptly.
