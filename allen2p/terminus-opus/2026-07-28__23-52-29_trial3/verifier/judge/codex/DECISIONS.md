# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local metadata CSV `ophys_experiment_table.csv`, intersects it with the NWB files actually present on disk, filters to experiments with `passive == False`, and then loads each `ophys_experiment_id` NWB file individually with `pynwb` plus `BehaviorOphysExperiment.from_nwb()`. It does not use the Allen SDK project cache or `project_code == 'VisualBehavior'`.

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

iii. In `CONVERSION_NOTES.md`, the AI says `BehaviorOphysExperiment.from_nwb()` is the key loading function and that it intentionally used `h5py` for fast stats collection and a load-process-release pattern for efficiency. It also states that it is using the "active experiments" present in the local data files.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values found in the filtered experiment list.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
...
asi.append(s2i[str(row['mouse_id'])])
```

iii. The notes report 38 subjects and treat mouse IDs as the subject identifiers. No additional justification beyond using the metadata table is recorded.

## 1-c. How are the data split into sessions?

i. The AI effectively treats each `ophys_experiment_id` as one session. It iterates row-by-row over the filtered experiment table and appends one session entry per experiment, rather than grouping multiple experiments by `ophys_session_id`.

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

iii. The notes describe "202 active experiments" and later equate that count with sessions, which shows the AI decided to work at experiment level rather than reconstructing SDK sessions.

## 1-d. How are the data split into trials?

i. Trials are taken from the Allen SDK `trials` table after filtering to go-or-catch, non-aborted, non-auto-rewarded rows. The AI builds one session-wide array of 93 ms bin centers from the earliest valid trial start to the latest valid trial stop, then defines each trial by masking those bins between the row's `start_time` and `stop_time`.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
```

iii. In the notes, the AI says trial curation is "Include Go + Catch, exclude Aborted + Auto-rewarded" and lists "session-wide resampling then trial extraction" as a deliberate optimization.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch)`, excluding `aborted` and `auto_rewarded`. Trials with fewer than 2 resampled bins are dropped. Experiments with fewer than 2 remaining trials are skipped from the final dataset.

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

iii. The notes explicitly justify including Go and Catch and excluding Aborted and Auto-rewarded trials. No separate justification is recorded for omitting trials with fewer than 2 bins, but it matches the decoder requirement that sessions need at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `ds.events.events`, i.e. deconvolved calcium events, not from `dff_traces`.

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

iii. The notes repeatedly state that the paper "uses events (deconvolved calcium events) for neural analysis" and that this choice matches the paper better than dF/F.

## 2-b. How is the `neural` data processed?

i. The events matrix is resampled onto a fixed 93 ms grid by averaging all event frames whose original ophys timestamps fall within each bin. If a bin has no samples, the previous bin value is copied forward. Trial matrices are then sliced from that resampled session matrix.

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
...
ntl.append(nr[:, ti])
```

iii. The notes say the AI had to reconcile mixed 11 Hz and 31 Hz frame rates, so it "resample[d] all to ~93ms bins." It also records a sanity check comparing original and converted trial means after resampling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level filtering in the code. All neurons present in `ds.events.events` are kept.

ii.
```python
return {
    'ophys_timestamps': ds.ophys_timestamps.copy(),
    'events': np.vstack(ds.events.events.values).astype(np.float32),
    ...
}
```

iii. The notes claim that `valid_roi` filtering has already been applied in the NWB files and that ROI filtering comes from the Allen pipeline, so no further cell filtering is needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start time in the sense that each trial keeps the 93 ms bins whose centers fall between that trial's `start_time` and `stop_time`. The metadata also labels the alignment event as "Trial start time."

ii.
```python
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
ntl.append(nr[:, ti])
...
'temporal_alignment_event': 'Trial start time',
'off_start': 0.0,
```

iii. The notes describe trial extraction as start/stop masking after session-wide resampling and later say the temporal alignment sanity check passed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 93 ms time bin for all experiments and trials. Yes, temporal rebinning is applied.

ii.
```python
TARGET_BIN_SIZE = 0.093
...
print(f"Mode: {'sample' if sample else 'full'}, Output: {outf}, Bin: {TARGET_BIN_SIZE*1000:.1f}ms")
...
'time_bin_size': TARGET_BIN_SIZE * 1000,
'bin_size_seconds': TARGET_BIN_SIZE,
```

iii. The notes explicitly justify this as the resolution needed to reconcile experiments acquired at both about 11 Hz and about 31 Hz, and they cite the paper/whitepaper frame-rate discrepancy as the reason.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `stimulus_presentations.image_name`, restricted to stimulus blocks whose `stimulus_block_name` contains `change_detection`, with omitted stimuli removed.

ii.
```python
sp = ed['stimulus_presentations']
...
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
```

iii. The notes map `stimulus_presentations.image_name` directly to the `image_identity` output and say image identity should be time-varying.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first builds a global sorted list of image names across all experiments using a fast `h5py` scan. During processing, each 93 ms bin is assigned the most recent non-omitted image whose `start_time` is at or before that bin, and the image names are converted to integer codes.

ii.
```python
all_img = sorted(img_set)
...
n2i = {n: i for i, n in enumerate(imn)}
...
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. The notes say image identity is categorical and time-varying, and later report a sanity check that 16 unique images were correctly mapped.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same session-wide 93 ms bin centers used for the resampled neural data, and each trial slices the same bin indices from both arrays.

ii.
```python
nr = resample_session(ev, ots, bc)
...
sidx = np.searchsorted(ist, bc, side='right') - 1
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The notes frame the whole pipeline as "session-wide resampling then trial extraction," so the intended justification is that all streams share one common binned timebase before trials are cut out.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the corresponding `start_time` values within the change-detection stimulus block.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
...
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The notes map `stimulus_presentations.is_change` directly to the `image_change` output and mention the 750 ms image-presentation interval as the relevant temporal structure.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every change event in the session, the AI marks `1` in all 93 ms bins from the change start time through the next 750 ms, and `0` elsewhere.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The notes justify the 750 ms interval by referring to 250 ms image presentation plus 500 ms inter-stimulus interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already represented as a binary categorical variable: `0` for no change and `1` for change. No additional thresholding is applied.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
      [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
```

iii. The justification is implicit in the task mapping table in the notes, which lists `stimulus_presentations.is_change` as a binary time-varying output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same 93 ms session-wide timebase as the neural data and then sliced by the same trial masks.

ii.
```python
nr = resample_session(ev, ots, bc)
...
cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
ti = np.where(tm)[0]
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The notes treat all outputs as sharing the same session-wide resampling scheme before trial extraction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from the `running_speed` table, specifically its `timestamps` and `speed` columns.

ii.
```python
run = ed['running_speed']
...
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. The notes list `running_speed` as the source variable and describe it as an output to be interpolated and discretized.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the 93 ms session-wide bin centers. The AI computes global percentile bin edges from subsampled running-speed data collected across all experiments, then digitizes each interpolated value into one of five bins.

ii.
```python
rs = f['processing']['running']['speed']['data'][::10]
...
rbe = pct_bins(rs_all, 5)
...
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
rb = dig(ri, rbe)
```

iii. The notes explicitly say `running_speed` is transformed by "Interpolate + 5 percentile bins" and describe the first-pass `h5py` scan as a speed optimization.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses five percentile bins computed from all collected running-speed values. It forces the outer bin edges to `-inf` and `inf`, digitizes values into bins `0` through `4`, and maps NaNs to the middle bin.

ii.
```python
def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    if len(v2) == 0: return np.linspace(-1, 1, n+1)
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
```

iii. The notes say the output should be "5 percentile bins." They do not separately justify the NaN-to-middle-bin choice.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 93 ms `bc` timebase used for the neural data, then sliced with the same trial indices.

ii.
```python
ri = interpolate.interp1d(...)(bc)
rb = dig(ri, rbe)
...
ti = np.where(tm)[0]
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The notes describe this as part of the common session-wide resampling/alignment strategy.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives this output from `eye_tracking.pupil_area` and `eye_tracking.timestamps`, not from `pupil_width`.

ii.
```python
eye = ed['eye_tracking']
...
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. The notes' mapping table explicitly says `eye_tracking.pupil_area` is the source for `pupil_diameter`, and later notes that a bug was fixed to use the NWB `area` key.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is linearly interpolated onto the 93 ms bin centers, then discretized using global percentile bins computed from subsampled pupil-area values collected from all experiments. No blink-removal step is implemented.

ii.
```python
pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
...
pbe = pct_bins(pa_all, 5)
...
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
```

iii. The notes say `pupil_diameter` should be "Interpolate + 5 percentile bins" and that they deliberately switched the fast stats pass to the `area` key. They do not mention blink filtering.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI uses the same five-percentile-bin procedure as for running speed, again mapping NaNs to the middle bin.

ii.
```python
pbe = pct_bins(pa_all, 5)
...
pb = dig(pi, pbe)
...
ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
      [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
```

iii. The notes specify five percentile bins but do not separately justify the missing-data treatment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated onto the same 93 ms `bc` timebase as the neural data and then trial-sliced with the same indices.

ii.
```python
pi = interpolate.interp1d(...)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
...
ti = np.where(tm)[0]
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The notes again justify this through the common session-wide resampling approach.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean `hit`, `miss`, `false_alarm`, and `correct_reject` columns in the trials table.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
      [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
```

iii. The notes report those four categories as the trial-outcome values and compare their counts against metadata tables as a sanity check.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps each trial to one integer outcome code and repeats that code across all time bins in the trial output matrix.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The notes justify this as a per-trial categorical output and record that the decoder expects integer-coded outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases: missing pupil-area stats during the fast pass are silently skipped; if a variable has no non-NaN values, fallback edges are generated; if running or pupil has fewer than two valid samples in an experiment, the interpolated series is set to all-NaN and later digitized to the middle bin; trials with fewer than two bins are skipped; experiments with fewer than two kept trials are skipped.

ii.
```python
try:
    pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
    pa_all.append(np.array(pa, dtype=np.float64))
except:
    pass

def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    if len(v2) == 0: return np.linspace(-1, 1, n+1)
...
ri = interpolate.interp1d(...)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
pi = interpolate.interp1d(...)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    ... continue
```

iii. The notes say missing data should be handled sensibly and emphasize efficiency and robustness, but they do not give a detailed conceptual rationale for each fallback beyond recording one pupil-area bug fix.

## 9-a. What are the most time-consuming steps of the code?

i. According to the AI's notes, loading each experiment through the SDK/NWB layer is the dominant cost, while processing after load is much cheaper. The code also performs a full first pass over all files for stats collection.

ii.
```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    t0 = time.time()
    ed = load_experiment(eid)
    tl = time.time()
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
    tp = time.time()
    print(f"  [{i+1}/{len(el)}] Exp {eid}: {nn}n {len(ntl)}t load={tl-t0:.1f}s proc={tp-tl:.1f}s")
```

iii. In `CONVERSION_NOTES.md`, the AI estimates about 5.5 s per experiment for loading, about 0.5 s for processing, and about 0.1 s for the `h5py` stats pass, and explicitly calls loading the expensive step.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several explicit Python loops in place: the per-bin loop in `resample_session`, the per-change-event loop for `image_change`, the per-trial loop over `vt.iterrows()`, and the per-file loop in `fast_collect_stats`. The notes mention optimizations, but not further vectorization of these loops.

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
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
...
for i, (_, row) in enumerate(exp_list.iterrows()):
    ...
```

iii. The only recorded justification is indirect: the notes say the AI prioritized `h5py` fast stats collection, session-wide resampling, and memory-efficient one-experiment-at-a-time processing.

## 9-c. What processing does the code repeat multiple times?

i. The code makes two passes over every experiment file. First, `fast_collect_stats()` opens each NWB with `h5py` to collect image names and global running/pupil statistics. Then `load_experiment()` opens the same NWB again through `pynwb` and AllenSDK to load the full data for actual conversion.

ii.
```python
def fast_collect_stats(exp_list):
    for i, (_, row) in enumerate(exp_list.iterrows()):
        ...
        with h5py.File(path, 'r') as f:
            ...

def load_experiment(eid):
    ...
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

iii. The notes explicitly present this first pass as an optimization for global stats collection, but it still means every file is read twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads or computes several things that are not used in the final pickle: `metadata` from each experiment is loaded but dropped, equipment counts are printed only for logging, the full fast-pass running and pupil arrays are created only to compute global bin edges and then discarded, and the interval scan collects image names from all interval groups before most of that information is collapsed into a sorted image list.

ii.
```python
return {
    ...
    'metadata': dict(ds.metadata),
}
...
print(f"  Equipment: {a['equipment_name'].value_counts().to_dict()}")
...
imn, rs_all, pa_all = fast_collect_stats(el)
rbe = pct_bins(rs_all, 5)
pbe = pct_bins(pa_all, 5)
del rs_all, pa_all
...
for k in f['intervals']:
    if 'image_name' in f['intervals'][k]:
        names = f['intervals'][k]['image_name'][:]
        ...
        img_set.update(...)
```

iii. The notes justify some of this as speed-oriented preprocessing, but they also emphasize memory cleanup afterward, which implies these intermediate products are not needed downstream once the final edges and mappings are built.
