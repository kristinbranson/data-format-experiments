# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads the experiment roster from `ophys_experiment_table.csv`, intersects it with locally available NWB files, keeps only non-passive experiments, does a fast first pass with `h5py` to collect image names and global running/pupil statistics, and then loads each experiment's full SDK object with `BehaviorOphysExperiment.from_nwb()`.

ii. 
```python
def get_experiment_list(sample=False):
    et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0]) 
                  for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
    m = et[et['ophys_experiment_id'].isin(nwb_ids)]
    a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
```

```python
def load_experiment(eid):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
    path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

iii. The justification appears in `CONVERSION_NOTES.md` Step 6 and trajectory steps 87-92: the agent explicitly chose a fast `h5py` pass because SDK loading was the runtime bottleneck, then a second SDK-based pass for the actual conversion.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the filtered experiment table. The subject list is sorted, and each kept experiment/session stores the corresponding integer index into that list.

ii. 
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
...
asi.append(s2i[str(row['mouse_id'])])
```

iii. `CONVERSION_NOTES.md` Step 9 reports 38 subjects, and trajectory steps 27-28 discuss selecting all active experiments and using `mouse_id` as the subject identifier.

## 1-c. How are the data split into sessions?

i. The script treats each `ophys_experiment_id` as one decoder session, even when multiple experiments belong to the same `ophys_session_id` in MESO recordings. Each experiment therefore becomes its own entry in `neural`, `input`, `output`, and `brain_region_idx`.

ii. 
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
    asi.append(s2i[str(row['mouse_id'])])
    abri.append(np.full(nn, r2i[row['targeted_structure']], dtype=np.int64))
```

iii. Trajectory step 43 states the rationale directly: MESO experiments share behavior within an `ophys_session`, but each experiment has its own neurons, so the agent chose `ophys_experiment = session` for the decoder format.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. After filtering, each retained trial uses `start_time` and `stop_time` to slice a session-wide rebinned time axis into one neural matrix and one output matrix per trial.

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

iii. `CONVERSION_NOTES.md` Steps 3 and 5, and trajectory steps 11 and 31, show the agent understood trials as the experiment-defined `trials` rows with `start_time`/`stop_time`.

## 1-e. How are trials filtered based on quality controls?

i. The code keeps only Go and Catch trials and excludes Aborted and Auto-rewarded trials, exactly as instructed. It also drops trials with fewer than two rebinned time points, and entire experiments are skipped if fewer than two valid trials remain.

ii. 
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
if len(vt) == 0: return [], [], nn
...
if nb < 2: return [], [], nn
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
```

iii. `CONVERSION_NOTES.md` Step 3 says “Include Go + Catch, exclude Aborted + Auto-rewarded,” and trajectory steps 11, 17, 22, and 31 repeat that filtering rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the SDK `events` table, specifically the per-cell `events` traces, together with `ophys_timestamps` to place those traces on time.

ii. 
```python
return {
    'ophys_timestamps': ds.ophys_timestamps.copy(),
    'events': np.vstack(ds.events.events.values).astype(np.float32),
    ...
}
```

iii. `CONVERSION_NOTES.md` Step 1 says the paper uses deconvolved calcium events, and trajectory steps 12, 43, and 58 explicitly choose events instead of `dff_traces`.

## 2-b. How is the `neural` data processed?

i. The event traces are stacked into a `neurons x time` float32 matrix, then averaged into fixed 93 ms bins on a session-wide grid before trial extraction.

ii. 
```python
TARGET_BIN_SIZE = 0.093
...
def resample_session(events, ophys_ts, bc):
    h = TARGET_BIN_SIZE / 2
    nn, nb = events.shape[0], len(bc)
    out = np.zeros((nn, nb), dtype=np.float32)
    li = np.searchsorted(ophys_ts, bc - h, side='left')
    ri = np.searchsorted(ophys_ts, bc + h, side='left')
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
```

iii. Trajectory steps 44 and 58 explain the main justification: the dataset mixes ~31 Hz CAM2P and ~11 Hz MESO experiments, so the agent chose a common ~93 ms bin size by downsampling faster sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script applies no explicit neural QC inside `convert_data.py`. It relies on the SDK/NWB contents returned by `BehaviorOphysExperiment.from_nwb()`, which already expose the filtered cell set, and otherwise keeps every event trace that is loaded.

ii. 
```python
with pynwb.NWBHDF5IO(path, 'r') as io:
    nwb = io.read()
    ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
    return {
        'events': np.vstack(ds.events.events.values).astype(np.float32),
```

iii. `CONVERSION_NOTES.md` Step 1 says “Cell filtering (valid_roi) already applied in NWB files,” and the agent did not add any extra filtering in the conversion code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not keep native ophys frames per trial. Instead it builds one continuous 93 ms bin grid from the first valid trial start to the last valid trial stop, rebins events onto that grid, and then slices each trial by its `start_time` and `stop_time`. The metadata labels the alignment event as “Trial start time.”

ii. 
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
nr = resample_session(ev, ots, bc)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'Trial start time',
    'off_start': 0.0, 'off_end': None,
```

iii. Trajectory steps 31, 43, and 58 say the goal was to “align all data to ophys timestamps,” but the implemented version operationalizes that as a rebinned session grid and trial-start slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed bin size of 0.093 s, stored as 93 ms in metadata. Yes: CAM2P data are downsampled into that bin size by averaging events in each bin; MESO data are approximately at that native rate already.

ii. 
```python
TARGET_BIN_SIZE = 0.093
...
'time_bin_size': TARGET_BIN_SIZE * 1000,
'bin_size_seconds': TARGET_BIN_SIZE,
```

```python
if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
```

iii. The rationale is explicit in trajectory steps 44 and 58 and in `CONVERSION_NOTES.md` Step 4: the agent wanted one common bin size across experiments with two native frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the SDK `stimulus_presentations` table, using `image_name` from rows whose `stimulus_block_name` contains `change_detection`, while excluding `omitted` flashes.

ii. 
```python
sp = ed['stimulus_presentations']
...
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `stimulus_presentations.image_name` to `output[0]: image_identity`, and trajectory step 26 identifies these exact stimulus columns as the relevant raw variables.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds one global categorical mapping of all image names across experiments, then assigns each time bin the most recent non-omitted change-detection image that started before that bin. Bins before the first image default to category 0.

ii. 
```python
n2i = {n: i for i, n in enumerate(imn)}
...
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. `CONVERSION_NOTES.md` Step 5 describes the variable as categorical and time-varying, and trajectory steps 68 and 86-87 show the agent deliberately created a global image vocabulary spanning both image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is sampled on the same rebinned session time axis `bc` used for neural data, by carrying the latest image category forward to each bin and then slicing those bins by trial.

ii. 
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The agent’s justification is implicit in trajectory steps 31, 44, and 58: all behavioral outputs were intended to live on the same common time grid as rebinned neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and `stimulus_presentations.start_time`, restricted to the `change_detection` stimulus block.

ii. 
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
...
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. `CONVERSION_NOTES.md` Step 5 maps `stimulus_presentations.is_change` to the image-change output, and trajectory step 26 identifies `is_change` and presentation timing as part of the stimulus table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every change presentation, the script marks all bins from that presentation’s `start_time` through the next 0.750 s as “change”; all other bins are “no_change.”

ii. 
```python
cts = cd[cd['is_change']==True]['start_time'].values
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. `CONVERSION_NOTES.md` Step 3 notes the 750 ms image presentation interval, and trajectory steps 14 and 31 use that interval as the behavioral timing unit.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is applied. The variable is already binary in the source table and is exported as categories `no_change` and `change`.

ii. 
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
ov = [imn, ['no_change','change'], ...]
```

iii. The justification is simply that the SDK already exposes a boolean `is_change`, so the agent preserved it as a binary decoder target.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change labels are written onto the same session-wide 93 ms bin centers used for the rebinned neural traces, then trial-sliced with the same `start_time`/`stop_time` mask.

ii. 
```python
cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
```

iii. This follows the agent’s general alignment choice from trajectory steps 44 and 58: put all outputs onto the shared rebinned ophys time axis.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the SDK `running_speed` table, specifically its `timestamps` and `speed` columns.

ii. 
```python
run = ed['running_speed']
...
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. `CONVERSION_NOTES.md` Step 1 identifies `dataset.running_speed`, and trajectory step 26 states that running speed is available as timestamps plus speed sampled at about 60 Hz.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script linearly interpolates valid running-speed samples onto the common 93 ms session bins and then discretizes those interpolated values.

ii. 
```python
vm = ~np.isnan(rsp)
ri = interpolate.interp1d(
    rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan
)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
rb = dig(ri, rbe)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Interpolate + 5 percentile bins,” and trajectory steps 31 and 44 explain that running speed needed to be put onto the neural time base.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The agent computes five global percentile bins from all experiments’ running-speed samples gathered in the fast first pass, then digitizes each interpolated value into one of those five categories.

ii. 
```python
def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    if len(v2) == 0: return np.linspace(-1, 1, n+1)
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e
...
rbe = pct_bins(rs_all, 5)
...
rb = dig(ri, rbe)
```

iii. The stated reason, in `CONVERSION_NOTES.md` Step 5 and trajectory steps 68 and 84-92, was to satisfy the task’s request for five equal-percentile bins while doing it once globally for the full dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation to the common rebinned ophys time axis `bc`, then per-trial arrays are obtained by slicing the same time bins used for neural data.

ii. 
```python
ri = interpolate.interp1d(...)(bc)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. The justification is the same shared-time-base design discussed in trajectory steps 31, 44, and 58.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the output name `pupil_diameter`, the code actually derives it from `eye_tracking['pupil_area']` and `eye_tracking['timestamps']`, not from pupil width or a diameter estimate.

ii. 
```python
eye = ed['eye_tracking']
...
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `eye_tracking.pupil_area` to `output[3]: pupil_diameter`, and trajectory steps 88-91 show the agent chose the NWB `pupil_tracking['area']` field for the fast pass as well.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script linearly interpolates the `pupil_area` trace onto the 93 ms session bins and then digitizes the interpolated values into percentile bins. No conversion from area to diameter is performed.

ii. 
```python
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(
    ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan
)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
```

iii. Trajectory steps 88-91 show the reasoning: the agent initially tested other pupil quantities, then settled on the NWB/SDK area field because it matched the SDK values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The agent computes five global percentile bins from concatenated pupil-area samples collected in the fast first pass, and each interpolated trial-bin value is digitized into those five categories.

ii. 
```python
pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
pa_all.append(np.array(pa, dtype=np.float64))
...
pbe = pct_bins(pa_all, 5)
...
pb = dig(pi, pbe)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Interpolate + 5 percentile bins,” and trajectory steps 90-92 justify using the NWB `area` field so the percentile bins match SDK pupil values.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The chosen pupil signal is interpolated onto the same 93 ms session bins as neural data and then cut into trials using the same trial-time masks.

ii. 
```python
pi = interpolate.interp1d(...)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
```

iii. The rationale is the common shared-time-axis approach used for all time-varying outputs, as described in trajectory steps 31, 44, and 58.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean outcome columns in the SDK `trials` table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. 
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. `CONVERSION_NOTES.md` Step 5 maps “trials outcome” to the decoder output, and trajectory steps 11 and 31 identify those four trial outcome classes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. For each retained trial, the code reduces the four boolean outcome flags to one integer category. It then repeats that category across every time bin in the trial, so the stored output is a constant time series rather than a 1D static label.

ii. 
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
np.full(len(ti), oc, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 calls this output “Per-trial categorical,” while trajectory step 59 shows the agent noticed the task wanted a static per-trial value but chose to broadcast it across time to fit the decoder/output stacking scheme.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing running or pupil samples are linearly interpolated when at least two valid points exist; otherwise the whole rebinned trace is `NaN`. `NaN` values are then forced into the middle category during digitization. Missing pupil streams in the fast pass are silently skipped. Empty neural bins copy the previous bin value, and too-short trials or experiments are dropped.

ii. 
```python
if len(v2) == 0: return np.linspace(-1, 1, n+1)
...
b[np.isnan(v)] = (len(e) - 1) // 2
...
ri = ... if vm.sum()>=2 else np.full(nb, np.nan)
pi = ... if vm2.sum()>=2 else np.full(nb, np.nan)
...
except:
    pass
...
elif b > 0: out[:, b] = out[:, b-1]
...
if len(ti) < 2: continue
```

iii. The agent’s justifications are mostly pragmatic rather than scientific: `CONVERSION_NOTES.md` Step 10 mentions fixing dtype and pupil extraction issues, and trajectory steps 88-92 emphasize robustness and runtime over explicit missing-data modeling.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading each NWB file through the AllenSDK in `load_experiment()`. The first pass was optimized because SDK-based stats collection was too slow, leaving per-experiment SDK loading as the main bottleneck.

ii. 
```python
def load_experiment(eid):
    ...
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
```

iii. `CONVERSION_NOTES.md` Steps 6, 7, and 9 quantify this directly, and trajectory steps 84-99 repeatedly describe SDK loading as the runtime bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization targets are the Python loop over rebinned time bins in `resample_session()`, the loop over change times when writing `cb`, and the row-wise `iterrows()` loop over trials.

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
```

iii. The agent did not explicitly call these loops out in the notes, but they are the remaining non-vectorized hot paths after the `h5py` optimization described in Step 6 and trajectory steps 87-92.

## 9-c. What processing does the code repeat multiple times?

i. It makes two dataset-wide passes: `fast_collect_stats()` over all experiments and then full processing of all experiments. Within each session it also computes full-session outputs and only later slices them into trials.

ii. 
```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
```

iii. Trajectory steps 84-87 revolve around the agent recognizing and trying to reduce repeated whole-dataset processing; the final code keeps the repeated pass but makes the first one fast with `h5py`.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are discarded: full-session rebinned neural/output arrays include inter-trial bins that are later thrown away; `metadata` is loaded into `ed` but never used in `process_experiment()`; and the full running/pupil sample arrays are only used to compute percentile edges and are then deleted.

ii. 
```python
return {
    ...
    'metadata': dict(ds.metadata),
}
...
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
del rs_all, pa_all
```

iii. `CONVERSION_NOTES.md` Step 6 frames the code as “session-wide resampling then trial extraction,” which is simpler to write but inevitably creates intermediate data that downstream decoding never uses.
