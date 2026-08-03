# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by first reading a CSV metadata table (`ophys_experiment_table.csv`) and cross-referencing it with available NWB files on disk. It filters to non-passive (`passive==False`) experiments that have corresponding NWB files. Each experiment is loaded individually via `BehaviorOphysExperiment.from_nwb()` using the AllenSDK. Each experiment corresponds to one imaging plane (one session = one experiment in this code, unlike the reference which groups multiple planes per session).

ii.
```python
def get_experiment_list(sample=False):
    et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0])
                  for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
    m = et[et['ophys_experiment_id'].isin(nwb_ids)]
    a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
    ...

def load_experiment(eid):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
    path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

iii. The AI chose to read from local NWB files directly using `pynwb` and the AllenSDK's `from_nwb()` method, rather than using the S3 cache interface. It filters by `passive==False` to get active behavior experiments. The AI notes in CONVERSION_NOTES.md that 202 active experiments were found across 38 mice.

## 1-b. How are the data split into subjects?

i. Subjects are determined from unique `mouse_id` values in the experiment table. Each experiment is processed independently, and the subject index is assigned per-experiment (not per-session).

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
...
asi.append(s2i[str(row['mouse_id'])])
```

iii. The AI sorts unique mouse IDs to create a deterministic subject list. Each processed experiment gets a subject index assigned from this mapping.

## 1-c. How are the data split into sessions?

i. The AI treats each experiment (each NWB file / imaging plane) as a separate "session" in the output data structure. It does NOT group multiple imaging planes from the same ophys session together. Each experiment is processed independently and appears as its own session in the output.

ii.
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    ed = load_experiment(eid)
    ...
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
```

iii. The AI iterates over each experiment in the experiment table and treats each as an independent session. This contrasts with the reference which groups experiments by `ophys_session_id` to combine neurons from multiple imaging planes into a single session.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `trials` table from the SDK. Go and catch trials are included; aborted and auto-rewarded trials are excluded. The trial window spans from `start_time` to `stop_time`. Neural data is resampled to a uniform 93ms time grid spanning the valid trial range, then each trial's bins are extracted.

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

iii. The AI uses the SDK's trials table to define trial boundaries and filters Go+Catch trials while excluding Aborted and Auto-rewarded. A session-wide time grid is created first, then trial-specific bins are extracted from it. Trials with fewer than 2 time bins are skipped.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) only Go and Catch trials are included, (2) aborted and auto-rewarded trials are excluded, (3) trials with fewer than 2 time bins are skipped, (4) experiments/sessions with fewer than 2 valid trials are skipped.

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

iii. The filtering matches the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded. The AI does not explicitly filter on `change_time` validity (unlike the reference), but the Go/Catch filter effectively ensures valid change events.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `events` (deconvolved calcium events), NOT from `dff_traces` (dF/F). The AI accesses `ds.events.events` from the AllenSDK experiment object.

ii.
```python
def load_experiment(eid):
    ...
    ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
    return {
        ...
        'events': np.vstack(ds.events.events.values).astype(np.float32),
        ...
    }
```

iii. The AI chose `events` based on the reference paper's methodology, which uses deconvolved calcium events rather than raw dF/F traces. The CONVERSION_NOTES.md states: "Paper uses 'events' (deconvolved calcium events) for neural analysis."

## 2-b. How is the `neural` data processed?

i. Neural data (events) is resampled from the native ophys frame rate to a uniform 93ms time grid. The resampling averages all ophys frames within each 93ms bin (using a window of +/- half the bin size around each bin center). If no frames fall in a bin, the previous bin's value is carried forward.

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
```

iii. The 93ms bin size was chosen to approximate the ~11 Hz MESO frame rate (1/11 Hz ~ 91ms). The AI applies this uniform binning to all experiments, including those recorded at ~31 Hz (CAM2P rigs), effectively downsampling the higher-rate recordings. This ensures uniform temporal resolution across all sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All neurons present in the SDK's `events` data (which already has ROI filtering via `valid_roi`) are included.

ii. N/A - no filtering code is present.

iii. The AI notes in CONVERSION_NOTES.md: "Cell filtering (valid_roi) already applied in NWB files" and "ROI filtering uses multi-label classifier."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. A session-wide time grid of 93ms bins is created starting from the earliest trial start time. Each trial extracts the bins falling within its `start_time` to `stop_time` window.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
nr = resample_session(ev, ots, bc)
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    ...
    ntl.append(nr[:, ti])
```

iii. By creating a session-wide grid and then slicing per trial, all trials share the same temporal grid, avoiding per-trial interpolation artifacts. The alignment is to the trial start, consistent with the instructions specifying alignment to ophys timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 93ms (TARGET_BIN_SIZE = 0.093 seconds). Temporal rebinning IS applied: the native ophys frames (at ~11 Hz for MESO or ~31 Hz for CAM2P) are resampled to the uniform 93ms grid by averaging frames within each bin.

ii.
```python
TARGET_BIN_SIZE = 0.093
...
'time_bin_size': TARGET_BIN_SIZE * 1000,  # 93.0 ms
```

iii. The AI chose 93ms to approximate the MESO frame rate, applying it uniformly. The CONVERSION_NOTES.md notes: "Resample all to ~93ms bins." The metadata stores the time bin size as 93.0ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column. The AI filters to change_detection stimulus blocks and excludes omitted stimuli.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
```

iii. The AI uses the `stimulus_presentations` table rather than the `trials` table for image identity, allowing frame-by-frame image identity based on stimulus onset times. This is a more granular approach than using initial/change image from the trials table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin, the most recent non-omitted stimulus presentation is identified using `np.searchsorted`. The image name is mapped to an integer code via a global mapping built from all unique image names across all sessions.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. Using `searchsorted` with `side='right'` minus 1 finds the last stimulus that started before each time bin center. This naturally handles the time-varying nature of image identity as new stimuli are presented throughout the trial.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 93ms time grid as the neural data (`bc` array), so alignment is inherent. The stimulus presentation start times are used to determine which image is on screen at each bin center.

ii.
```python
# Same bc (bin centers) used for both neural and image identity
sidx = np.searchsorted(ist, bc, side='right') - 1
```

iii. Both neural and output data use the same time grid, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically the `is_change` column. Stimuli flagged as `is_change==True` in the change_detection block are used.

ii.
```python
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The AI uses `is_change` from stimulus presentations rather than the `go` column from the trials table. This marks all change events detected in the stimulus stream, including changes in both go and catch contexts.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created where bins within 750ms after each change stimulus onset are set to 1. All other bins are 0.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The 750ms window covers one stimulus presentation (250ms) plus the inter-stimulus interval (500ms). This marks the transient change event rather than the entire post-change period.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1). No thresholding is needed - it is directly computed as 0 (no change) or 1 (change present within 750ms window).

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The binary nature of the variable means no discretization or thresholding is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the same 93ms time grid (`bc` array).

ii. See 4-b code - uses the same `bc` array as neural data.

iii. Alignment is inherent through shared time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically the `speed` and `timestamps` columns.

ii.
```python
run = ed['running_speed']
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. The AllenSDK's `running_speed` attribute provides pre-processed running wheel speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is (1) NaN-filtered, (2) linearly interpolated to the 93ms time grid, and (3) discretized into 5 percentile-based bins. Bin edges are computed globally across ALL experiments using subsampled data (every 10th point) collected via h5py in a fast pre-pass. Bin edges are set to -inf and +inf at the extremes.

ii.
```python
# Interpolation
vm = ~np.isnan(rsp)
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
rb = dig(ri, rbe)

# Bin edge computation (fast pre-pass)
def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

# Discretization
def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2  # NaN -> middle bin
    return b.astype(np.int64)
```

iii. Global percentile bins ensure balanced classes across the dataset. The pre-pass using h5py is a speed optimization. Setting extremes to +/-inf ensures all values are captured. NaN values are mapped to the middle bin (bin 2).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using global bin edges. NaN values are mapped to the middle bin (index 2).

ii.
```python
def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
```

iii. The 5 percentile bins match the instructions. The bin edge computation uses `np.percentile` with edges at 0, 20, 40, 60, 80, 100 percentiles, ensuring roughly equal counts per bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 93ms time grid (`bc` array) as the neural data, then the same trial time indices are used to extract per-trial values.

ii.
```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
...
# Same ti indices used for both neural and running
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. Alignment is guaranteed by using the same time grid for all variables.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil data is derived from `dataset.eye_tracking`, specifically the `pupil_area` column (NOT `pupil_width`). For the fast pre-pass stats collection, the AI reads pupil area directly from the NWB file via h5py at `acquisition/EyeTracking/pupil_tracking/area`.

ii.
```python
# In process_experiment:
ets, pa = eye['timestamps'].values, eye['pupil_area'].values

# In fast_collect_stats (h5py pre-pass):
pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
```

iii. The AI uses `pupil_area` rather than `pupil_width` (which the reference uses). This is a notable difference - pupil area and pupil width are different measurements that scale differently (area ~ width^2).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is (1) NaN-filtered, (2) linearly interpolated to the 93ms time grid, and (3) discretized into 5 percentile-based bins. Bin edges are computed globally using subsampled data. Unlike the reference, the AI does NOT explicitly filter out blink frames using `likely_blink`.

ii.
```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
```

iii. The AI relies on NaN filtering to handle blinks (blink frames may have NaN pupil area), but does not explicitly use the `likely_blink` flag. The discretization approach is the same as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins with global bin edges. NaN values mapped to the middle bin.

ii.
```python
pb = dig(pi, pbe)
# dig() is the same function used for running speed
```

iii. Identical discretization scheme as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed - interpolated to the same 93ms time grid and extracted using the same trial time indices.

ii. See 5-d code snippets - same `bc` and `ti` arrays are used.

iii. Alignment is inherent through the shared time grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` (implicitly) in the trials table, checked in priority order.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. The AI uses a priority-based assignment: hit=0, miss=1, false_alarm=2, correct_reject=3 (default). This assumes the four outcomes are mutually exclusive for non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0-3) and broadcast as a constant value across all time bins within the trial. The output order is: hit(0), miss(1), false_alarm(2), correct_reject(3).

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]))
```

iii. The constant value per trial reflects that trial outcome is a per-trial variable. The coding scheme matches the reference's `TRIAL_OUTCOMES` list ordering.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **NaN in running/pupil data**: NaN values are filtered before interpolation. Any remaining NaNs after interpolation are mapped to the middle bin during discretization.
- **Missing pupil data**: If `pupil_area` has fewer than 2 valid points, the entire session's pupil data is set to NaN (then mapped to middle bin).
- **Short trials**: Trials with fewer than 2 time bins are skipped.
- **Empty sessions**: Experiments with fewer than 2 valid trials are skipped entirely.
- **h5py stats collection**: Missing pupil data in the NWB file is caught by try/except.

ii.
```python
# NaN handling in discretization
def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)

# Missing pupil in stats
try:
    pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
    pa_all.append(np.array(pa, dtype=np.float64))
except:
    pass
```

iii. The AI takes a pragmatic approach to missing data. NaN-to-middle-bin mapping is a reasonable default that avoids biasing toward extreme values. The try/except for pupil data handles sessions without eye tracking gracefully.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `load_experiment()` using `BehaviorOphysExperiment.from_nwb()`. From the conversion output, load times range from 3-25 seconds per experiment. Total processing for 202 experiments took the bulk of the runtime.

ii.
```python
ed = load_experiment(eid)
# Output shows: load=5.3s proc=0.3s per experiment
```

iii. The CONVERSION_NOTES.md estimates ~5.5s average load time per experiment vs ~0.5s for processing, making I/O the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_session` function contains a loop over all time bins that could potentially be vectorized. The image change computation also loops over change times.

ii.
```python
# Loop over bins in resample_session
for b in range(nb):
    if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
    elif b > 0: out[:, b] = out[:, b-1]

# Loop over change times
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The bin loop in `resample_session` handles variable-width bins (different numbers of ophys frames per bin), making vectorization non-trivial. The change time loop is small (few changes per trial) so optimization isn't critical.

## 9-c. What processing does the code repeat multiple times?

i. The AI performs a separate fast pre-pass using h5py to collect statistics (image names, running/pupil distributions) for bin edge computation, then loads and processes each experiment again with the full SDK. This means each NWB file is effectively accessed twice.

ii.
```python
# First pass: h5py stats
imn, rs_all, pa_all = fast_collect_stats(el)
rbe = pct_bins(rs_all, 5)
pbe = pct_bins(pa_all, 5)

# Second pass: full SDK loading
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
```

iii. The two-pass approach is a design choice: the fast h5py pass collects global statistics cheaply (~0.1s per file), avoiding a full SDK load for every file twice. The alternative would be a single pass that stores all data in memory, which the AI avoided for memory efficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores full session metadata, stimulus presentations, and trial tables during loading, but only a subset of this information is used in the final output. The `metadata` dict from `load_experiment` is never used.

ii.
```python
def load_experiment(eid):
    ...
    return {
        ...
        'stimulus_presentations': ds.stimulus_presentations.copy(),  # large table
        'metadata': dict(ds.metadata),  # unused
    }
```

iii. The stimulus_presentations table is used for image identity and change detection, but the full metadata dict is loaded and never referenced. This wastes memory but doesn't affect correctness.
