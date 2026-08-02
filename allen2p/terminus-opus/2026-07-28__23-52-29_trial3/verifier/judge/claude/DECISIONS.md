# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by first reading a CSV metadata table (`ophys_experiment_table.csv`) and matching experiment IDs against available NWB files on disk. It filters to only non-passive (active) experiments. Each experiment is loaded individually from NWB files using `BehaviorOphysExperiment.from_nwb()`. A fast pre-pass with `h5py` collects global statistics (image names, running speed, pupil area) before the main processing loop loads each experiment via the AllenSDK.

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

iii. The AI chose to load NWB files directly rather than using the SDK's S3 cache, since the data was already available locally. The two-pass approach (h5py for stats, then SDK for full loading) was an optimization to compute global bin edges without loading full SDK objects.

## 1-b. How are the data split into subjects?

i. Subjects are derived from unique `mouse_id` values in the filtered experiment table.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field uniquely identifies each animal across the dataset.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session. Each NWB file / experiment ID becomes one session in the output.

ii.
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    ed = load_experiment(eid)
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
```

iii. From CONVERSION_NOTES.md: "202 active experiments" are processed, each as a separate session. The AI notes that the paper used a subset (familiar sessions on MESO rig), but chose to include all active experiments.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `trials` table from the AllenSDK. Go and catch trials are included; aborted and auto-rewarded trials are excluded. Trials are extracted by finding time bins that fall within each trial's `start_time` to `stop_time` window.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
```

iii. Per the instructions, Go and Catch trials are included while Aborted and Auto-rewarded are excluded. The AI uses the resampled time bin centers (`bc`) to determine which bins fall within each trial window.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials with fewer than 2 time bins are skipped. Sessions (experiments) with fewer than 2 valid trials are skipped.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
    skipped += 1; continue
```

iii. The filtering matches the instructions to exclude aborted and auto-rewarded trials. The minimum trial count ensures degenerate sessions are excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events` — the deconvolved calcium events — rather than `dff_traces` (dF/F).

ii.
```python
def load_experiment(eid):
    ...
    return {
        ...
        'events': np.vstack(ds.events.events.values).astype(np.float32),
        ...
    }
```

iii. From CONVERSION_NOTES.md Step 1: "Paper uses 'events' (deconvolved calcium events) for neural analysis." The AI followed the paper's methodology which used events for decoding rather than raw dF/F traces.

## 2-b. How is the `neural` data processed?

i. Neural events are resampled to a fixed 93ms time bin by averaging events within each bin window. A session-wide set of evenly-spaced bin centers is computed from the first trial's start to the last trial's stop, and for each bin, events within a half-bin-width window are averaged.

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

iii. The 93ms bin size was chosen based on the MESO rig frame rate (~11 Hz). This rebins the data to a uniform time base across experiments with different frame rates (CAM2P at ~31 Hz vs MESO at ~11 Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neural data. All neurons present in the NWB files (which already have `valid_roi` filtering applied) are included.

ii. N/A — no filtering code.

iii. From CONVERSION_NOTES.md: "ROI filtering already applied (valid_roi=True)" in the NWB files. The AI relied on the SDK's pre-applied quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. A session-wide resampled array is computed first, then per-trial slices are extracted based on which bin centers fall within each trial's `start_time` to `stop_time` window. Alignment is to trial start time.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
nr = resample_session(ev, ots, bc)
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    ...
    ntl.append(nr[:, ti])
```

iii. The bin centers span from the first trial start to the last trial stop, and each trial extracts the bins that fall within its time window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to a fixed 93ms time bin size (TARGET_BIN_SIZE = 0.093 seconds). This is applied uniformly to all experiments regardless of their native frame rate.

ii.
```python
TARGET_BIN_SIZE = 0.093
...
'time_bin_size': TARGET_BIN_SIZE * 1000,  # 93.0 ms
```

iii. The 93ms bin size approximately matches the MESO rig's frame interval (~11 Hz). Rebinning normalizes the temporal resolution across CAM2P (~31 Hz) and MESO (~11 Hz) experiments.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column of non-omitted stimuli in the change detection block. The `start_time` of each stimulus presentation is used to determine which image is on screen at each time bin.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
sidx = np.searchsorted(ist, bc, side='right') - 1
```

iii. The stimulus_presentations table provides the exact timing of each stimulus flash, allowing time-varying image identity assignment at each bin center.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping. For each time bin, the most recent non-omitted stimulus presentation determines the image identity using `searchsorted`. Bins before the first stimulus get code 0.

ii.
```python
n2i = {n: i for i, n in enumerate(imn)}
...
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. A global mapping across all sessions ensures consistent integer codes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same resampled bin centers (`bc`) as the neural data, so alignment is automatic. For each bin center, the most recent stimulus presentation's image is assigned.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
```

iii. Using the same time base (bin centers) for both neural and output ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically using rows where `is_change == True` to identify change events.

ii.
```python
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The `is_change` column in the stimulus_presentations table marks which stimulus presentations are change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary indicator is computed: for each change event, time bins within a 750ms window after the change start_time are set to 1, otherwise 0. This is applied to ALL change events (both go and catch trials where is_change is True).

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The 750ms window covers one stimulus flash (250ms) plus the inter-stimulus interval (500ms).

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding — it is a binary variable (0 = no change, 1 = change).

ii. See 4-b above.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same bin centers as neural data, so alignment is automatic.

ii. Same bin centers `bc` as neural data.

iii. Same time base ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, specifically the `speed` and `timestamps` columns.

ii.
```python
run = ed['running_speed']
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. The SDK's `running_speed` attribute provides the standard locomotion measurement.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated (after removing NaN values) to the resampled bin centers, then discretized into 5 percentile-based bins. Bin edges are set to `-inf` and `inf` at the extremes. NaN values are mapped to the middle bin.

ii.
```python
vm = ~np.isnan(rsp)
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
rb = dig(ri, rbe)

def pct_bins(v, n=5):
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
```

iii. Percentile-based binning ensures roughly equal class counts. Setting edge bins to +/-inf ensures all values are captured. NaN mapped to middle bin rather than bin 0.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 equal percentile bins (0th, 20th, 40th, 60th, 80th, 100th percentiles). Edges are set to -inf/inf.

ii. See 5-b.

iii. Percentile-based binning ensures balanced classes.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same bin centers used for neural data, then extracted per-trial using the same time bin indices.

ii.
```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
```

iii. Same time base ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_area` column (NOT pupil_width).

ii.
```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. From CONVERSION_NOTES.md Step 10: "Fixed pupil area collection in h5py (use area key, not width*height)". The AI chose `pupil_area` as the measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is linearly interpolated (after removing NaN values) to the resampled bin centers, then discretized into 5 percentile-based bins. NaN values from interpolation are mapped to the middle bin.

ii.
```python
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc)
pb = dig(pi, pbe)
```

iii. Same approach as running speed. NaN values are removed before interpolation (rather than using `likely_blink` flag for blink filtering).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — 5 equal percentile bins with -inf/inf edges. NaN mapped to middle bin.

ii. See 6-b and `pct_bins`/`dig` functions.

iii. Same percentile-based approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated to the same bin centers used for neural data.

ii. Same as running speed alignment.

iii. Same time base ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table. A priority-based check maps each trial to one of these four categories.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. These four columns are the SDK's canonical trial outcome labels, mutually exclusive for valid (non-aborted, non-auto-rewarded) trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to an integer code (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast as a constant across all time bins in the trial.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]))
```

iii. The mapping order is fixed. The outcome is per-trial, so it is replicated across all time bins.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **NaN running speed**: NaN values removed before interpolation; remaining NaN after interpolation mapped to middle bin.
- **NaN pupil area**: Same NaN removal approach before interpolation.
- **Missing eye tracking data**: If no valid pupil data, a full-NaN array is used.
- **Few trials**: Sessions with < 2 valid trials are skipped.
- **Empty time bins in resampling**: Bins with no ophys frames copy the previous bin's value.

ii.
```python
# NaN handling in dig()
b[np.isnan(v)] = (len(e) - 1) // 2

# Empty bin in resampling
elif b > 0: out[:, b] = out[:, b-1]

# Few trials
if len(ntl) < 2: ... skipped += 1; continue
```

iii. NaN-to-middle-bin mapping avoids propagating missing data. Forward-fill for empty resampling bins prevents gaps.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment via the AllenSDK (`load_experiment()`) is the bottleneck, taking 3-25 seconds per experiment. The h5py-based stats collection is fast (~0.1s per experiment). Total processing time for 202 experiments was approximately 30 minutes.

ii. N/A — evident from timing output in `conversion_full_out.txt`.

iii. SDK loading involves reading NWB files and constructing Python objects, which is I/O and CPU bound.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_session` function iterates over time bins in a Python loop, which could be vectorized. The per-trial loop in `process_experiment` is sequential but contains mostly vectorized operations within each iteration.

ii.
```python
def resample_session(events, ophys_ts, bc):
    ...
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
    return out
```

iii. The resampling loop iterates over potentially thousands of time bins. Vectorization using segment-based approaches or numpy advanced indexing could improve performance.

## 9-c. What processing does the code repeat multiple times?

i. The code does a two-pass approach: first a fast h5py pass to collect global statistics (image names, running/pupil values for bin edges), then a full SDK pass to load and process each experiment. The h5py pass reads running speed and pupil data at subsampled resolution (every 10th sample), which is then re-read in full during the SDK pass. This is intentional duplication for efficiency.

ii.
```python
# First pass (h5py, subsampled)
rs = f['processing']['running']['speed']['data'][::10]
...
# Second pass (full SDK)
run = ed['running_speed']
```

iii. The two-pass approach trades a small amount of redundant reading for much faster global statistics computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes session-wide resampled neural and behavioral data for the entire period from first trial start to last trial stop. Time bins that fall between trials (inter-trial intervals) are computed but discarded when extracting per-trial data. Additionally, the running/pupil stats collected via h5py in the first pass cover the entire recording, not just trial periods.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
# All bins computed, but only trial-overlapping bins used
```

iii. Computing inter-trial bins is wasted work but simplifies the implementation. The global stats collection is approximate (subsampled) and covers all time, but bin edges are similar enough for percentile-based discretization.
