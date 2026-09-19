# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a metadata CSV (`ophys_experiment_table.csv`) from the local data directory and cross-references it with available NWB files on disk. It filters to active (non-passive) experiments. Each experiment is loaded individually from its NWB file using `BehaviorOphysExperiment.from_nwb()`. A fast pre-pass using `h5py` collects global statistics (image names, running/pupil distributions) before the full SDK load.

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

iii. The AI chose to load data directly from NWB files rather than through the SDK cache (`VisualBehaviorOphysProjectCache.from_s3_cache`). It filters experiments by `passive==False` to select active behavior sessions. The fast h5py pre-pass was used for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values across all selected experiments, stored as sorted strings.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
```

iii. Each experiment row has a `mouse_id`. The AI collects unique values to form the subject list.

## 1-c. How are the data split into sessions?

i. Each experiment (imaging plane) is treated as a separate "session" in the output. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session. Each of the 202 experiments becomes one entry in the output lists.

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
    asi.append(s2i[str(row['mouse_id'])])
    abri.append(np.full(nn, r2i[row['targeted_structure']], dtype=np.int64))
```

iii. The AI iterates over the experiment list directly. Each experiment corresponds to a single imaging plane. The CONVERSION_NOTES.md states "202 active experiments" and the output has 202 sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `trials` table from the SDK. Each non-aborted, non-auto-rewarded trial that is either a Go or Catch trial is included. The trial window spans from `start_time` to `stop_time`, resampled into 93ms bins. Trials with fewer than 2 time bins after resampling are excluded.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
```

iii. The filtering criteria match the instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded. The AI uses the SDK trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch, (2) not aborted, (3) not auto-rewarded, (4) must have at least 2 time bins after resampling. Sessions with fewer than 2 valid trials are skipped.

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

iii. The filtering is consistent with the instructions. The AI does not explicitly check for valid `change_time`, but instead relies on the Go/Catch filter which implicitly ensures a valid change event context.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `events` (deconvolved calcium events), NOT `dff_traces` (dF/F). The AI accesses `ds.events.events.values` from each experiment.

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

iii. The CONVERSION_NOTES.md states: "Paper uses 'events' (deconvolved calcium events) for neural analysis." The AI chose events because the reference paper analyzes deconvolved calcium events.

## 2-b. How is the `neural` data processed?

i. The neural events data is resampled from the native ophys frame rate to fixed 93ms time bins. The resampling computes the mean of all ophys frames falling within each bin's window (bin center +/- half bin width). If no frames fall in a bin, the previous bin's value is carried forward.

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

iii. The 93ms bin size was chosen to harmonize data from different frame rates (MESO at ~11 Hz / ~91ms and CAM2P at ~31 Hz / ~32ms) into a uniform temporal resolution. The resampling uses a simple averaging approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to the neural data. All neurons present in the NWB files (which have already passed the Allen SDK's valid_roi filtering) are included.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES.md states: "ROI filtering already applied (valid_roi=True)" and "Cell filtering (valid_roi) already applied in NWB files."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time (`start_time` from the trials table). Time bins are created spanning from the earliest trial start to the latest trial stop across the session. Each trial extracts the bins falling within its `[start_time, stop_time)` window.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    ...
    ntl.append(nr[:, ti])
```

iii. The AI creates a session-wide set of time bins and then selects the subset that falls within each trial's window. This means alignment is to trial start (the first bin center at or after `start_time`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning to a fixed 93ms bin size (`TARGET_BIN_SIZE = 0.093`). This differs from the native frame rates of the data (~91ms for MESO, ~32ms for CAM2P).

ii.
```python
TARGET_BIN_SIZE = 0.093
...
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
nr = resample_session(ev, ots, bc)
```

iii. The AI's CONVERSION_NOTES state: "Resample all to ~93ms bins" to harmonize different frame rates. The metadata reports `time_bin_size: 93.0` ms.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, specifically the `image_name` column from rows in the `change_detection` stimulus block that are not omitted.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
```

iii. The AI uses the stimulus presentations table to determine which image is on screen at each time point, filtering out omitted stimuli. This is a time-varying approach based on stimulus onset times.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping. For each time bin, the most recent non-omitted stimulus presentation determines the image identity. This is computed using `np.searchsorted` on stimulus start times.

ii.
```python
n2i = {n: i for i, n in enumerate(imn)}
...
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. By using `searchsorted` with `side='right'` minus 1, the AI finds the most recent stimulus presentation for each time bin. This gives a time-varying image identity that changes with each new stimulus presentation.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same session-wide time bin centers (`bc`) as the neural data, then sliced per trial using the same indices. Both use the same resampled time base.

ii.
```python
# Both computed on bc (bin centers):
nr = resample_session(ev, ots, bc)
sidx = np.searchsorted(ist, bc, side='right') - 1
...
# Same trial indices:
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], ...]))
```

iii. Alignment is guaranteed because both neural and output variables are indexed on the same time bin array.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations` where `is_change==True`. These are the stimulus presentations marked as image changes in the stimulus table.

ii.
```python
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. The AI uses the `is_change` flag from stimulus_presentations rather than the `go` column from the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change event, time bins within a 750ms window starting at the change time are marked as 1. All other bins are 0.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The 750ms window corresponds to one image presentation (250ms) plus the inter-stimulus interval (500ms).

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary variable (0 or 1), not thresholded from a continuous value. 0 = no change, 1 = change occurring.

ii. See 4-b.

iii. Binary by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the same session-wide bin centers, then sliced per trial.

ii. See 3-c.

iii. Same time base alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed` (speed and timestamps).

ii.
```python
run = ed['running_speed']
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. Standard SDK interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the session's time bin centers (93ms bins), excluding NaN values before interpolation. Then discretized into 5 percentile-based bins using globally computed bin edges. Bin edges are set to [-inf, ..., inf] at the extremes. NaN values after interpolation are mapped to the middle bin.

ii.
```python
vm = ~np.isnan(rsp)
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
rb = dig(ri, rbe)
...
def pct_bins(v, n=5):
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
```

iii. Global bin edges computed from subsampled (every 10th sample) running speed data across all experiments via h5py pre-pass.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 equal percentile bins. Bin edges are computed globally using `np.percentile` with linspace(0, 100, 6), then outer edges are set to +/- infinity. NaN values are mapped to the middle bin (bin 2).

ii. See 5-b code.

iii. Percentile-based binning ensures roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same session-wide bin centers as the neural data, then sliced per trial using the same indices.

ii.
```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
```

iii. Same time base as neural data ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area` (pupil area, NOT width or diameter).

ii.
```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. The AI uses `pupil_area` from the eye tracking data. The CONVERSION_NOTES confirm this: "Fixed pupil area collection in h5py (use area key, not width*height)."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area values with NaN are excluded before interpolation. The remaining values are linearly interpolated to session bin centers. Then discretized into 5 percentile bins with globally computed edges. NaN values are mapped to the middle bin.

ii.
```python
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc)
pb = dig(pi, pbe)
```

iii. NaN removal before interpolation prevents NaN contamination. Note that unlike the reference, the AI does NOT specifically filter out blink frames using `likely_blink` - it only removes NaN values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins with global edges, NaN mapped to middle bin.

ii. See 5-b code for `pct_bins` and `dig` functions.

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed - interpolated to session bin centers, then sliced per trial.

ii. See 6-b code.

iii. Same time base alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` (implicitly, as the else case) in the trials table.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. Same four outcome categories as the reference, derived from the SDK's canonical trial outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) using a conditional chain. The code is then broadcast as a constant across all time bins in the trial.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]))
```

iii. The mapping order matches `output_values[4] = ['hit', 'miss', 'false_alarm', 'correct_reject']`. Static per-trial as required by the instructions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If `load_experiment` or `process_experiment` fails, the experiment is skipped (though no explicit try/except is shown in `process_experiment`; it returns empty lists for no valid trials).
- **Short trials**: Trials with fewer than 2 time bins are skipped.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **Missing behavioral data**: NaN values in running speed and pupil area are filtered before interpolation, and any remaining NaN after interpolation are mapped to the middle bin.
- **Missing eye tracking**: If fewer than 2 valid pupil data points exist, the entire session gets NaN pupil values (mapped to middle bin).

ii.
```python
if len(vt) == 0: return [], [], nn
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    ...
    skipped += 1; continue
...
b[np.isnan(v)] = (len(e) - 1) // 2
...
pi = interpolate.interp1d(...)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
```

iii. The AI handles edge cases defensively with graceful fallbacks.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `load_experiment()` using the SDK's `BehaviorOphysExperiment.from_nwb()`, which takes 3-25 seconds per experiment depending on neuron count. The CONVERSION_NOTES estimate 18 minutes for loading 202 experiments.

ii. N/A

iii. Per the conversion output, load times range from 3.1s to 24.6s per experiment, with processing typically 0.3-1.3s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_session` function has a Python loop over all time bins (`for b in range(nb)`) that could be vectorized. Also, the image change computation loops over change times.

ii.
```python
def resample_session(events, ophys_ts, bc):
    ...
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
    return out
...
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The bin loop iterates over typically ~8000-9000 bins per session, which is a significant inner loop. The change loop is much smaller (one iteration per change event).

## 9-c. What processing does the code repeat multiple times?

i. The AI performs a two-pass approach: first a fast h5py pass to collect global statistics (image names, running/pupil distributions for bin edges), then a full SDK load for processing. The running speed and pupil data are effectively read twice - once via h5py (subsampled) and once via the SDK.

ii.
```python
# Pass 1: h5py
imn, rs_all, pa_all = fast_collect_stats(el)
rbe = pct_bins(rs_all, 5)
pbe = pct_bins(pa_all, 5)

# Pass 2: SDK
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
```

iii. The two-pass approach was an optimization tradeoff: the h5py pass is much faster (~0.1s/experiment) than the SDK load (~5s/experiment), so collecting statistics first avoids needing to store all data in memory for a second pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `stimulus_presentations` and `metadata` from each experiment during `load_experiment()`, though `metadata` is not used in processing. The fast stats pass also reads running/pupil data that gets re-read in the main pass. Additionally, the global resampling creates neural and behavioral arrays spanning the full valid trial time range (min start to max stop), including inter-trial intervals, though only the within-trial portions are used.

ii.
```python
'stimulus_presentations': ds.stimulus_presentations.copy(),
'metadata': dict(ds.metadata),
...
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
nr = resample_session(ev, ots, bc)  # includes inter-trial gaps
```

iii. The inter-trial data in the resampled arrays is computed but never used, as only trial-specific bins are extracted.
