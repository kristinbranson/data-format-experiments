# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK project cache to discover sessions. It reads the local `ophys_experiment_table.csv`, intersects that table with NWB files present on disk, filters to `passive == False`, then loads each experiment directly from its NWB file with `BehaviorOphysExperiment.from_nwb`. It also performs a first pass over every NWB file with `h5py` to collect image names and running/pupil statistics before the main processing pass.

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
    path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

iii. `CONVERSION_NOTES.md` says the agent intentionally used local NWB files and AllenSDK loaders, and later added an `h5py` first pass as a speed optimization. The trajectory explicitly says the goal was to avoid the SDK overhead by doing “fast stats collection” with `h5py`.

## 1-b. How are the data split into subjects?

i. Subjects are defined as the unique `mouse_id` values found in the filtered experiment list.

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
```

iii. The agent's notes repeatedly summarize the dataset in terms of unique mice, and the trajectory treats `mouse_id` as the subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one session. It iterates row-by-row through the filtered experiment table and appends one session entry for each experiment, rather than grouping experiments that share the same `ophys_session_id`.

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

iii. In the trajectory the agent explicitly reasoned that each experiment should be treated as a separate session because each experiment has its own neurons, even though it had already observed that some `ophys_session_id` values contain multiple planes.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. The AI keeps `go` or `catch` trials, excludes aborted and auto-rewarded trials, builds a session-wide uniform time grid `bc` spanning from the minimum valid `start_time` to the maximum valid `stop_time`, resamples the whole session to that grid, and then defines each trial by masking `bc` to `start_time <= bc < stop_time`.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
```

```python
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
    ntl.append(nr[:, ti])
```

iii. The notes and trajectory say the agent wanted a common bin size across experiments, so it moved from native ophys frames to 93 ms bins and segmented trials after that resampling.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `go` or `catch`, with `aborted` and `auto_rewarded` removed. Trials with fewer than 2 resampled bins are skipped. Experiments with fewer than 2 remaining trials are skipped. The code does not explicitly require non-null `change_time`.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
if len(vt) == 0: return [], [], nn
...
if len(ti) < 2: continue
...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
```

iii. `CONVERSION_NOTES.md` lists “Include Go + Catch, exclude Aborted + Auto-rewarded” as the trial curation rule. The trajectory also reports these as the valid trial types the agent decided to keep.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the AllenSDK `events` table, specifically `ds.events.events`, not from `dff_traces`.

ii.
```python
return {
    'ophys_timestamps': ds.ophys_timestamps.copy(),
    'events': np.vstack(ds.events.events.values).astype(np.float32),
    ...
}
```

iii. The notes say “Paper uses ‘events’ (deconvolved calcium events) for neural analysis,” and later list `events -> neural` in the variable-mapping table. The trajectory also states that the paper used discrete calcium events and that the script should match that.

## 2-b. How is the `neural` data processed?

i. The AI resamples the full-session event matrix to fixed 93 ms bins. For each bin it averages the event values whose native ophys timestamps fall within a half-bin window around the bin center. If a bin has no source frames, it copies the previous bin forward.

ii.
```python
TARGET_BIN_SIZE = 0.093

def resample_session(events, ophys_ts, bc):
    h = TARGET_BIN_SIZE / 2
    ...
    li = np.searchsorted(ophys_ts, bc - h, side='left')
    ri = np.searchsorted(ophys_ts, bc + h, side='left')
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
```

iii. `CONVERSION_NOTES.md` says the agent found both 11 Hz and 31 Hz acquisitions and decided to “Resample all to ~93ms bins.” The trajectory explicitly frames this as a common-time-bin design choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit cell-level filtering in `convert_data.py`. The code loads all rows in `ds.events.events` and keeps them all. The implied QC decision is to trust the NWB/AllenSDK preprocessing and ROI filtering already present upstream.

ii.
```python
'events': np.vstack(ds.events.events.values).astype(np.float32),
```

iii. `CONVERSION_NOTES.md` states “Cell filtering (valid_roi) already applied in NWB files” and “ROI filtering already applied.” The trajectory shows the agent investigated `valid_roi` and concluded no additional filtering was needed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. After full-session resampling, the AI aligns trial neural data to trial start by selecting the resampled bins whose centers fall within each trial's `[start_time, stop_time)` interval. In the saved data, each trial starts at its own first selected bin; the code does not keep a separate relative-time axis.

ii.
```python
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
ntl.append(nr[:, ti])
```

```python
'temporal_alignment_event': 'Trial start time',
'off_start': 0.0, 'off_end': None,
```

iii. The notes' metadata says the alignment event is “Trial start time.” The trajectory also discusses segmenting trials by `start_time` and `stop_time` after creating a common bin grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 93 ms bins (`0.093` s). Yes, the code explicitly rebins the native ophys data to that common resolution.

ii.
```python
TARGET_BIN_SIZE = 0.093
...
'time_bin_size': TARGET_BIN_SIZE * 1000,
'bin_size_seconds': TARGET_BIN_SIZE,
```

iii. `CONVERSION_NOTES.md` repeatedly highlights a 93 ms common time bin and explains it as the compromise between 11 Hz MESO data and faster CAM2P data.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, using `stimulus_block_name`, `image_name`, `omitted`, and `start_time`. The code restricts to `change_detection` stimulus blocks and removes omitted-image presentations.

ii.
```python
sp = ed['stimulus_presentations']
...
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
```

iii. The mapping table in `CONVERSION_NOTES.md` says `stimulus_presentations.image_name -> output[0]: image_identity`. The trajectory shows the agent inspected `stimulus_presentations` and chose it as the source for image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first builds a global image-name vocabulary across all experiments in `fast_collect_stats`. During processing, each non-omitted image name is mapped to an integer code. Each session-wide bin gets the code of the most recent non-omitted image whose `start_time` is at or before that bin center.

ii.
```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
n2i = {n: i for i, n in enumerate(imn)}
...
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. The notes emphasize a “fast stats collection” pass to get all image names up front. The trajectory says this first pass was needed so image identity could use a single global mapping.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same session-wide 93 ms bin centers `bc` used for neural data, then each trial takes the subset of image-identity bins indexed by `ti`.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The agent's general alignment rule in the notes is to resample everything to the common ophys-derived bin grid first, then slice trials from that shared grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and `stimulus_presentations.start_time`, again within the `change_detection` block.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. `CONVERSION_NOTES.md` maps `stimulus_presentations.is_change` to the image-change output, and the trajectory discusses using the 750 ms image presentation interval from the paper to define the change period.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector over the session-wide bins, then for every change stimulus sets bins from `ct` to `ct + 0.750` seconds to 1.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The notes cite the paper's 250 ms stimulus plus 500 ms gray interval and say “750ms image presentation intervals.” The trajectory uses that timing as the rationale for a 750 ms change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: `0` for `no_change`, `1` for `change`.

ii.
```python
ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
      [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
```

iii. The target task asked for a binary image-change output, and the agent encoded it directly as a two-level categorical variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is defined on the same session-wide resampled grid `bc` as the neural data and then sliced into per-trial segments with the same trial index array `ti`.

ii.
```python
cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The notes describe a shared binning/alignment strategy for all outputs after resampling the session.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `running_speed.timestamps` and `running_speed.speed`.

ii.
```python
run = ed['running_speed']
...
rts, rsp = run['timestamps'].values, run['speed'].values
```

iii. Both the notes and the trajectory identify `running_speed` as the source stream for locomotion.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI does two processing stages. First, `fast_collect_stats` reads running speed directly from NWB with `h5py` and subsamples every tenth point to estimate global percentile bin edges. Second, for each experiment it linearly interpolates running speed onto the 93 ms session-wide bin centers and digitizes the interpolated values with those global edges.

ii.
```python
rs = f['processing']['running']['speed']['data'][::10]
...
rbe = pct_bins(rs_all, 5)
```

```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
rb = dig(ri, rbe)
```

iii. `CONVERSION_NOTES.md` describes the `h5py` pass as a major optimization and says running speed should be “Interpolate + 5 percentile bins.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into 5 percentile bins using global edges. Missing values are assigned to the middle bin, not to a special missing category.

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
```

iii. The notes specify five percentile bins for running speed. The code itself shows the specific missing-data choice: NaNs go to the central bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated onto the same session-wide 93 ms bin centers `bc` as the neural data, then trial segments are cut out with the same `ti` index arrays.

ii.
```python
ri = interpolate.interp1d(...)(bc)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The notes state that all streams are aligned after resampling to the common bin grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives this output from `eye_tracking.pupil_area`, with timestamps from `eye_tracking.timestamps`. It does not use `pupil_width`.

ii.
```python
eye = ed['eye_tracking']
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
```

iii. `CONVERSION_NOTES.md` explicitly maps `eye_tracking.pupil_area` to the pupil output. The trajectory shows the agent inspected the NWB `area` field and decided it matched the SDK's `pupil_area`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. As with running speed, the code first collects a global sample of pupil-area values with `h5py` and uses them to compute 5 percentile bin edges. Then each experiment's `pupil_area` signal is linearly interpolated onto the 93 ms bin centers and digitized. The code does not remove `likely_blink` frames before interpolation.

ii.
```python
pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
...
pbe = pct_bins(pa_all, 5)
```

```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
```

iii. The notes say the agent switched to the NWB `area` field after checking it against the SDK output. There is no mention of blink removal in the notes; the trajectory instead focuses on matching the SDK's `pupil_area`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil area is thresholded into 5 global percentile bins, with NaNs mapped to the middle bin.

ii.
```python
pb = dig(pi, pbe)
...
b[np.isnan(v)] = (len(e) - 1) // 2
```

iii. The notes specify five percentile bins for pupil. The exact NaN handling comes from the `dig` helper.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated onto the same resampled bin centers `bc` used for neural data, then sliced into per-trial segments with the same trial indices.

ii.
```python
pi = interpolate.interp1d(...)(bc)
...
ti = np.where(tm)[0]
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The agent uses the same shared-grid alignment rule for pupil as for running speed and image variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean fields `hit`, `miss`, and `false_alarm`, with all other retained trials implicitly mapped to the fourth category (`correct_reject`).

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. The notes list the four standard trial outcomes and the trajectory identifies hit/miss/false_alarm/correct_reject as the trial outcome categories of interest.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the categorical outcome to an integer code and repeats that code across every time bin in the trial, making the output time-shaped even though it is conceptually static per trial.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. The notes list `trial_outcome` as a categorical decoder output, and the trajectory shows the agent adapting outputs to the decoder's expected time-varying matrix format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing data mainly through permissive defaults. Missing running or pupil values after interpolation are converted to NaNs and then assigned to the middle discrete bin. Trials with fewer than 2 bins are skipped, and experiments with fewer than 2 retained trials are dropped. In `fast_collect_stats`, missing pupil data are silently ignored by `try/except`. There is no outer `try/except` around experiment loading in the final version.

ii.
```python
ri = ... if vm.sum()>=2 else np.full(nb, np.nan)
...
pi = ... if vm2.sum()>=2 else np.full(nb, np.nan)
...
b[np.isnan(v)] = (len(e) - 1) // 2
```

```python
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

iii. The notes mention fixing pupil-area extraction and describe the dataset as having sparse events and occasional missing values. The code shows the final concrete policy: keep processing, fill NaNs into a default bin, and silently skip unusable short trials or experiments.

## 9-a. What are the most time-consuming steps of the code?

i. The main bottleneck is loading each experiment from NWB through `pynwb` and `BehaviorOphysExperiment.from_nwb`. The AI therefore added a separate fast `h5py` pass for image/running/pupil statistics to avoid paying the full SDK cost twice.

ii.
```python
with pynwb.NWBHDF5IO(path, 'r') as io:
    nwb = io.read()
    ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
```

```python
def fast_collect_stats(exp_list):
    """Use h5py for fast collection of image names and running/pupil stats."""
```

iii. `CONVERSION_NOTES.md` gives runtime estimates of roughly 5.5 s average load time per experiment and explicitly calls out `h5py for fast stats collection` as the key optimization.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code contains several Python loops that could have been reduced or vectorized: the loop over bins in `resample_session`, the loop over change times when painting the 750 ms image-change windows, the per-trial `iterrows()` loop, and the pass over all experiments in `fast_collect_stats`.

ii.
```python
for b in range(nb):
    if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
    elif b > 0: out[:, b] = out[:, b-1]
```

```python
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
```

iii. The notes say the agent prioritized speed, but the final design still keeps several Python-level loops because it first constructs session-wide arrays and then slices them trial by trial.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats dataset traversal in two passes. `fast_collect_stats` opens every NWB file once to gather image names and percentile statistics, and then the main loop opens every experiment again to load the full SDK object and process trials. It also computes session-wide outputs over the entire valid-trial span before re-slicing them into trials.

ii.
```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ed = load_experiment(eid)
```

iii. The trajectory explicitly acknowledges the two-pass structure and justifies it as the cost of getting a global image vocabulary and global percentile bins before final assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest unnecessary work is full-session processing that is later cut back to trial slices: the code resamples the entire session span between the first valid trial start and the last valid trial stop, computes full-session image identity/change/running/pupil arrays on that span, and then discards any bins not belonging to a retained trial. The preliminary `fast_collect_stats` pass also processes image/running/pupil data solely to support later binning/mapping and then discards those raw arrays.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
nr = resample_session(ev, ots, bc)
...
ib = np.zeros(nb, dtype=np.int64)
cb = np.zeros(nb, dtype=np.int64)
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
```

```python
imn, rs_all, pa_all = fast_collect_stats(el)
...
del rs_all, pa_all
```

iii. The notes frame these as performance-oriented design choices, but they still create intermediate session-wide results that are not directly preserved in the final per-trial dataset.
