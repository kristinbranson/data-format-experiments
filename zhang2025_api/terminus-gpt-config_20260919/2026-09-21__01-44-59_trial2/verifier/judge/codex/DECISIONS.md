# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds a local ONE parquet index from a staged cache, identifies candidate sessions by checking whether required dataset filenames are present in the local index, and then restricts processing to a remapped subset of methods-paper EIDs when `repro_ephys_release.txt` is available. Session-level objects are loaded with `one.load_object(...)`, and spike sorting is loaded probe-by-probe with `SpikeSortingLoader`.

ii. 
```python
def get_one():
    if not (ses.exists() and dat.exists() and ses.stat().st_size > 100):
        make_parquet_db(CACHE_ROOT, out_dir=INDEX_DIR, hash_ids=True, hash_files=False)
    one = ONE(cache_dir=CACHE_ROOT, mode='local')
    one.load_cache(tables_dir=INDEX_DIR)
    return one
```

```python
def candidate_eids(one):
    core = (dataset_eids(one, '_ibl_trials.table.pqt') &
            dataset_eids(one, '_ibl_wheel.timestamps.npy') &
            dataset_eids(one, '_ibl_wheel.position.npy') &
            dataset_eids(one, 'spikes.times.npy'))
    ...
    if TARGET_EIDS.exists():
        targets = [x.strip() for x in TARGET_EIDS.read_text().splitlines() if x.strip()]
        ...
        if selected:
            return selected
```

```python
tr = trial_frame(one.load_object(eid, 'trials'))
wheel = one.load_object(eid, 'wheel')
camera, ct, me = load_camera(one, eid)
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
```

iii. In `CONVERSION_NOTES.md`, the AI says the staged release manifests were stale, so it used `make_parquet_db` to create an accurate local ONE index. It also says processing all complete local sessions would be too slow, so it restricted the cohort to the methods-paper session list after mapping those EIDs to the local hashed IDs.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the session metadata cache. After all retained sessions are processed, the AI creates a sorted unique `subjects` list and builds `subject_idx` by indexing each session's subject into that list.

ii. 
```python
def session_details(one, eid):
    row = one._cache['sessions'].loc[eid]
    return {k: (str(row[k]) if k in row and pd.notna(row[k]) else None)
            for k in ('subject', 'date', 'number', 'lab')}
```

```python
subjects.append(info['subject'] or 'unknown')
...
subject_list=sorted(set(subjects))
subject_idx=np.array([subject_list.index(x) for x in subjects],dtype=np.int64)
```

iii. The notes say the subject identifier should come directly from ONE session metadata rather than being inferred from paths or filenames.

## 1-c. How are the data split into sessions?

i. Each selected ONE `eid` is treated as one session. The converter iterates over the selected EIDs and stores one entry per EID in the top-level `neural`, `input`, and `output` session lists.

ii. 
```python
one=get_one(); eids=candidate_eids(one)
...
for k,eid in enumerate(eids,1):
    n,i,o,r,info,cont=process_session(one,eid,plot=args.show_processing and k<=2)
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. The notes treat a session as the natural unit returned by ONE and preserve that unit in the converted dataset.

## 1-d. How are the data split into trials?

i. The AI loads the ONE `trials` object, normalizes it into a pandas DataFrame, and then uses one row per trial. After trial filtering, `idx = np.flatnonzero(mask)` defines the retained trials, and all per-trial arrays are created by indexing into those retained rows.

ii. 
```python
def trial_frame(tr):
    if 'table' in tr and isinstance(tr['table'], pd.DataFrame):
        df = tr['table'].copy()
        ...
        return df
    ...
    return pd.DataFrame(cols)
```

```python
tr = trial_frame(one.load_object(eid, 'trials'))
mask = valid_trials(tr, wt, ct)
idx = np.flatnonzero(mask)
...
for j in range(len(idx)):
    ...
    neural.append(neural_arr[j]); inputs.append(inp); outputs.append(out)
```

iii. The notes describe the trials table as the source of trial rows, with one common mask applied before all per-trial arrays are built.

## 1-e. How are trials filtered based on quality controls?

i. The AI requires finite `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, and `firstMovement_times`; keeps only `choice` values in `{-1, 1}` and prior values close to `{0.2, 0.5, 0.8}`; optionally enforces `0 < duration <= 10` when `intervals` are present; requires the entire `[-0.5, 1.5]` window to lie within the min/max wheel and camera timestamps; then removes any trial whose interpolated wheel or whisker trace contains `NaN`. Sessions with fewer than two retained trials are excluded.

ii. 
```python
required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
            'stimOn_times', 'firstMovement_times']
for c in required:
    mask &= np.isfinite(df[c].to_numpy(float))
...
mask &= np.isin(choice, [-1, 1])
mask &= np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(axis=1)
...
mask &= np.isfinite(duration) & (duration <= 10) & (duration > 0)
...
mask &= (starts >= wheel_t[0]) & (ends <= wheel_t[-1])
mask &= (starts >= cam_t[0]) & (ends <= cam_t[-1])
```

```python
wheel_cont = sample_trials(wt, ws, stim)
whisk_cont = sample_trials(ct, me, stim)
finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
idx, stim = idx[finite], stim[finite]
...
if len(idx) < 2: raise RuntimeError('fewer than two finite behavior trials')
```

iii. The notes say this combines paper/code trial-field checks with additional complete-support requirements for wheel and whisker streams, and that missing outputs should be dropped rather than imputed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is built from per-spike `spikes['times']` and `spikes['clusters']`. Cluster metadata, especially `cluster_id`, `label`, and `atlas_id`, are used to decide which units survive QC and how they are annotated by brain region.

ii. 
```python
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
cid = np.asarray(clusters['cluster_id']).astype(int)
label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
```

```python
sc = np.asarray(spikes['clusters']).astype(int)
all_t.append(np.asarray(spikes['times'])[keep].astype(np.float64))
all_c.append(mapped)
```

iii. The notes explicitly say the converter should use `SpikeSortingLoader`, merged cluster metadata, and Beryl anatomy, with spike times and cluster assignments as the basis of the neural representation.

## 2-b. How is the `neural` data processed?

i. The AI merges all probes within a session, remaps surviving cluster IDs to a compact contiguous range, sorts all surviving spikes by time, and bins spikes into 20 ms stimulus-aligned bins. The stored neural arrays are float32 spike counts per bin, not firing rates in Hz and not smoothed traces.

ii. 
```python
lut = {int(c): i + offset for i, c in enumerate(good_ids)}
...
all_t.append(np.asarray(spikes['times'])[keep].astype(np.float64))
all_c.append(mapped)
...
t = np.concatenate(all_t); c = np.concatenate(all_c)
order = np.argsort(t, kind='stable')
return t[order], c[order], np.asarray(all_reg, dtype=object), raw_units, probes
```

```python
def bin_spikes(spike_t, spike_c, stim, n_units):
    out = np.zeros((len(stim), n_units, N_TIME), dtype=np.float32)
    for i, st in enumerate(stim):
        lo, hi = st + OFF_START, st + OFF_END
        ...
        out[i] = np.bincount(flat, minlength=n_units * N_TIME).reshape(n_units, N_TIME)
    return out
```

iii. In the notes, the AI says the reference cache stage uses spike counts, so it intentionally preserved spike counts rather than applying later model-side standardization or firing-rate conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `label >= 1` and `atlas_id > 0`. Probes with no surviving units are skipped; sessions with no surviving units raise an error and are excluded.

ii. 
```python
label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
good = (label >= 1) & (atlas > 0)
good_ids = cid[good]
if not len(good_ids):
    continue
...
if offset == 0:
    raise RuntimeError('no QC-passing units')
```

iii. The notes justify this as using the strict `label >= 1` unit QC from the reference code while additionally requiring a resolved anatomical assignment.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to stimulus onset. For each retained trial, the code defines a window from `stimOn_times - 0.5` to `stimOn_times + 1.5`, slices spikes in that absolute-time interval, and bins them relative to the trial start of that aligned window.

ii. 
```python
stim_all = tr.stimOn_times.to_numpy(float)
stim = stim_all[idx]
...
lo, hi = st + OFF_START, st + OFF_END
a, b = np.searchsorted(spike_t, [lo, hi])
relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
```

iii. The notes say the task and methods code both require stimulus alignment with the `[-0.5, 1.5]` window, so the AI followed that instead of the movement-aligned analyses in the data paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 20 ms bins, giving 100 bins over the 2 s window from -0.5 s to +1.5 s. No additional temporal rebinning or smoothing is applied.

ii. 
```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
```

iii. The notes repeatedly state that the methods cache defaults are `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, and `binsize=0.02`, so the AI copied those settings directly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the alignment event `stimOn_times` plus a fixed relative bin-center grid. The actual stored values are the relative bin centers, not a raw recorded signal.

ii. 
```python
stim_all = tr.stimOn_times.to_numpy(float)
...
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

```python
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. The notes describe this input as the fixed bin-center time axis for the stimulus-aligned window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI defines a constant vector of bin centers from -0.49 s to +1.49 s at 20 ms spacing and repeats that vector for every retained trial.

ii. 
```python
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

```python
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. The notes say this is the common trial-relative time grid shared by all modalities.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is exactly the same time grid used for neural binning: neural spikes are binned into bins defined by `EDGES_REL`, and the time input is the corresponding `CENTERS_REL` values for those bins.

ii. 
```python
relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
```

```python
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

iii. The notes say all modalities share a single stimulus-relative 100-bin axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial sequence of `probabilityLeft` values. A new block starts whenever `probabilityLeft` changes from one trial to the next.

ii. 
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
    return out
```

```python
prior_all = tr.probabilityLeft.to_numpy(float)
block_no_all = trial_number_in_block(prior_all)
```

iii. The notes justify this by saying the raw trials table does not contain a block index, so block structure must be recovered from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a zero-based count over the full trial sequence, resets the count whenever `probabilityLeft` changes, and then looks up that precomputed count for each retained trial. The scalar count is repeated across all 100 time bins in the trial input array.

ii. 
```python
for i in range(1, len(prior)):
    out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
```

```python
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. The notes explicitly say the count should be computed before filtering so that dropped trials still advance the animal's true position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials.choice`. After filtering to `{-1, 1}`, the AI maps `choice == 1` to class 1 and `choice == -1` to class 0, then repeats that class across the whole trial.

ii. 
```python
choice = tr.choice.to_numpy(float)[idx]
choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
```

```python
out = np.vstack((np.full(N_TIME, choice_cls[j], np.int64),
                 np.full(N_TIME, prior_cls[j], np.int64),
                 wheel_cls[j], whisk_cls[j])).astype(np.int64)
```

iii. The notes say the task needs a binary categorical choice output repeated over time, and the code comment shows the AI interpreted `-1` as left and `+1` as right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The only processing is filtering to valid binary choices and converting them to a binary class with a boolean comparison `choice == 1`. The resulting class is then broadcast to all time bins of the trial.

ii. 
```python
mask &= np.isin(choice, [-1, 1])
...
choice_cls = (choice == 1).astype(np.int64)
```

```python
np.full(N_TIME, choice_cls[j], np.int64)
```

iii. The notes present this as a task-required categorical remapping with no additional temporal processing.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials.probabilityLeft`.

ii. 
```python
prior_all = tr.probabilityLeft.to_numpy(float)
prior = prior_all[idx]
prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)
```

iii. The notes say the source prior has exactly the three expected values `{0.2, 0.5, 0.8}` and should be converted into the requested 3-class output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Trials are first filtered so the prior is close to one of `{0.2, 0.5, 0.8}`, then each retained prior is assigned to the nearest of those three values by `argmin`, yielding classes `0`, `1`, and `2`. The class is repeated across time bins.

ii. 
```python
mask &= np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(axis=1)
```

```python
prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)
...
np.full(N_TIME, prior_cls[j], np.int64)
```

iii. The notes describe this as a straightforward task-mandated categorical conversion.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel timestamps and wheel position arrays in the ONE `wheel` object.

ii. 
```python
wheel = one.load_object(eid, 'wheel')
wt, ws = wheel_speed(wheel['timestamps'], wheel['position'])
```

```python
def wheel_speed(t, pos):
    t, pos = clean_timeseries(t, pos)
    speed = np.abs(np.gradient(pos, t))
    ...
    return t, speed
```

iii. The notes say the requested output is speed rather than signed velocity, so the converter should take the absolute magnitude of wheel motion.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI removes nonfinite samples, sorts timestamps, drops duplicate times, computes timestamp-aware absolute velocity with `np.gradient`, interpolates that continuous speed onto the common stimulus-aligned bin centers for each trial, and stores those continuous values only long enough to discretize them.

ii. 
```python
def clean_timeseries(t, x):
    ...
    order = np.argsort(t, kind='stable'); t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], x[keep]
```

```python
def wheel_speed(t, pos):
    t, pos = clean_timeseries(t, pos)
    speed = np.abs(np.gradient(pos, t))
    speed[~np.isfinite(speed)] = np.nan
    return t, speed
```

```python
wheel_cont = sample_trials(wt, ws, stim)
```

iii. The notes say the code uses a timestamp-aware gradient after timestamp QC because the staged local cache made an explicit SessionLoader-based wheel path inconvenient, while still matching the stimulus-relative binning of the reference pipeline.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI discretizes wheel speed into three within-session classes using the 1/3 and 2/3 quantiles over all retained wheel-speed values in that session. If those thresholds collapse because the distribution is degenerate, it falls back to thresholding based on the distinct values present.

ii. 
```python
def discretize_tertiles(values):
    q1, q2 = np.nanquantile(values, [1/3, 2/3])
    method = 'quantile'
    if not q2 > q1:
        uniq = np.unique(values[finite])
        if len(uniq) >= 3:
            q1, q2 = np.quantile(uniq, [1/3, 2/3]); method = 'unique_value_quantile'
        elif len(uniq) == 2:
            q1, q2 = uniq[0], uniq[0]; method = 'binary_degenerate'
        else:
            q1 = q2 = uniq[0]; method = 'constant'
    out = np.zeros(values.shape, dtype=np.int64)
    out[values > q1] = 1
    out[values > q2] = 2
```

```python
wheel_cls, wheel_thr, wheel_method = discretize_tertiles(wheel_cont)
```

iii. The notes justify within-session tertiles because wheel scale is session-dependent and the task explicitly requires three bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is aligned by evaluating wheel speed at exactly the same stimulus-relative bin centers used for the neural bins.

ii. 
```python
q = stim[:, None] + CENTERS_REL[None, :]
y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
```

```python
wheel_cont = sample_trials(wt, ws, stim)
```

iii. The notes say all modalities should share a common 100-bin stimulus-relative timeline.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `times` and `ROIMotionEnergy` in a side-camera ONE object. The AI prefers the left camera when valid and falls back to the right camera.

ii. 
```python
def load_camera(one, eid):
    for side in ('left', 'right'):
        try:
            cam = one.load_object(eid, f'{side}Camera', collection='alf')
            t, me = clean_timeseries(cam['times'], cam['ROIMotionEnergy'])
            if len(t) > 100 and len(t) == len(me):
                return side, t, me
```

iii. The notes say left/right camera streams should not be averaged together; instead one side should be chosen consistently per session, with left preferred.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI cleans timestamps and values, removes nonfinite samples and duplicate times, then linearly interpolates the released motion-energy signal onto the common trial-relative bin centers. It does not apply additional filtering or normalization before discretization.

ii. 
```python
def clean_timeseries(t, x):
    ...
    ok = np.isfinite(t) & np.isfinite(x)
    ...
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], x[keep]
```

```python
camera, ct, me = load_camera(one, eid)
...
whisk_cont = sample_trials(ct, me, stim)
```

iii. The notes say the released ROI motion energy should be used as provided and aligned onto the same decoder bins as the other streams.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: within-session tertiles based on the 1/3 and 2/3 quantiles of the retained whisker-motion-energy values, with a fallback for degenerate distributions.

ii. 
```python
whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)
```

```python
out[values > q1] = 1
out[values > q2] = 2
```

iii. The notes say session-specific thresholds are necessary because motion-energy values are in arbitrary units and can vary across sessions and camera views.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned by interpolation to the same `CENTERS_REL` grid that defines the neural bins.

ii. 
```python
q = stim[:, None] + CENTERS_REL[None, :]
y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
```

```python
whisk_cont = sample_trials(ct, me, stim)
```

iii. The notes say the wheel, whisker, and neural data all share a single stimulus-relative 100-bin axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are excluded rather than imputed. The AI removes nonfinite continuous samples, drops trials missing required task variables, drops trials outside wheel/camera support, drops trials whose interpolated wheel or whisker traces contain `NaN`, skips invalid camera streams, skips probes with no QC-passing units, and excludes sessions with no QC units or fewer than two retained trials.

ii. 
```python
ok = np.isfinite(t) & np.isfinite(x)
...
mask &= np.isfinite(df[c].to_numpy(float))
...
if len(idx) < 2: raise RuntimeError(f'only {len(idx)} valid trials')
...
finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
idx, stim = idx[finite], stim[finite]
...
if offset == 0:
    raise RuntimeError('no QC-passing units')
```

```python
except Exception as exc:
    failures.append({'eid':str(eid),'error':f'{type(exc).__name__}: {exc}'})
    print('  EXCLUDED:',failures[-1]['error'],flush=True)
```

iii. The notes explicitly say missing outputs should not be fabricated and that the correct response is to exclude affected trials or sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading and processing spike-sorting data for each probe/session, especially large spike arrays and merged cluster metadata. The notes also identify full-session iteration over hundreds of sessions as the main runtime bottleneck.

ii. 
```python
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
```

```python
for k,eid in enumerate(eids,1):
    ...
    n,i,o,r,info,cont=process_session(one,eid,plot=args.show_processing and k<=2)
```

iii. The notes say full spike arrays can contain tens of millions of spikes per probe and that processing all complete local sessions would have taken roughly 82 minutes without the cohort restriction.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining per-trial Python loop is the spike-binning loop in `bin_spikes`. There is also a per-trial assembly loop that repeatedly builds `input`, `output`, and `neural` list entries one trial at a time. Behavioral interpolation was already vectorized across all trial/bin query points.

ii. 
```python
def bin_spikes(spike_t, spike_c, stim, n_units):
    out = np.zeros((len(stim), n_units, N_TIME), dtype=np.float32)
    for i, st in enumerate(stim):
        ...
        out[i] = np.bincount(flat, minlength=n_units * N_TIME).reshape(n_units, N_TIME)
```

```python
inputs=[]; outputs=[]; neural=[]
for j in range(len(idx)):
    inp = np.vstack((CENTERS_REL.astype(np.float32),
                     np.full(N_TIME, block_no_all[idx[j]], np.float32)))
    out = np.vstack((np.full(N_TIME, choice_cls[j], np.int64),
                     np.full(N_TIME, prior_cls[j], np.int64),
                     wheel_cls[j], whisk_cls[j])).astype(np.int64)
```

iii. The notes emphasize vectorized behavior interpolation and `bincount`-based spike binning as speedups, but the final implementation still leaves the per-trial spike loop and per-trial output assembly in Python.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly casts `CENTERS_REL` to float32 and repeatedly constructs constant trial-wise arrays inside the per-trial loop. It also repeatedly performs linear `list.index(...)` searches when building `subject_idx` and `brain_region_idx`.

ii. 
```python
for j in range(len(idx)):
    inp = np.vstack((CENTERS_REL.astype(np.float32),
                     np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

```python
subject_idx=np.array([subject_list.index(x) for x in subjects],dtype=np.int64)
...
region_idx=[np.array([brain_regions.index(x) for x in r],dtype=np.int64) for r in region_names]
```

iii. The notes are mostly focused on large I/O/runtime costs rather than these smaller repeated operations, so these repetitions appear to be incidental implementation choices rather than explicitly justified decisions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes continuous wheel and whisker traces for every retained trial even though the saved dataset keeps only the discretized class labels. Those continuous arrays are discarded after discretization, except when optional plotting is enabled.

ii. 
```python
wheel_cont = sample_trials(wt, ws, stim)
whisk_cont = sample_trials(ct, me, stim)
...
wheel_cls, wheel_thr, wheel_method = discretize_tertiles(wheel_cont)
whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)
```

```python
continuous = (wheel_cont, whisk_cont) if plot else None
return neural, inputs, outputs, regions, info, continuous
```

iii. The notes justify the continuous traces as necessary intermediates for thresholding and plotting, but they are not preserved in the final saved decoder dataset.
