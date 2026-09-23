# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `one.search()` as the main session index. It reads `/app/code/code_zhang2025/data/bwm_release.csv`, optionally restricts that table with `/app/data/DATALIMIT_SUBSET.csv`, groups rows by `eid`, and then loads session contents from the staged local ONE cache. Spike data are loaded probe-by-probe with `SpikeSortingLoader`; wheel data are loaded with `SessionLoader`; trials and whisker motion-energy arrays are read directly from the local ALF files.

ii. 
```python
def selected_sessions(n=None):
    bwm = pd.read_csv(RELEASE, index_col=0)
    subset_file = ROOT/'data/DATALIMIT_SUBSET.csv'
    if subset_file.exists():
        subset = pd.read_csv(subset_file)
        values = set(subset.astype(str).to_numpy().ravel())
        bwm = bwm[bwm.eid.astype(str).isin(values) | bwm.pid.astype(str).isin(values)]
```

```python
one = ONE(cache_dir=CACHE, mode='local', silent=True)
one._cache = load_tables(CACHE/'Brainwidemap')
```

```python
spikes, clusters = local_spiking_data(one, eid, pid, pname)
trials = load_local_trials(one, eid)
wheel_all, whisk_all, wheel_good, whisk_good = local_continuous_behaviors(one, eid, trials)
```

iii. The trajectory first justified a small deterministic seed-42 subset as a manageable proxy, then later switched to processing the full bundled BWM release after discovering the staged local dataset layout. It also justified local cache loading as a way to avoid mutating or reloading the shared cache tables and to work without Alyx network access.

## 1-b. How are the data split into subjects?

i. Subjects come from the `subject` column of `bwm_release.csv`. Each grouped `eid` contributes one `(subject, eid, pids, probe_names)` row, and the final output builds `subjects` as sorted unique subject names with `subject_idx` pointing from each session to its subject.

ii. 
```python
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid),
                 list(group.pid.astype(str)), list(group.probe_name.astype(str))))
```

```python
subjects = sorted(set(s['subject'] for s in sessions))
smap = {s:i for i,s in enumerate(subjects)}
subject_idx=np.array([smap[s['subject']] for s in sessions], np.int32)
```

iii. No separate trajectory argument was given beyond using the release table as the session manifest.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. All probes listed for a given `eid` are merged into one session entry.

ii. 
```python
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid),
                 list(group.pid.astype(str)), list(group.probe_name.astype(str))))
```

iii. The trajectory treated the release table as the authoritative session list and later reported accounting for every `eid` in that release.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the consolidated `_ibl_trials.table.pqt` table. The code assumes each row of that table is one trial, then filters rows with a Boolean mask and keeps `idx = np.flatnonzero(valid)`.

ii. 
```python
def load_local_trials(one, eid):
    path = Path(one.eid2path(eid))/'alf'
    files = sorted(path.glob('#*#/_ibl_trials.table.pqt'))
    if not files:
        files = sorted(path.glob('_ibl_trials.table.pqt'))
    return pd.read_parquet(files[-1])
```

```python
valid = paper_mask & wheel_good & whisk_good
idx = np.flatnonzero(valid)
```

iii. The trajectory says the newest consolidated trial table was used to avoid obsolete ALF attributes and revision issues.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have non-missing required columns, reaction time between 0.08 and 2.0 s, trial duration from `goCue_times` to `feedback_times` at most 10 s, and nonzero choice. After that, the trial must also have usable wheel and whisker traces spanning the stimulus-aligned window; sessions with fewer than two valid trials are dropped.

ii. 
```python
required = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
            'firstMovement_times', 'feedbackType']
valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
rt = trials.firstMovement_times.to_numpy() - trials.stimOn_times.to_numpy()
duration = trials.feedback_times.to_numpy() - trials.goCue_times.to_numpy()
valid &= (rt >= 0.08) & (rt <= 2.0)
valid &= duration <= 10.0
valid &= trials.choice.to_numpy() != 0
```

```python
valid = paper_mask & wheel_good & whisk_good
idx = np.flatnonzero(valid)
if len(idx) < 2:
    raise RuntimeError('fewer than two valid trials')
```

iii. The trajectory explicitly says it wanted the “paper trial QC” plus a “proper intersection of wheel, whisker, and trial-validity masks” so that every retained trial had all requested decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays are derived from per-spike times and per-spike cluster IDs loaded probe-by-probe, with the cluster table used for anatomical labels.

ii. 
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
neural_df = {'spike_times': spikes['times'], 'spike_clusters': spikes['clusters']}
```

iii. The trajectory repeatedly states that neural processing uses the probe spike-sorting outputs and merged probes from the Zhang code path.

## 2-b. How is the `neural` data processed?

i. Probe spike trains are merged within session, then `bin_spiking_data` bins spikes into a 2 s stimulus-aligned window with 20 ms bins. The result is transposed to neuron-by-time per trial and stored as `int16` spike counts; the code does not convert counts to firing rates.

ii. 
```python
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
...
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
...
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
```

iii. The trajectory explicitly justified “20-ms spike counts from -0.5 to 1.5 s around stimulus onset” and described the representation as “all Kilosort 2.5 clusters, probes merged by session.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps all merged clusters from every listed probe. There is no quality-label filter and no exclusion of `void` clusters before neural construction.

ii. 
```python
cluster_ids = np.arange(len(clusters), dtype=np.int64)
...
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. The trajectory explicitly says “Reference neural processing uses all Kilosort clusters (`qc=None`), merges probes within each session,” and the earlier draft also annotated this as “Every merged cluster is retained.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times` using a fixed window from -0.5 s to +1.5 s around stimulus onset.

ii. 
```python
OFF0, OFF1 = -0.5, 1.5
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
```

iii. The trajectory explicitly calls this a “common 20 ms stimulus-aligned window” and repeatedly describes the converter as stimulus aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins across the 2 s window. No temporal rebinning or smoothing is applied to the neural array after spike counting.

ii. 
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
```

iii. The trajectory explicitly says “20-ms spike counts” and “Every retained trial has 100 stimulus-aligned 20 ms bins.”

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time input is not read from a raw data column directly. It is synthesized from the fixed stimulus-aligned window definition, implicitly tied to `stimOn_times` because all trialized signals are aligned to that event.

ii. 
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
...
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. The trajectory justified a common stimulus-aligned representation, but did not give a separate rationale for encoding elapsed time other than sharing that time axis.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It is created as a fixed 100-sample vector spanning the stimulus-aligned window at 20 ms spacing and then copied into every trial. The code uses `-0.48, -0.46, ... , 1.50` rather than explicit bin centers.

ii. 
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
...
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. No explicit trajectory justification was given beyond wanting all variables on the same 100-bin temporal axis.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same fixed-length vector is paired with every neural trial, so it is session- and trial-aligned by construction. However, the code’s time values correspond to the interpolation grid used for behavior, not an explicit copy of the neural bin centers.

ii. 
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1, dtype=np.float32)*BIN + OFF0
...
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. The trajectory justification was that static variables would be repeated across time and all outputs would share one 100-bin axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` sequence in the trials table. A change in `probabilityLeft` starts a new block.

ii. 
```python
def trial_number_in_block(prob):
    ...
    for i, value in enumerate(prob):
        if i == 0 or value != previous:
            k = 1
```

```python
block_trial = trial_number_in_block(trials.probabilityLeft.to_numpy())[idx]
```

iii. No separate trajectory justification was given, but this follows the standard IBL interpretation that block identity is recovered from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the full per-session `probabilityLeft` sequence before trial filtering and increments a counter within each run of equal values. The first trial in a block is encoded as `1`, not `0`.

ii. 
```python
def trial_number_in_block(prob):
    out = np.empty(len(prob), dtype=np.float32)
    k = 0
    previous = None
    for i, value in enumerate(prob):
        if i == 0 or value != previous:
            k = 1
        else:
            k += 1
        out[i] = k
```

iii. The trajectory did not explicitly justify the one-based counting choice.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the `choice` column of the trials table after invalid trials are removed.

ii. 
```python
choice_raw = trials.choice.to_numpy()[idx]
```

iii. The trajectory notes that the code enforces remaining choices to be `-1` or `1` after trial filtering.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code converts the filtered `choice` values to a binary output using `choice_raw == 1`, producing `1` for raw `+1` trials and `0` otherwise.

ii. 
```python
if not np.all(np.isin(choice_raw, [-1, 1])):
    raise ValueError(f'unexpected choices {np.unique(choice_raw)}')
choice = (choice_raw == 1).astype(np.int8)
```

iii. The trajectory’s only explicit rationale was checking that only `-1` and `1` remained before binarization.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. 
```python
pleft_raw = trials.probabilityLeft.to_numpy()[idx]
```

iii. No separate trajectory rationale was given beyond following the decoder specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The filtered probabilities are rounded to one decimal place and mapped as `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`.

ii. 
```python
pmap = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([pmap[round(float(x), 1)] for x in pleft_raw], np.int8)
```

iii. The trajectory treated this as a straightforward task-driven categorical remapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from wheel timestamps and wheel velocity returned by `SessionLoader.load_wheel()`. The script uses the absolute value of `sl.wheel.velocity`.

ii. 
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
wheel_t = sl.wheel.times.to_numpy()
wheel_v = np.abs(sl.wheel.velocity.to_numpy())
```

iii. The trajectory explicitly described this as “wheel speed from smoothed absolute wheel velocity.”

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. After taking absolute wheel velocity, the code linearly interpolates it onto a fixed 100-sample stimulus-aligned grid for each trial, provided the source trace covers the full window within one bin at each edge.

ii. 
```python
def interpolate(times, values):
    ...
    ib = np.searchsorted(times, beg, side='right')
    ie = np.searchsorted(times, end, side='left')
    tx, vx = times[ib:ie], values[ib:ie]
    ...
    xi = np.linspace(beg+BIN, end, nbin)
    out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. The trajectory justified reuse of SessionLoader wheel processing and said behavior traces would be aligned after interpolation onto the common stimulus-aligned grid.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized after all retained sessions are processed. The script concatenates every retained wheel sample across sessions, computes global tertile edges, and applies `np.digitize` per trial.

ii. 
```python
def category_edges(sessions, key):
    values = np.concatenate([np.concatenate(s[key]) for s in sessions])
    edges = np.quantile(values, [1/3, 2/3])
```

```python
wheel_edges = category_edges(sessions, 'wheel')
...
wb = np.digitize(s['wheel'][j], wheel_edges).astype(np.int8)
```

iii. The trajectory explicitly says “Dynamic variables will be discretized into global tertiles after alignment.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Each trial’s wheel trace is aligned to `stimOn_times` and interpolated onto the same 100-step stimulus-aligned trial window used for neural trialization, but the interpolation grid is the bin right-edge sequence `beg + BIN` through `end`.

ii. 
```python
beg, end = onset + OFF0, onset + OFF1
...
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. The trajectory justified putting all variables on one shared 100-bin axis.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<side>Camera.ROIMotionEnergy.npy` together with `_ibl_<side>Camera.times.npy`. The code prefers the left camera and falls back to the right camera if needed, selecting a motion-energy file whose length matches the camera timestamps.

ii. 
```python
for view in ('left', 'right'):
    tf = sorted(alf.glob(f'#*#/_ibl_{view}Camera.times.npy'))
    ...
    mf = sorted(alf.glob(f'#*#/{view}Camera.ROIMotionEnergy.npy'))
```

```python
if len(m) == len(t):
    match = m
```

iii. The trajectory explicitly described whisker motion energy as using the left camera with right fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion-energy trace is used without additional filtering. Like wheel speed, it is linearly interpolated trial-by-trial onto a 100-sample stimulus-aligned grid, and trials lacking usable coverage are rejected.

ii. 
```python
whisk, whisk_good = interpolate(motion_t, motion_v)
```

```python
if len(vx) == 0 or not np.all(np.isfinite(vx)):
    out.append(None); continue
if abs(beg-tx[0]) > BIN or abs(end-tx[-1]) > BIN:
    out.append(None); continue
```

iii. The trajectory justified matching the paper-style whisker source and putting it on the same aligned time axis as the other outputs.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded the same way as wheel speed: all retained whisker samples across sessions are pooled, global tertile cut points are computed, and `np.digitize` converts each time point to one of three categories.

ii. 
```python
whisk_edges = category_edges(sessions, 'whisk')
...
mb = np.digitize(s['whisk'][j], whisk_edges).astype(np.int8)
```

iii. The trajectory explicitly grouped the dynamic variables together and said they would use “global tertiles after alignment.”

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned per trial to stimulus onset and interpolated onto the same 100-step window as the neural trials, again using the right-edge interpolation grid.

ii. 
```python
beg, end = onset + OFF0, onset + OFF1
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. The trajectory’s rationale was the same shared 100-bin stimulus-aligned representation used across outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is mostly drop-or-skip based. Trials with missing required fields or incomplete wheel/whisk coverage are dropped. Sessions with fewer than two jointly valid trials are skipped. Sessions without matching whisker timestamps and motion-energy arrays are skipped with a failure marker. The script also caches session successes and failures to avoid recomputation.

ii. 
```python
if motion_t is None:
    raise FileNotFoundError('no matching whisker motion energy and camera times')
...
valid = paper_mask & wheel_good & whisk_good
idx = np.flatnonzero(valid)
if len(idx) < 2:
    raise RuntimeError('fewer than two valid trials')
```

```python
fail_file = WORK/f'{eid}.failed'
...
if fail_file.exists():
    failures.append((eid, fail_file.read_text().strip()))
    print(f'SKIP {eid}: cached failure: {fail_file.read_text().strip()}', flush=True)
    continue
```

iii. The trajectory explicitly justified intersecting validity masks so requested outputs never contain missing data, and later described the final excluded sessions as those lacking matching whisker streams or having fewer than two valid trials.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is loading spike-sorting data for each probe and then trial-binning the merged spikes. Behavior interpolation is lighter by comparison.

ii. 
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

```python
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
```

iii. The trajectory repeatedly focused optimization effort on cache-table handling and resumable per-session caching, which implies probe data loading and session processing were the dominant costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorizable loops are the Python loop used to compute `trial_number_in_block`, the per-trial interpolation loop inside `local_continuous_behaviors()`, and the per-trial assembly loop that creates input/output matrices.

ii. 
```python
for i, value in enumerate(prob):
    ...
```

```python
for i, onset in enumerate(trials.stimOn_times.to_numpy()):
    ...
    out.append(np.interp(xi, tx, vx).astype(np.float32))
```

```python
for j in range(len(s['neural'])):
    T = s['neural'][j].shape[1]
    inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. The trajectory does not discuss vectorization directly; these are deductions from the final code structure.

## 10-c. What processing does the code repeat multiple times?

i. The same interpolation logic is run separately for wheel and whisker traces. On reruns, the script also repeatedly scans cached session pickle and failure files during assembly.

ii. 
```python
wheel, wheel_good = interpolate(wheel_t, wheel_v)
whisk, whisk_good = interpolate(motion_t, motion_v)
```

```python
if fail_file.exists():
    ...
if cache_file.exists():
    print('Loading intermediate', cache_file, flush=True)
```

iii. No explicit trajectory justification was given for this repetition beyond making the conversion resumable.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script includes operational work that is not needed by the decoder itself: sharding support, cache-only mode, per-session intermediate pickle files, failure-marker files, and extra metadata such as `source_trial_idx`, `failed_sessions`, and stored discretization edges. It also keeps a stale top-of-file docstring describing a 10-session seed-42 subset even though the final code processes the full release by default.

ii. 
```python
shard_count = int(os.environ.get('IBL_SHARD_COUNT', '1'))
shard_index = int(os.environ.get('IBL_SHARD_INDEX', '0'))
rows = [row for i, row in enumerate(rows) if i % shard_count == shard_index]
...
if os.environ.get('IBL_CACHE_ONLY') == '1':
    sessions.pop()
    del sess
```

```python
session_info.append(dict(eid=s['eid'], subject=s['subject'],
    n_trials=len(s['neural']), n_neurons=len(s['regions']),
    source_trial_indices=s['source_trial_idx'].tolist()))
...
session_info=session_info, failed_sessions=failures
```

iii. The trajectory explicitly justified the intermediate caches, sharding, and failure markers as practical mechanisms for scaling to the full release and resuming interrupted processing.
