# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script initializes `ONE`, loads the `Brainwidemap` cache, and patches the cached trials-table metadata so `SessionLoader` can see the canonical trials table. It does not discover sessions from `one.search`; instead it reads `/app/code/code_zhang2025/data/bwm_release.csv` to define the cohort of session EIDs and probe PIDs. Scientific arrays are then loaded through `SessionLoader` and `SpikeSortingLoader`.

ii. 
```python
def init_one() -> ONE:
    one = ONE(silent=True)
    one.load_cache(tag='Brainwidemap', clobber=True)
    ds = one._cache['datasets']
    mask = ds.rel_path.str.endswith('_ibl_trials.table.pqt')
    ds.loc[mask, 'default_revision'] = True
    return one
```

```python
def cohort(sample: bool) -> tuple[pd.DataFrame, list[str]]:
    f = pd.read_csv(FREEZE).drop(columns=['Unnamed: 0'], errors='ignore')
    eids = list(dict.fromkeys(f.eid.astype(str)))
    if sample:
        eids = eids[:2]
    return f[f.eid.astype(str).isin(eids)].copy(), eids
```

```python
sl = SessionLoader(one=one, eid=eid)
loader = SpikeSortingLoader(pid=str(row.pid), one=one)
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, and trajectory step 50, the AI says the freeze CSV should define the paper/reference cohort while all scientific data still load exclusively through ONE and brainbox. Trajectory steps 31 to 34 justify the in-memory metadata patch as a workaround for the staged cache's trials-table revision issue.

## 1-b. How are the data split into subjects?

i. Subjects are assigned per session after processing. The script looks up each session's subject in `one._cache['sessions']`; if that fails, it falls back to the corresponding subject stored in the freeze CSV. Final `subjects` is the sorted unique list, and `subject_idx` maps each retained session to that list.

ii.
```python
try:
    details = one._cache['sessions'].loc[uuid.UUID(str(eid))]
    subject, lab, date = str(details.subject), str(details.lab), str(details.date)
except (KeyError, ValueError):
    fr = freeze.iloc[0]
    subject, lab, date = str(fr.subject), str(fr.lab), str(fr.date)
```

```python
subjects=sorted({x['subject'] for x in infos})
subject_idx=np.array([subjects.index(x['subject']) for x in infos],dtype=np.int32)
```

iii. The notes say subject identity should come from ONE session metadata, with the freeze table used only as a provenance fallback after a UUID-index bug was discovered and fixed in Step 7.

## 1-c. How are the data split into sessions?

i. A session is one unique EID from the freeze file. The main loop iterates over those EIDs, processes each one independently, and appends one session entry each to `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
eids = list(dict.fromkeys(f.eid.astype(str)))
```

```python
for i,eid in enumerate(eids,1):
    ...
    result=('ok',eid,process_session(one,freeze[freeze.eid.astype(str)==eid],eid))
```

iii. The notes explicitly say session order should follow the freeze/reference order and that the session is the natural processing unit for loading, QC, and saving.

## 1-d. How are the data split into trials?

i. Trials come directly from the session trials table loaded by `SessionLoader`. Each row of `sl.trials` is treated as one native trial; filtering later selects a subset of row indices, and one converted trial is produced for each retained row.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
tr = sl.trials
```

```python
idx = np.flatnonzero(valid)
for j, raw_i in enumerate(idx):
    ...
    inputs.append(inp); outputs.append(out); neural.append(neural_all[j].copy())
```

iii. The AI treated the IBL trials table as the authoritative trial split. Its notes describe the target as one converted sample per retained trial row.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if several conditions pass: all required trial columns are finite; `choice` is `-1` or `1`; `probabilityLeft` is `0.2`, `0.5`, or `0.8`; total trial duration is at most 10 s; first-movement latency after stimulus onset is between 0.08 and 2.0 s; and the interpolated wheel-speed and whisker traces are finite at every requested bin. Sessions with fewer than two surviving trials are dropped.

ii.
```python
required = ['choice','probabilityLeft','feedbackType','feedback_times','stimOn_times',
            'firstMovement_times','intervals_0','intervals_1']
...
valid = np.all(np.isfinite(vals), axis=1)
valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
valid &= (tr.intervals_1.to_numpy() - tr.intervals_0.to_numpy() <= 10.0)
latency = tr.firstMovement_times.to_numpy() - stim
valid &= (latency >= 0.08) & (latency <= 2.0)
...
valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
if valid.sum() < 2:
    raise RuntimeError(f'only {valid.sum()} valid trials')
```

iii. In Step 4 and Step 5 of `CONVERSION_NOTES.md`, the AI says it intentionally combined the paper's movement-latency/no-go filtering with extra reference-code checks for finite required events, valid priors, maximum trial duration, and complete behavior windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved neural matrices are derived from per-probe spike times and spike cluster assignments. Cluster metadata are also loaded to identify good clusters and assign region names, but the actual per-trial neural array is built from `spikes['times']` and `spikes['clusters']`.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
ids, labels, acr = cluster_fields(clusters)
...
pm = bin_probe(spikes['times'], spikes['clusters'], good_ids, stim)
```

iii. The notes describe the neural source as merged spike-sorting outputs from `SpikeSortingLoader`, with quality labels and acronyms coming from merged cluster metadata.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 100 bins of 20 ms over the window from `-0.5` to `+1.5` s around stimulus onset. Good units from all valid probes in the session are concatenated into one population. The saved representation is integer spike counts per bin, not firing rates and not smoothed.

ii.
```python
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
```

```python
bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
valid = (bi >= 0) & (bi < 100)
flat = ui[keep][valid].astype(np.int64) * 100 + bi[valid]
cnt = np.bincount(flat, minlength=len(good_ids) * 100).reshape(len(good_ids), 100)
out[ti] = cnt.astype(np.uint16)
```

```python
'neural_representation':'uint16 spike counts per 20 ms bin; well-isolated clusters label>=1'
```

iii. The notes justify this as "raw spike counts" matching the methods-paper cache and preserving Poisson structure, and explicitly say no smoothing or normalization should be applied before saving.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only if `label >= 1` and their acronym is not in `{'', 'void', 'root', 'nan', 'none'}`. If a probe has no surviving units it is skipped; if no probe in a session yields any surviving unit, the session is excluded.

ii.
```python
INVALID_REGIONS = {'', 'void', 'root', 'nan', 'none'}
...
ids, labels, acr = cluster_fields(clusters)
region_ok = np.array([x.strip().lower() not in INVALID_REGIONS for x in acr])
keep = (labels >= 1) & region_ok
good_ids, good_acr = ids[keep], acr[keep]
if not len(good_ids):
    probe_info.append({'pid': str(row.pid), 'probe': row.probe_name, 'units': 0})
    continue
```

iii. Step 5 of the notes states the intended rule as "label >= 1" plus a valid non-root anatomical assignment. Trajectory steps 9, 11, 26, 45, and 50 show the AI tied `label >= 1` to the paper's well-isolated-unit criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. For each retained trial, the code counts spikes between `stimOn_times - 0.5 s` and `stimOn_times + 1.5 s`, and assigns them to relative bins on that common window.

ii.
```python
stim = tr.stimOn_times.to_numpy(float)
...
lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
...
bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
```

iii. The notes repeatedly state that the decoder task should use stimulus-onset alignment and the same `[-0.5, 1.5]` s window as the methods-paper cache.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins, exactly 100 bins per trial over a 2 s window. There is no additional temporal rebinning after this fixed binning.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
assert EDGES_REL.size == 101 and CENTERS_REL.size == 100
```

iii. The notes explicitly tie this to the reference cache configuration `interval_len=2`, `binsize=0.02`, `align_time='stimOn_times'`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The input is tied to `trials.stimOn_times`, but the saved values themselves are synthetic fixed bin centers relative to that event: `-0.49, -0.47, ..., 1.49`.

ii.
```python
stim = tr.stimOn_times.to_numpy(float)
```

```python
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
...
inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. In Step 5, the AI says this input should be the common 100-bin stimulus-aligned time axis used for all trials.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw time series is transformed here; the code simply defines the common bin centers for the fixed decoding window and repeats that same vector for every trial.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
...
inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. The justification in the notes is that this is a decoder-design variable, not a measured raw signal.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100 stimulus-relative bin centers as the neural binning. The neural spike counts and the first input row are therefore on the same per-trial time grid.

ii.
```python
q = np.asarray(stim)[:, None] + centers_rel[None, :]
```

```python
bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
```

```python
inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. The notes describe this as using one common stimulus-aligned 20 ms grid for neural data and all time-varying variables.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` trial column. A new block starts whenever `probabilityLeft` changes from the previous native trial.

ii.
```python
def trial_number_in_block(prior: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
    return out
```

iii. Step 5 in the notes explicitly says the block counter should be reconstructed from contiguous runs of `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The counter is zero-based and increments by one whenever the current trial has the same finite prior as the previous trial; otherwise it resets to zero. It is computed on the full native trial sequence before trial filtering and then repeated across all 100 time bins for retained trials.

ii.
```python
block_num = trial_number_in_block(prior)
...
inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. The AI justified this in Step 5 by saying block position is an experimental variable and should not be compressed by removing invalid trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials.choice`. The script keeps only trials with `choice` equal to `-1` or `1`, then encodes `-1` as category `0` and `+1` as category `1`.

ii.
```python
choice = tr.choice.to_numpy(float)
valid &= np.isin(choice, [-1, 1])
...
out[0] = 0 if choice[raw_i] == -1 else 1
```

iii. The notes' mapping table in Step 5 states this decision explicitly as "`-1` (left) -> 0; `+1` (right) -> 1," so the code is implementing the AI's stated convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Beyond filtering out non-binary choices, the code only recodes the sign into `0` or `1` and then repeats that category across all 100 bins of the trial.

ii.
```python
valid &= np.isin(choice, [-1, 1])
...
out = np.empty((4, 100), dtype=np.uint8)
out[0] = 0 if choice[raw_i] == -1 else 1
```

iii. The AI justified the repetition across time in Step 5 as a way to keep all outputs in a common `(4, 100)` array per trial.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from `trials.probabilityLeft`.

ii.
```python
prior = tr.probabilityLeft.to_numpy(float)
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
...
out[1] = prior_map[float(prior[raw_i])]
```

iii. The notes state that the biased-choice task uses exactly these three prior values and that the decoder task requires the `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` recoding.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code filters to the three allowed raw values, maps them to `0/1/2`, and repeats the category across all 100 bins of each retained trial.

ii.
```python
valid &= np.isin(prior, [0.2, 0.5, 0.8])
...
out[1] = prior_map[float(prior[raw_i])]
```

iii. Step 5 of the notes says the repetition across time is intentional so all requested outputs share the same temporal shape.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel output comes from the wheel stream loaded by `SessionLoader.load_wheel()`. The script uses the loader-provided `velocity` and takes its absolute value to obtain speed.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
w = sl.wheel
speed = interpolate_trials(w['times'].to_numpy(), np.abs(w['velocity'].to_numpy()), stim)
```

iii. The notes say this matches the reference path: wheel position/timestamps are converted by `SessionLoader` into velocity, and decoder output uses absolute speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. After `SessionLoader` constructs the wheel trace, the script linearly interpolates absolute velocity onto the 100 common trial bin centers. It does not smooth or normalize the interpolated trace itself. That continuous trace is then discretized session-wise.

ii.
```python
def interpolate_trials(times, values, stim, centers_rel=CENTERS_REL):
    ...
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
    flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
    return flat.reshape(q.shape).astype(np.float32)
```

```python
speed = interpolate_trials(w['times'].to_numpy(), np.abs(w['velocity'].to_numpy()), stim)
```

iii. The notes justify this as using the reference-processed wheel velocity from brainbox and then putting it on the same decoder time grid as the spikes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is converted into three classes by taking the 1/3 and 2/3 quantiles of all finite wheel samples from retained trials in that session, then applying `np.digitize` to assign class `0`, `1`, or `2`.

ii.
```python
def quantile_classes(x: np.ndarray, valid_trials: np.ndarray):
    vals = x[valid_trials]
    vals = vals[np.isfinite(vals)]
    q = np.quantile(vals, [1/3, 2/3]).astype(float)
    if q[1] <= q[0]:
        q[1] = np.nextafter(q[0], np.inf)
    cls = np.digitize(x, q, right=False).astype(np.uint8)
    return cls, q.tolist()
```

```python
speed_cls, speed_q = quantile_classes(speed, valid)
...
out[2] = speed_cls[raw_i]
```

iii. The notes explicitly describe this as per-session tertiles fit on retained finite samples so classes stay balanced despite scale differences.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is evaluated at the same stimulus-relative 100 bin centers used for the neural matrices, so the saved wheel class time series is aligned bin-for-bin with neural activity.

ii.
```python
q = np.asarray(stim)[:, None] + centers_rel[None, :]
...
speed = interpolate_trials(w['times'].to_numpy(), np.abs(w['velocity'].to_numpy()), stim)
...
out[2] = speed_cls[raw_i]
```

iii. The notes say all time-varying streams should share one common stimulus-aligned 20 ms grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the whisker motion-energy camera stream loaded by `SessionLoader.load_motion_energy`. The script tries the left camera first and falls back to the right camera if needed, using that side's `times` and `whiskerMotionEnergy`.

ii.
```python
for side in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[side])
        key = f'{side}Camera'
        me = sl.motion_energy[key]
        vals = me['whiskerMotionEnergy'].to_numpy()
        whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
        if np.isfinite(whisk).any():
            return speed, whisk, side, errors
```

iii. Step 5 in the notes says to prefer left and otherwise use right, matching the AI's reading of the reference helper logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is linearly interpolated onto the 100 common trial bin centers and then discretized session-wise into three classes. The script does not apply additional smoothing or normalization.

ii.
```python
whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
...
whisk_cls, whisk_q = quantile_classes(whisk, valid)
...
out[3] = whisk_cls[raw_i]
```

iii. The notes justify this as using the released whisker motion-energy signal directly and only resampling it onto the decoder grid before categorization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same rule as wheel speed: take the 1/3 and 2/3 quantiles of all finite whisker samples from retained trials in that session and apply `np.digitize` to obtain classes `0`, `1`, and `2`.

ii.
```python
whisk_cls, whisk_q = quantile_classes(whisk, valid)
...
out[3] = whisk_cls[raw_i]
```

iii. The notes explicitly say wheel and whisker should both be converted to three balanced per-session classes by tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is interpolated at the same stimulus-relative 100 bin centers as the wheel and neural data, so the saved whisker classes are aligned bin-for-bin with the neural matrices.

ii.
```python
q = np.asarray(stim)[:, None] + centers_rel[None, :]
flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
...
out[3] = whisk_cls[raw_i]
```

iii. The AI's notes describe one common stimulus-aligned 20 ms grid for all time-varying streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script drops or skips incomplete data rather than imputing it. Trials with missing required events or non-finite interpolated behavior bins are excluded. If the left camera fails it tries the right camera. If a probe fails to load or has no surviving units it is skipped, but the session can survive if another probe works. Sessions are excluded when fewer than two valid trials remain or no probe yields any valid units. For session provenance only, subject/lab/date can fall back from ONE cache metadata to the freeze CSV.

ii.
```python
good = np.isfinite(times) & np.isfinite(values)
...
if times.size < 2:
    return np.full((len(stim), len(centers_rel)), np.nan, dtype=np.float32)
```

```python
for side in ('left', 'right'):
    try:
        ...
    except Exception as ex:
        errors.append(f'{side}:{type(ex).__name__}:{ex}')
raise RuntimeError('no usable whisker motion energy; ' + '; '.join(errors))
```

```python
except Exception as ex:
    failures.append({'pid': str(row.pid), 'probe': row.probe_name,
                     'error': f'{type(ex).__name__}: {ex}'})
...
if not mats:
    raise RuntimeError('no probe yielded qualified units')
```

iii. The notes say missing scientific data should lead to trial, probe, or session exclusion with explicit logging, while metadata-only fallbacks are acceptable if scientific arrays still came from ONE.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading spike-sorting data for each probe and binning those spikes trial-by-trial. The notes also emphasize that some probes contain tens of millions of spikes, so neural loading dominates runtime and memory.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
```

```python
for ti, st in enumerate(stim):
    lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
    ...
    cnt = np.bincount(flat, minlength=len(good_ids) * 100).reshape(len(good_ids), 100)
```

iii. Step 6 and Step 7 of the notes explicitly call out the size of the raw spike arrays and describe searchsorted plus `bincount` binning as the main performance bottleneck that needed optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious remaining loop is the per-trial spike-binning loop in `bin_probe`. The short Python loop in `trial_number_in_block` could also be vectorized, although it is much smaller. Behavior interpolation was already vectorized across all trial/bin query points, so that part is no longer a per-trial Python loop.

ii.
```python
def trial_number_in_block(prior: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
    return out
```

```python
for ti, st in enumerate(stim):
    lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
    ...
```

iii. In Step 6, the AI says it deliberately vectorized behavior interpolation and replaced slower generic neural routines with `searchsorted` plus `bincount`, leaving only the trial loop in neural binning as the main remaining non-vectorized section.

## 10-c. What processing does the code repeat multiple times?

i. There is not much repeated scientific preprocessing, but some repeated administrative work remains. `load_behavior` may try to load left-camera motion energy and then right-camera motion energy for the same session. Session summary statistics such as class counts are recomputed from the already-built `outputs` list. Each worker also re-runs `init_one()` when it starts.

ii.
```python
for side in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[side])
        ...
```

```python
'choice_counts': dict(Counter(int(outputs[k][0,0]) for k in range(len(outputs)))),
'prior_counts': dict(Counter(int(outputs[k][1,0]) for k in range(len(outputs)))),
'wheel_class_counts': np.bincount(np.concatenate([x[2] for x in outputs]), minlength=3).tolist(),
'whisker_class_counts': np.bincount(np.concatenate([x[3] for x in outputs]), minlength=3).tolist(),
```

```python
def _worker_init():
    global _WORKER_ONE
    _WORKER_ONE = init_one()
```

iii. The notes mainly discuss speedups rather than repeated work, so this answer is drawn mostly from the code structure itself. The AI did not present these repeats as scientifically necessary.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several extra computations are for diagnostics or bookkeeping rather than the decoder payload itself. The code checks `feedbackType` and `feedback_times` even though they are not saved as decoder variables. It returns continuous `speed[idx]` and `whisk[idx]` from `process_session` only for optional plotting. It also builds detailed per-session metadata summaries and optional processing plots that are not used by `train_decoder.py`.

ii.
```python
required = ['choice','probabilityLeft','feedbackType','feedback_times','stimOn_times',
            'firstMovement_times','intervals_0','intervals_1']
```

```python
return neural, inputs, outputs, regions, info, speed[idx], whisk[idx]
```

```python
if args.show_processing and len(infos)<=2:
    processing_plot(eid,n,y,speed,whisk)
```

```python
'wheel_tertiles': speed_q, 'whisker_tertiles': whisk_q, 'probes': probes,
'probe_failures': probe_failures, 'behavior_load_notes': behavior_errors,
...
'session_info':infos,'excluded_sessions':excluded,'candidate_sessions':len(eids),
```

iii. The notes justify these extras as validation, provenance, and sanity-check machinery rather than part of the model-ready data representation.
