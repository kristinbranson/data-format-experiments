# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent creates two local `ONE` clients, one over the `Brainwidemap` table and one over the `2025_Q3_IBL_et_al_BWM` update table. It discovers candidate sessions by scanning ONE cache metadata for required datasets, then loads trials through a custom `load_trial_table()` helper, wheel through `ONE.load_object()`, whisker motion energy through `ONE.load_dataset()`, and spikes through `SpikeSortingLoader`.

ii. ```python
def make_ones():
    return (ONE(cache_dir=CACHE, tables_dir=BASE_TABLES, mode='local'),
            ONE(cache_dir=CACHE, tables_dir=UPDATE_TABLES, mode='local'))
```

```python
core = es(bd, bp, 'spikes.times') & es(bd, bp, '_ibl_wheel.timestamps.npy')
left = es(ud, up, 'leftCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_leftCamera.times.npy')
right = es(ud, up, 'rightCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_rightCamera.times.npy')
```

```python
trials = load_trial_table(base, eid)
wheel, wheel_raw = load_wheel_speed(base, eid, align)
whisk, camera, whisk_raw = load_whisker(base, update, eid, align, left_set, right_set)
neural, regions, uuids, nspikes = load_binned_neural(base, eid, align)
```

iii. In `CONVERSION_NOTES.md`, the agent says it intentionally kept all scientific reads inside ONE/brainbox, but added custom revision handling because the cached release metadata sometimes omitted revision components for trial and motion-energy datasets.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are recovered from the session reference returned by ONE. After all sessions are processed, the code builds a sorted unique subject list and a `subject_idx` array mapping each session to its subject.

ii. ```python
ref = base.eid2ref(eid)
out = dict(eid=str(eid), subject=str(ref.subject), neural=neural, ...)
```

```python
subjects = sorted({s['subject'] for s in sessions}); smap={x:i for i,x in enumerate(subjects)}
...
'subject_idx':np.array([smap[s['subject']] for s in sessions],dtype=np.int32),
```

iii. The notes say subject identity should come from the ONE session metadata/API rather than from path parsing.

## 1-c. How are the data split into sessions?

i. Each `eid` is treated as one session. `candidate_eids()` returns the eligible session IDs, and `process_session()` handles one `eid` at a time.

ii. ```python
return sorted(core & (left | right), key=str), left, right
```

```python
def process_session(base, update, eid, left_set, right_set, show=False):
```

iii. The notes describe the source cohort as the 459-session public BWM release and state that a session remains usable only if the required trial, wheel, spike, and whisker datasets exist.

## 1-d. How are the data split into trials?

i. Trials come directly from the IBL trial table: the code loads the full `_ibl_trials.table.pqt`, then each table row is one trial. Trial-level arrays are later built by masking rows and iterating over retained trial indices.

ii. ```python
return one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf',
                        revision=rev, check_hash=False)
```

```python
mask = trial_mask(trials)
original_idx = np.flatnonzero(mask)
align = trials.stimOn_times.to_numpy(float)[mask]
```

iii. The notes treat the aggregate trials parquet as the authoritative per-trial source.

## 1-e. How are trials filtered based on quality controls?

i. The code keeps only trials with finite `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, and `firstMovement_times`; choice must be `-1` or `+1`; prior must be `0.2`, `0.5`, or `0.8`; reaction time must be 0.08 to 2.0 s. After the wheel and whisker traces are built, trials whose 100-bin behavior windows contain any `NaN` are dropped. After neural binning, trials with zero spikes across all curated neurons are dropped too.

ii. ```python
required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
            'stimOn_times', 'firstMovement_times']
...
m &= np.isin(choice, [-1., 1.])
m &= np.isin(prior, [.2, .5, .8])
m &= (rt >= .08) & (rt <= 2.0)
```

```python
complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
...
neural_valid = np.any(neural != 0, axis=(1, 2))
```

iii. The notes justify the finite-field and RT filters from the data paper, and justify complete behavior coverage because wheel and whisker are required decoder outputs. They also note that zero-neural trials were removed because the supplied validator rejects trials with no neural observation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is derived from spike times and spike cluster assignments per probe. Cluster metadata are used for curation and region labels, but the actual binned neural matrix is built from `spikes['times']` and `spikes['clusters']`.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
merged = ssl.merge_clusters(spikes, clusters, channels).to_df()
```

```python
spike_clusters = np.asarray(spikes['clusters'])
binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                              spike_clusters[selected_spikes], ids, align,
                              pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The notes map `spikes.times` and `spikes.clusters` to `neural`, with cluster `label` and `acronym` only supporting filtering and annotation.

## 2-b. How is the `neural` data processed?

i. For each probe, the code bins curated spikes into 100 stimulus-aligned 20 ms bins using `bin_spikes2D`, concatenates the probe-wise trial-by-cluster arrays along the neuron axis, and stores the result as `float32`. It keeps spike counts, not firing rates.

ii. ```python
binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                              spike_clusters[selected_spikes], ids, align,
                              pre_time=.5, post_time=1.5, bin_size=DT)
...
x = np.concatenate(arrays, axis=1)
...
return x.astype(np.float32), regions, uuids, spike_total
```

iii. The notes explicitly say it would preserve counts rather than convert to rates or z-scores, arguing that “no information is lost.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only clusters with `label >= 1` after merging cluster metrics/histology, remaps acronyms to Beryl, and then excludes acronyms in `{'void', 'root', 'nan', 'None', ''}`. Only spikes belonging to the surviving cluster IDs are binned.

ii. ```python
BAD_REGIONS = {'void', 'root', 'nan', 'None', ''}
```

```python
acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
good = (label >= 1) & ~np.isin(acr, list(BAD_REGIONS))
ids = merged.index.to_numpy()[good]
selected_spikes = np.isin(spike_clusters, ids)
```

iii. The notes justify `label >= 1` from the data paper’s “well-isolated neurons” criterion, and justify removing `root`/`void` as a gray-matter restriction after Beryl remapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. The code passes the retained stimulus-onset times as the `align` argument to `bin_spikes2D`, with `pre_time=.5` and `post_time=1.5`, so bin 0 spans 500 ms before stimulus onset and the last bin ends 1.5 s after.

ii. ```python
align = trials.stimOn_times.to_numpy(float)[mask]
...
binned, tscale = bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The notes say the task requirement overrides the method paper’s target-specific alignments, so everything is put on a common stimulus-aligned window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins across a fixed 2 s window, giving 100 bins per trial. No additional resampling is applied to the neural signal beyond this binning.

ii. ```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
```

```python
binned, tscale = bin_spikes2D(..., pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The notes tie this directly to the Zhang stimulus-aligned cache format: 100 bins of 20 ms from -0.5 to +1.5 s.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `stimOn_times`, plus the fixed decoder bin grid. The stored values themselves are the fixed bin centers, not raw timestamps copied from file.

ii. ```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
...
align = trials.stimOn_times.to_numpy(float)[mask]
```

iii. The notes describe this input as the common stimulus-aligned time base used for every retained trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code computes a fixed `float32` vector of 100 bin centers from -0.49 to 1.49 s and reuses that same vector for every trial.

ii. ```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
...
ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
```

iii. The notes say this is simply the shared bin-center grid implied by the common stimulus-aligned representation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is exactly the same 100-bin grid used for neural binning. The neural spikes are binned relative to `align`, and the first decoder input row is `BIN_CENTERS`, so the two share a bin-by-bin time axis.

ii. ```python
binned, tscale = bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
```

```python
ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
```

iii. The notes repeatedly describe a single common time base for all trial-aligned streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` sequence. A change in `probabilityLeft` starts a new block.

ii. ```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
```

iii. The notes say the trial table does not provide a separate block ID, so blocks must be reconstructed from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code walks through the full prior sequence before trial filtering, resets the counter to 0 whenever the prior changes, increments within a block, then masks that per-trial vector and broadcasts the retained scalar across all 100 bins of each trial.

ii. ```python
block_num = trial_number_in_block(trials.probabilityLeft.to_numpy(float))
mask = trial_mask(trials)
...
block_num = block_num[mask]
```

```python
ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
```

iii. The notes explicitly justify computing this before filtering so removed trials still advance the experimental block count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the raw `trials.choice` column.

ii. ```python
choice_raw = trials.choice.to_numpy(float)[mask]
```

iii. The notes identify `choice` as the sole raw source for this output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering to valid choice values, the code maps `choice_raw == 1` to class 1 and everything else retained (`-1`) to class 0, then broadcasts the per-trial class across the 100 bins.

ii. ```python
choice = (choice_raw == 1).astype(np.uint8)
```

```python
outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8),
                       np.full(N_BINS,s['prior'][j],np.uint8),wc[j],mc[j])))
```

iii. The notes justify this by stating that the task semantics should be interpreted as raw `choice=-1` meaning left and raw `choice=+1` meaning right.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. This output is derived from `trials.probabilityLeft`.

ii. ```python
prior_raw = trials.probabilityLeft.to_numpy(float)[mask]
```

iii. The notes refer to `probabilityLeft` as the block prior variable to be decoded.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code remaps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts that per-trial class across all 100 bins.

ii. ```python
prior = np.select([prior_raw == .2, prior_raw == .5, prior_raw == .8], [0, 1, 2]).astype(np.uint8)
```

```python
outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8),
                       np.full(N_BINS,s['prior'][j],np.uint8),wc[j],mc[j])))
```

iii. The notes say this mapping follows the task’s required output coding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the raw wheel timestamps and wheel position arrays loaded from the `wheel` object.

ii. ```python
w = base.load_object(eid, 'wheel', collection='alf')
ts = np.asarray(w['timestamps'], float)
pos = np.asarray(w['position'], float)
```

iii. The notes map wheel `timestamps` and `position` to the wheel-speed output.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code interpolates wheel position to 1 kHz with `brainbox.behavior.wheel.interpolate_position`, computes filtered velocity with `velocity_filtered`, takes the absolute value to get speed, then linearly interpolates the speed trace to the 100 stimulus-aligned bin centers for each trial.

ii. ```python
ipos, its = wheellib.interpolate_position(ts, pos, freq=1000)
vel, _ = wheellib.velocity_filtered(ipos, fs=1000)
return interp_trials(its, np.abs(vel), align), (ts, pos, its, vel)
```

```python
query = align[:, None] + BIN_CENTERS[None, :]
out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The notes justify the first part as the reference brainbox wheel-processing workflow. The notes are inconsistent about the last step: earlier they say values would be averaged within 20 ms bins, but the final code actually samples by interpolation at bin centers.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are processed, the code pools every retained wheel-speed bin from every session, computes global 1/3 and 2/3 quantiles, and discretizes each wheel bin with `np.digitize()` into classes 0, 1, and 2.

ii. ```python
wheel_vals = np.concatenate([s['wheel'].ravel() for s in sessions])
thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float),
              'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float)}
```

```python
wc=np.digitize(s['wheel'],thresholds['wheel']).astype(np.uint8)
```

iii. The notes justify global tertiles by saying fixed thresholds make wheel classes comparable across sessions, and store the thresholds in metadata.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is evaluated on the same 100 bin centers relative to each trial’s `stimOn_times`, so it is bin-aligned to the neural data.

ii. ```python
query = align[:, None] + BIN_CENTERS[None, :]
out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

```python
binned, tscale = bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The notes describe a shared stimulus-aligned 20 ms grid for neural, wheel, whisker, and the time input.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The whisker output is derived from `<side>Camera.ROIMotionEnergy.npy` and the matching `_ibl_<side>Camera.times.npy`. The code prefers the left camera and falls back to the right camera.

ii. ```python
for camera, available in [('left', left_set), ('right', right_set)]:
    if uid not in available:
        continue
    ...
    me = load_motion_energy(update, eid, camera)
    times = base.load_dataset(eid, f'_ibl_{camera}Camera.times.npy', collection='alf')
```

iii. The notes say this matches the reference preference for left camera with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code uses the released motion-energy trace directly, checks that it has the same length as the camera timestamps, and linearly interpolates the trace to the 100 stimulus-aligned bin centers for each trial. No additional filtering or normalization is applied.

ii. ```python
if len(me) != len(times):
    raise RuntimeError(f'{camera} motion/timestamp length mismatch {len(me)} != {len(times)}')
return interp_trials(times, me, align), camera, (times, me)
```

```python
query = align[:, None] + BIN_CENTERS[None, :]
out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The notes justify using the released motion-energy trace as-is. As with wheel, the planning notes mention bin-wise averaging in one place, but the implemented code uses interpolation at bin centers.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The code pools all retained whisker-motion-energy bins across all sessions, computes global 1/3 and 2/3 quantiles, and discretizes each bin with `np.digitize()` into classes 0, 1, and 2.

ii. ```python
whisk_vals = np.concatenate([s['whisker'].ravel() for s in sessions])
thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float),
              'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float)}
```

```python
mc=np.digitize(s['whisker'],thresholds['whisker']).astype(np.uint8)
```

iii. The notes justify global tertiles the same way as wheel: shared thresholds were intended to keep class meanings comparable across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is sampled at the same 100 bin centers relative to `stimOn_times`, so it shares the neural trial grid.

ii. ```python
return interp_trials(times, me, align), camera, (times, me)
```

```python
binned, tscale = bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The notes describe a single stimulus-aligned grid for all time-varying streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code drops invalid data rather than imputing it. Missing required trial fields fail the trial mask; missing or incomplete wheel/whisker coverage produces `NaN` and removes the trial; missing cameras trigger left-to-right fallback and then session failure if neither works; probes with no surviving curated clusters are skipped, and sessions with fewer than two complete valid trials or zero curated neurons are dropped. Trials with no spikes in any curated neuron are also removed.

ii. ```python
if len(times) < 2:
    return np.full((len(align), N_BINS), np.nan, np.float32)
```

```python
for camera, available in [('left', left_set), ('right', right_set)]:
    ...
raise RuntimeError('no loadable paired motion-energy camera; ' + ' | '.join(errors))
```

```python
if not arrays:
    raise RuntimeError('zero curated neurons')
...
if len(align) < 2:
    raise RuntimeError(f'fewer than two complete valid trials ({len(align)})')
```

iii. The notes explicitly say missing or empty behavior bins should invalidate a trial rather than be imputed, and that sessions with fewer than two valid trials or zero curated neurons should be excluded and reported.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading large spike-sorting datasets and binning those spikes into trial-aligned neural tensors. The notes say sessions with tens of millions of spikes dominate runtime, with serial processing around 4.3 s per session before multiprocessing.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
```

```python
binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                              spike_clusters[selected_spikes], ids, align,
                              pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. `CONVERSION_NOTES.md` explicitly identifies spike loading and brainbox binning as the dominant cost and motivates later multiprocessing from that bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already pushed most expensive sample-level work into vectorized NumPy or brainbox helpers. The remaining obvious Python loops are the small sequential loop in `trial_number_in_block()`, the per-session/per-trial assembly loop in `build_output()`, and the repeated worker-side recomputation of candidate-session metadata.

ii. ```python
for i in range(1, len(prior)):
    out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
```

```python
for s in sessions:
    ...
    for j in range(s['n_trials_valid']):
        ns.append(s['neural'][j])
        ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS,s['trial_number'][j],np.float32))))
        outs.append(np.vstack((np.full(N_BINS,s['choice'][j],np.uint8),
                               np.full(N_BINS,s['prior'][j],np.uint8),wc[j],mc[j])))
```

iii. The notes claim “vectorized masks/interpolation/binning” as a speedup, so the agent’s main efficiency decision was to vectorize the heavy numerical work and leave smaller bookkeeping loops in Python.

## 10-c. What processing does the code repeat multiple times?

i. Each worker process rebuilds the ONE clients and reruns `candidate_eids()` even though the main process has already computed the candidate session list. The code also re-instantiates `BrainRegions()` inside probe processing and repeatedly allocates broadcast arrays trial by trial in `build_output()`.

ii. ```python
def _process_worker(eid_str):
    base, update = make_ones()
    eids, left, right = candidate_eids(base, update)
    return process_session(base, update, base.to_eid(eid_str), left, right, False)[0]
```

```python
acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
```

iii. The notes do not call this out directly, but the code structure shows duplicated session-discovery work per worker and repeated per-trial allocations during output assembly.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion carries several extra bookkeeping values that the decoder does not use: per-neuron UUIDs, retained original trial indices, per-session spike totals, failure summaries, and optional raw wheel/whisker/trial objects for plotting. It also spends work checking for zero-neural trials solely to satisfy the supplied validator rather than the scientific conversion itself.

ii. ```python
return x.astype(np.float32), regions, uuids, spike_total
```

```python
out = dict(eid=str(eid), subject=str(ref.subject), neural=neural,
           regions=regions, uuids=uuids, choice=choice, prior=prior,
           trial_number=block_num.astype(np.float32), wheel=wheel, whisker=whisk,
           trial_indices=original_idx.astype(np.int32), camera=camera,
           n_trials_raw=len(trials), n_trials_valid=len(align), n_zero_neural=n_zero_neural, nspikes=nspikes)
```

```python
if args.show_processing:
    for s in sessions[:2]:
        if s['eid'] in raws: make_plot(...)
```

iii. The notes justify some of this as validation and provenance logging, but these pieces are not part of the downstream decoder inputs/outputs themselves.
