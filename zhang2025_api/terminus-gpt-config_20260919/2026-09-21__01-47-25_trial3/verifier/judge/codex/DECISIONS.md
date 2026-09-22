# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent creates two local `ONE` clients, one for the base Brainwidemap tables and one for the 2025 update. It derives candidate EIDs by intersecting cached dataset-index entries for spikes, wheel, and a paired left/right camera stream. It then processes all 445 candidates (eight worker processes in full mode). Trial tables, wheel, camera arrays, and spike sorting are loaded through ONE/brainbox, although the code inspects cache paths and mutates ONE's in-memory revision metadata to resolve the trial table.

ii.
```python
def make_ones():
    return (ONE(cache_dir=CACHE, tables_dir=BASE_TABLES, mode='local'),
            ONE(cache_dir=CACHE, tables_dir=UPDATE_TABLES, mode='local'))

core = es(bd, bp, 'spikes.times') & es(bd, bp, '_ibl_wheel.timestamps.npy')
left = es(ud, up, 'leftCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_leftCamera.times.npy')
right = es(ud, up, 'rightCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_rightCamera.times.npy')
return sorted(core & (left | right), key=str), left, right
```

iii. The notes justify this as compliance with the ONE-only requirement and as a way to reconcile stale/stripped revision records. They report 445 retained sessions from the 459-session ephys release, with 14 lacking a paired whisker stream.

## 1-b. How are the data split into subjects?

i. Each processed session obtains its subject from `base.eid2ref(eid)`. At assembly, unique subject strings are sorted and each session receives an integer index into that vocabulary.

ii.
```python
ref = base.eid2ref(eid)
out = dict(eid=str(eid), subject=str(ref.subject), ...)
subjects = sorted({s['subject'] for s in sessions})
smap = {x:i for i,x in enumerate(subjects)}
'subject_idx': np.array([smap[s['subject']] for s in sessions], dtype=np.int32)
```

iii. The agent notes that ONE already exposes the unique subject identity, avoiding path parsing. The full result contains 136 subjects.

## 1-c. How are the data split into sessions?

i. ONE EIDs define sessions. `process_session` is called once per EID; results are sorted by EID and appended as one element per session in each target list.

ii.
```python
futs = {ex.submit(_process_worker, str(eid)):str(eid) for eid in eids}
...
sessions.sort(key=lambda x:x['eid'])
for s in sessions:
    ...
    neural.append(ns); inputs.append(ins); outputs.append(outs)
```

iii. The notes treat the native EID as the session boundary and parallelize independent sessions for speed.

## 1-d. How are the data split into trials?

i. The aggregate trials table supplies one row per trial. Retained row indices and `stimOn_times` form the trial list; neural and behavioral arrays are indexed by that same retained-trial order and finally converted to nested per-trial lists.

ii.
```python
original_idx = np.flatnonzero(mask)
align = trials.stimOn_times.to_numpy(float)[mask]
...
for j in range(s['n_trials_valid']):
    ns.append(s['neural'][j])
```

iii. The agent states that the native table already has one row per trial and emphasizes synchronous filtering so all streams remain matched.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have finite choice, prior, feedback type/time, stimulus onset, and first movement; choice must be ±1, prior one of 0.2/0.5/0.8, and stimulus-to-first-movement time 0.08–2 s. Trials are then removed if either interpolated behavioral stream has any NaN, and later if all curated neural counts are zero. Sessions with fewer than two surviving trials are rejected.

ii.
```python
required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
            'stimOn_times', 'firstMovement_times']
for c in required:
    m &= np.isfinite(trials[c].to_numpy(dtype=float))
m &= np.isin(choice, [-1., 1.])
m &= np.isin(prior, [.2, .5, .8])
m &= (rt >= .08) & (rt <= 2.0)
...
complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
neural_valid = np.any(neural != 0, axis=(1, 2))
```

iii. The 80 ms–2 s and no-choice rules are attributed to the paper/reference. Complete behavior is required to avoid extrapolation, and 39 all-zero neural trials were removed because the supplied validator warned about them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from per-probe `spikes.times` and `spikes.clusters`. Merged cluster/channel metadata provide QC labels, anatomical acronyms, IDs, and UUIDs.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
merged = ssl.merge_clusters(spikes, clusters, channels).to_df()
binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                              spike_clusters[selected_spikes], ids, align, ...)
```

iii. The notes identify these as the native Neuropixels spike time/assignment arrays and explain that this is electrophysiology, so delta-F/F is inapplicable.

## 2-b. How is the `neural` data processed?

i. For each probe, selected spikes are binned with `bin_spikes2D` into 100 20-ms bins from −0.5 to +1.5 s around each stimulus. Probe arrays are concatenated along the neuron axis and stored as float32 **spike counts**; they are not divided by bin width, smoothed, square-root transformed, or standardized.

ii.
```python
binned, tscale = bin_spikes2D(..., align,
                              pre_time=.5, post_time=1.5, bin_size=DT)
arrays.append(np.asarray(binned))
x = np.concatenate(arrays, axis=1)
return x.astype(np.float32), regions, uuids, spike_total
```

iii. The agent says raw counts match the reference binning stage and preserve exact observations; float32 was chosen to satisfy the validator. It explicitly declined optional standardization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters must have `label >= 1`. Allen acronyms are mapped to Beryl, and labels in `{'void','root','nan','None',''}` are excluded. Spikes are retained only if their cluster ID is selected; sessions with no curated neurons are rejected.

ii.
```python
acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
good = (label >= 1) & ~np.isin(acr, list(BAD_REGIONS))
ids = merged.index.to_numpy()[good]
selected_spikes = np.isin(spike_clusters, ids)
```

iii. The notes justify `label >= 1` from the data paper's 75,708 well-isolated neurons and exclude `root`/`void` as non-gray or invalid anatomical assignments.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`; brainbox bins spikes from 0.5 s before to 1.5 s after each onset on the shared session clock.

ii.
```python
align = trials.stimOn_times.to_numpy(float)[mask]
bin_spikes2D(..., ids, align, pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The agent cites the reference stimulus-aligned interval and shared IBL clock, and independently checked one converted trial against source spikes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms: 100 bins spanning −0.5 to +1.5 s. Spikes are directly counted at that resolution, with no later rebinning.

ii.
```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
```

iii. The notes cite the reference code and method paper's 20-ms, 100-step representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured window and bin width relative to each trial's `stimOn_times`, rather than from a separate sampled raw stream.

ii.
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
align = trials.stimOn_times.to_numpy(float)[mask]
```

iii. The agent says stimulus onset is the required alignment event and uses the reference window/bin configuration.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code calculates bin centers −0.49, −0.47, …, 1.49 seconds and repeats that row for every retained trial.

ii.
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS, s['trial_number'][j], np.float32))))
```

iii. Bin centers were chosen to represent the same instants used for continuous behavior sampling and spike bins.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the center coordinate of each neural bin. Both have 100 positions relative to the same `stimOn_times` event.

ii.
```python
bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
np.vstack((BIN_CENTERS, ...))
```

iii. The notes report direct checks of common bin centers and stimulus alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full unfiltered `trials.probabilityLeft` sequence; a change in prior starts a new block.

ii.
```python
block_num = trial_number_in_block(trials.probabilityLeft.to_numpy(float))
```

iii. The agent explains that `probabilityLeft` is constant within a block and that counting before filtering preserves the animal's true position.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter resets to zero when the prior differs from the preceding trial and increments otherwise. It is filtered after computation and broadcast across all 100 time bins.

ii.
```python
for i in range(1, len(prior)):
    out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
...
np.full(N_BINS, s['trial_number'][j], np.float32)
```

iii. This implements a continuous per-trial decoder input while retaining gaps caused by later QC filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials-table `choice` column.

ii.
```python
choice_raw = trials.choice.to_numpy(float)[mask]
choice = (choice_raw == 1).astype(np.uint8)
```

iii. The notes identify IBL `+1` as left and `-1` as right; no-choice zero trials are excluded.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent's documented decision is to map retained `+1` to 0 (left) and `-1` to 1 (right), then broadcast the scalar across time. However, the implementation uses `(choice_raw == 1)`, which actually maps `+1` to 1 and `-1` to 0—the reverse of that stated decision.

ii.
```python
choice = (choice_raw == 1).astype(np.uint8)
np.full(N_BINS, s['choice'][j], np.uint8)
```

iii. The notes claim this follows the task's required left=0/right=1 mapping and report raw-to-converted checks, but that justification conflicts with the shown Boolean expression.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
prior_raw = trials.probabilityLeft.to_numpy(float)[mask]
```

iii. The notes identify it as the block prior required by the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to classes 0, 1, and 2 and broadcast across time.

ii.
```python
prior = np.select([prior_raw == .2, prior_raw == .5, prior_raw == .8],
                  [0, 1, 2]).astype(np.uint8)
```

iii. This is the exact mapping specified by the task; the 0.5 class is deliberately retained.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from the wheel object's raw `timestamps` and `position` arrays.

ii.
```python
w = base.load_object(eid, 'wheel', collection='alf')
ts = np.asarray(w['timestamps'], float)
pos = np.asarray(w['position'], float)
```

iii. The notes follow the reference/brainbox wheel loader and distinguish speed magnitude from signed velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, brainbox computes a filtered derivative, its absolute value is taken, and linear interpolation samples speed at each trial's 20-ms bin centers. Nonfinite/duplicate stream samples are removed before interpolation.

ii.
```python
ipos, its = wheellib.interpolate_position(ts, pos, freq=1000)
vel, _ = wheellib.velocity_filtered(ipos, fs=1000)
return interp_trials(its, np.abs(vel), align), ...
```

iii. The agent says this matches recommended brainbox/reference processing. Although an earlier planning table said “average within” bins, the final code, metadata, and README describe sampling at bin centers.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are computed from the 1/3 and 2/3 quantiles of **all retained wheel samples across all sessions**. `np.digitize` maps each value to low/medium/high (0/1/2).

ii.
```python
wheel_vals = np.concatenate([s['wheel'].ravel() for s in sessions])
thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float), ...}
wc = np.digitize(s['wheel'], thresholds['wheel']).astype(np.uint8)
```

iii. The agent chose global tertiles to make class semantics comparable across sessions and guarantee approximately balanced classes globally; thresholds are saved in metadata.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Absolute velocity is linearly interpolated at `stimOn_times + BIN_CENTERS`, so every behavioral value corresponds to a neural bin center.

ii.
```python
query = align[:, None] + BIN_CENTERS[None, :]
out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The shared session clock and identical center grid are cited as sufficient alignment; extrapolation is forbidden.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It comes from left-camera `ROIMotionEnergy` and matching `_ibl_leftCamera.times`, falling back to the analogous right-camera pair.

ii.
```python
for camera, available in [('left', left_set), ('right', right_set)]:
    me = load_motion_energy(update, eid, camera)
    times = base.load_dataset(eid, f'_ibl_{camera}Camera.times.npy', collection='alf')
```

iii. The left preference and right fallback are justified as matching the reference behavior-loading logic while maximizing usable sessions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy values are used without filtering or normalization. After finite-value cleanup, timestamp sorting, and duplicate removal, they are linearly interpolated at neural-bin centers.

ii.
```python
return interp_trials(times, me, align), camera, (times, me)
...
out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The agent says the released ROI motion energy needs no additional processing. Final documentation describes center sampling despite an earlier plan mentioning within-bin averages.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global 1/3 and 2/3 quantiles are computed over every retained whisker sample from every session, then `np.digitize` produces classes 0/1/2.

ii.
```python
whisk_vals = np.concatenate([s['whisker'].ravel() for s in sessions])
'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float)
mc = np.digitize(s['whisker'], thresholds['whisker']).astype(np.uint8)
```

iii. As for wheel, global thresholds were chosen for common class meanings and global balance and are stored in metadata.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated at `stimOn_times + BIN_CENTERS`, exactly the center grid used to label neural bins.

ii.
```python
query = align[:, None] + BIN_CENTERS[None, :]
out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
```

iii. The agent relies on synchronized IBL timestamps and rejects trials requiring extrapolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing required columns, ambiguous revision records, missing probe collections, zero curated neurons, and mismatched camera lengths raise session-level errors; full conversion logs and skips failed sessions. The camera loader falls back left-to-right. Nonfinite stream samples and duplicate timestamps are cleaned, but trials with incomplete interpolated traces are dropped. Sessions with fewer than two trials and all-zero-neural trials are excluded. In the final run, all 445 candidates succeeded.

ii.
```python
except Exception as exc:
    failures.append((eid,type(exc).__name__,str(exc)))
    print(f'SKIP {eid} {type(exc).__name__}: {exc}',flush=True)
...
ok = np.isfinite(times) & np.isfinite(values)
keep = np.r_[True, np.diff(times) > 0]
```

iii. The notes stress no silent session loss, no extrapolation, synchronous removal across streams, and validation after fixes. Camera fallback and cache-revision reconciliation are documented as robustness measures.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/merging large spike-sorting arrays and binning spikes per session dominate. Full conversion is parallelized across up to eight processes and took about 371 seconds; serialization of the 11.28-GB pickle is also material.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
binned, tscale = bin_spikes2D(...)
with ProcessPoolExecutor(max_workers=workers) as ex:
```

iii. The agent's runtime notes estimate session conversion from sample runs and use process-level parallelism because sessions are independent.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit loop in `trial_number_in_block` could be replaced by vectorized change detection and grouped cumulative counts. The final per-session/per-trial construction loop could be replaced partly by array broadcasting, though nested lists are ultimately required. Probe and session loops reflect heterogeneous data and are less naturally vectorized.

ii.
```python
for i in range(1, len(prior)):
    out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
...
for j in range(s['n_trials_valid']):
    ns.append(s['neural'][j])
```

iii. The agent did not explicitly discuss these loops in its notes; it instead vectorized interpolation queries and delegated spike binning to brainbox.

## 10-c. What processing does the code repeat multiple times?

i. Every worker recreates both ONE clients and recomputes `candidate_eids`; `BrainRegions()` is instantiated within each probe iteration; and constant bin-center/scalar rows are materialized once per trial during output assembly.

ii.
```python
def _process_worker(eid_str):
    base, update = make_ones()
    eids, left, right = candidate_eids(base, update)
...
acr = BrainRegions().acronym2acronym(...)
```

iii. The worker-local ONE clients are intentional because clients are not pickled/shared safely. The agent did not document the other repetition as a performance issue.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion, `load_wheel_speed` and `load_whisker` return raw diagnostic tuples even when `show=False`; `process_session` then discards them. `bin_spikes2D` returns `tscale`, which is unused. The code counts all loaded spikes only for logs/metadata and keeps UUID/trial-index metadata not used by the supplied trainer. Each worker also computes the entire candidate list only to process one EID.

ii.
```python
binned, tscale = bin_spikes2D(...)
return interp_trials(its, np.abs(vel), align), (ts, pos, its, vel)
...
return process_session(base, update, base.to_eid(eid_str), left, right, False)[0]
```

iii. Raw tuples support optional plots and metadata supports reproducibility, so they are purposeful diagnostics, but the normal full path does not consume the raw tuples or `tscale`. The notes do not identify these costs explicitly.
