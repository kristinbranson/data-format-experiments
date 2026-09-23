# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a local ONE index over the staged cache, finds sessions with trials, wheel, spikes, and a left or right motion-energy stream, then restricts them to sessions from the methods paper's `repro_ephys_release.txt`. It maps published EIDs to locally hashed EIDs using lab, subject, date, and session number. Each selected session is loaded through `ONE.load_object`; spike sorting is loaded probe-by-probe with `SpikeSortingLoader`. The full run converted 22 sessions.

ii.
```python
make_parquet_db(CACHE_ROOT, out_dir=INDEX_DIR, hash_ids=True, hash_files=False)
one = ONE(cache_dir=CACHE_ROOT, mode='local')
one.load_cache(tables_dir=INDEX_DIR)
...
complete = core & (left | right)
...
selected = [local_by_key[key(row)] for _, row in target_rows.iterrows() if key(row) in local_by_key]
...
tr = trial_frame(one.load_object(eid, 'trials'))
wheel = one.load_object(eid, 'wheel')
```

iii. The notes say ONE/brainbox had to be used exclusively and that rebuilding the index handled stale release metadata. Although the planning notes initially identified the broad BWM cohort as appropriate, the final justification says processing all 447 complete sessions would take too long, so the agent applied the methods-paper cohort and retained its 22 locally complete sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from ONE's session table. Converted session subjects are deduplicated and sorted; each session gets an integer index into that list. Because the restricted cohort has one session per subject, the result contains 22 subjects and 22 sessions.

ii.
```python
def session_details(one, eid):
    row = one._cache['sessions'].loc[eid]
    return {k: (str(row[k]) if k in row and pd.notna(row[k]) else None)
            for k in ('subject', 'date', 'number', 'lab')}
...
subject_list=sorted(set(subjects))
subject_idx=np.array([subject_list.index(x) for x in subjects],dtype=np.int64)
```

iii. The agent regarded the ONE session metadata as the authoritative subject identifier and recorded unknown only if that metadata was absent.

## 1-c. How are the data split into sessions?

i. Each ONE EID is treated as one decoder session. Probes within an EID are merged into that session, while the outer `neural`, `input`, and `output` lists receive one entry per successfully processed EID.

ii.
```python
for k,eid in enumerate(eids,1):
    n,i,o,r,info,cont=process_session(one,eid,plot=args.show_processing and k<=2)
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. The notes justify this because probes recorded during the same behavior session are not independent and the paper combines them at session level.

## 1-d. How are the data split into trials?

i. The loaded ALF trials object is normalized to a DataFrame with one row per trial. Surviving row indices are used consistently to select stimulus times and trial-level variables. Per-trial neural/input/output arrays are appended in source order.

ii.
```python
tr = trial_frame(one.load_object(eid, 'trials'))
mask = valid_trials(tr, wt, ct)
idx = np.flatnonzero(mask)
...
for j in range(len(idx)):
    neural.append(neural_arr[j]); inputs.append(inp); outputs.append(out)
```

iii. The agent notes that the trial table already defines trial boundaries and that source indices are retained in metadata for auditability.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have finite choice, prior, feedback type/time, stimulus onset, and first movement; choice must be ±1 and prior near 0.2/0.5/0.8. If interval columns exist, duration must be positive and at most 10 s. The full [-0.5, 1.5] s window must lie inside wheel and camera endpoint support, and interpolated behavior must be finite. Sessions with fewer than two trials are excluded. The code does not apply the reference 80 ms–2 s reaction-time mask.

ii.
```python
required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
            'stimOn_times', 'firstMovement_times']
...
mask &= np.isin(choice, [-1, 1])
mask &= np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(axis=1)
...
mask &= np.isfinite(duration) & (duration <= 10) & (duration > 0)
...
finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
```

iii. The notes say this is the union of paper-required event detection, the methods code's 10 s limit, valid categories, and complete support for requested behavior. They incorrectly characterize it as matching the common/reference mask despite omitting the stated reaction-time bounds.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from `spikes.times` and `spikes.clusters`. Merged cluster metadata supplies `cluster_id`, QC `label`, and `atlas_id` for unit selection and anatomical assignment.

ii.
```python
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
...
all_t.append(np.asarray(spikes['times'])[keep].astype(np.float64))
all_c.append(mapped)
```

iii. The notes identify spike times/assignments as the electrophysiology signal and merged cluster metadata as the appropriate QC/anatomy source.

## 2-b. How is the `neural` data processed?

i. QC-passing units from all probes are compactly renumbered, concatenated, and time-sorted. For each trial, spikes are histogrammed into 100 nonoverlapping 20 ms bins. The stored values are float32 spike counts, not firing rates; unlike the human reference, the agent does not divide counts by 0.02 s.

ii.
```python
flat = clu[ok] * N_TIME + relbin[ok]
out[i] = np.bincount(flat, minlength=n_units * N_TIME).reshape(n_units, N_TIME)
...
neural_representation='QC-filtered spike counts per 20 ms bin'
```

iii. The agent states that the reference caching stage stores spike counts and that standardization happens later, so it deliberately preserves counts rather than standardizing or converting to rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units require merged cluster `label >= 1` and Allen `atlas_id > 0`. Probes with no surviving units are skipped; sessions with no QC-passing units are excluded. Anatomy is remapped to Beryl, but the filter is not the reference's explicit Beryl-acronym `void` exclusion and the output retains 429 `root` units.

ii.
```python
label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
good = (label >= 1) & (atlas > 0)
...
beryl_ids = BR.remap(atlas[good], source_map='Allen', target_map='Beryl')
```

iii. The notes justify `label >= 1` as the paper's fully passing-unit criterion and positive atlas ID as resolved anatomy. They explicitly retain `root`, arguing that it is a valid positive mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is defined from 0.5 s before to 1.5 s after `stimOn_times`. Absolute spike times in that window are converted to bins relative to the window start, so stimulus onset is time zero.

ii.
```python
lo, hi = st + OFF_START, st + OFF_END
a, b = np.searchsorted(spike_t, [lo, hi])
relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
```

iii. The agent chose this because the task explicitly requests stimulus alignment and the methods caching defaults use the identical event and window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms. Event spikes are binned directly into 100 bins over [-0.5, 1.5); no later temporal rebinning or smoothing is performed.

ii.
```python
DT = 0.020
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
N_TIME = len(CENTERS_REL)
```

iii. The notes cite both the paper and methods code as using nonoverlapping 20 ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the common bin grid around trial-table `stimOn_times`, specifically the relative centers of the neural bins.

ii.
```python
stim_all = tr.stimOn_times.to_numpy(float)
...
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

iii. The agent describes stimulus onset as the requested and reference-code alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Fixed 20 ms bin edges from -0.5 to 1.5 s are converted to centers (-0.49 through 1.49 s) and the same float32 vector is used for every trial.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
...
inp = np.vstack((CENTERS_REL.astype(np.float32), ...))
```

iii. The notes justify centers as the time represented by each binned neural/behavior sample.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the centers of exactly the edges used for spike histograms; both have 100 corresponding time positions relative to the same stimulus onset.

ii.
```python
relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
...
inp = np.vstack((CENTERS_REL.astype(np.float32), ...))
```

iii. Independent checks in the notes report exact edges, centers, shapes, and a source spike histogram match.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full, unfiltered sequence of `trials.probabilityLeft`; a change in prior denotes a new block.

ii.
```python
prior_all = tr.probabilityLeft.to_numpy(float)
block_no_all = trial_number_in_block(prior_all)
```

iii. The notes state that the trials data have no separate block ID, while probability-left is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter starts at zero, increments while consecutive prior values are equal, and resets when the prior changes. It is computed before trial filtering and repeated across all 100 bins of a retained trial.

ii.
```python
for i in range(1, len(prior)):
    out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
...
np.full(N_TIME, block_no_all[idx[j]], np.float32)
```

iii. The notes say pre-filter computation preserves the animal's actual position in the original block; zero-based indexing makes block onset explicit.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `trials.choice` after retaining only -1 and +1 values.

ii.
```python
choice = tr.choice.to_numpy(float)[idx]
choice_cls = (choice == 1).astype(np.int64)
```

iii. The agent's notes assert that IBL -1 means left and +1 means right, but the human reference uses the opposite convention (+1 left, -1 right).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The Boolean test maps -1 to class 0 and +1 to class 1, then repeats that class across 100 time bins. This is reversed relative to the reference mapping required by the declared `['left', 'right']` labels.

ii.
```python
choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
...
np.full(N_TIME, choice_cls[j], np.int64)
```

iii. The agent reports independent spot checks, but those only confirm consistency with its chosen mapping, not the mapping's semantic correctness.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials.probabilityLeft` for each retained source trial.

ii.
```python
prior_all = tr.probabilityLeft.to_numpy(float)
prior = prior_all[idx]
```

iii. The notes identify this trial-table field as the task block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Each value is assigned to the nearest member of [0.2, 0.5, 0.8], producing class 0, 1, or 2, and repeated across time. Earlier validation already rejects values not close to those three.

ii.
```python
prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)
...
np.full(N_TIME, prior_cls[j], np.int64)
```

iii. The agent follows the mapping explicitly specified by the task and rejects unexpected/nonfinite values before conversion.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived directly from `_ibl_wheel.timestamps` and `_ibl_wheel.position` loaded as the ONE wheel object.

ii.
```python
wheel = one.load_object(eid, 'wheel')
wt, ws = wheel_speed(wheel['timestamps'], wheel['position'])
```

iii. The agent notes that native wheel data provide position/timestamps and speed must therefore be derived.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Nonfinite points are removed, samples are stable-sorted, duplicate/nonincreasing timestamps are removed, and `abs(np.gradient(position, time))` produces speed. It is linearly interpolated at neural-bin centers. This differs from the reference's `SessionLoader.load_wheel`, which regularizes at 1 kHz and computes low-pass-filtered velocity.

ii.
```python
t, pos = clean_timeseries(t, pos)
speed = np.abs(np.gradient(pos, t))
...
y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan)
```

iii. The notes allow “brainbox wheel processing (or gradient after timestamp QC)” and chose the gradient route, describing it as timestamp-aware. That does not reproduce the reference filter/interpolation procedure.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Across all retained trial-time values within each session, 1/3 and 2/3 quantiles define classes: values ≤q1 are 0, q1&lt;value≤q2 are 1, and values &gt;q2 are 2. Degenerate thresholds fall back to quantiles over unique values or constant/binary handling.

ii.
```python
q1, q2 = np.nanquantile(values, [1/3, 2/3])
...
out[values > q1] = 1
out[values > q2] = 2
```

iii. The agent chose within-session tertiles for balanced categories and consistency across sessions, and records thresholds/methods in metadata.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is linearly interpolated at the absolute timestamps formed by each retained stimulus onset plus the same 100 relative bin centers used by neural data.

ii.
```python
q = stim[:, None] + CENTERS_REL[None, :]
y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan)
```

iii. The notes say wheel and spikes share experiment time, so sampling on the common bin-center grid produces binwise alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from a side camera's `times` and `ROIMotionEnergy`, preferring the left camera and falling back to right.

ii.
```python
cam = one.load_object(eid, f'{side}Camera', collection='alf')
t, me = clean_timeseries(cam['times'], cam['ROIMotionEnergy'])
```

iii. The agent says each released stream is already the whisker-pad ROI metric and that asynchronous left/right streams should not be averaged.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released values are cleaned for finite, ordered, unique timestamps and linearly interpolated at the common trial-bin centers. No filtering or normalization is applied before categorization.

ii.
```python
t, me = clean_timeseries(cam['times'], cam['ROIMotionEnergy'])
...
whisk_cont = sample_trials(ct, me, stim)
```

iii. The notes state that no additional processing of released motion energy is necessary.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-wide 1/3 and 2/3 quantile categorization and degeneracy handling as wheel speed.

ii.
```python
whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)
```

iii. The agent particularly justifies session thresholds for motion energy because its arbitrary scale can differ between cameras and sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at `stimOn_times + CENTERS_REL`, exactly the same 100 trial-relative center times represented by neural bins.

ii.
```python
q = stim[:, None] + CENTERS_REL[None, :]
y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan)
```

iii. The agent relies on synchronized experiment timestamps and reports common-grid and finite-value checks.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Timeseries nonfinite values and duplicate timestamps are removed. Trials with missing required fields, unsupported categories, incomplete endpoint coverage, or nonfinite interpolated behavior are dropped. A camera side can fall back from left to right. A whole session is caught and recorded in `excluded_sessions` if an object fails, no unit remains, or fewer than two trials remain; values are not imputed.

ii.
```python
ok = np.isfinite(t) & np.isfinite(x)
...
for side in ('left', 'right'):
    try: ...
    except Exception as exc: ...
...
except Exception as exc:
    failures.append({'eid':str(eid),'error':f'{type(exc).__name__}: {exc}'})
```

iii. The notes argue that zero is meaningful for behavior, so missing dynamic outputs must not be zero-filled; exclusion remains auditable in metadata.

## 10-a. What are the most time-consuming steps of the code?

i. The main costs are building/scanning the local index, loading large spike-sorting arrays for every probe, merging cluster metadata, filtering/mapping millions of spike cluster assignments, sorting concatenated spikes, binning trials, and writing the nearly 1 GB pickle. The notes identify the broad-cohort runtime as the reason for restricting conversion to 22 sessions.

ii.
```python
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
...
mapped = np.fromiter((lut[int(c)] for c in sc[keep]), ...)
...
order = np.argsort(t, kind='stable')
```

iii. The notes measured 159 s for 22 sessions and called processing all 447 complete sessions “excessive prospective runtime.”

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python generator mapping every retained spike through a dictionary is the most important vectorization target; an array lookup would be faster. The per-trial spike-binning loop and final per-trial packaging loop could also be vectorized or batched. `trial_number_in_block` could be computed with change points and cumulative arithmetic, and repeated `list.index` assembly could use dictionaries.

ii.
```python
mapped = np.fromiter((lut[int(c)] for c in sc[keep]), dtype=np.int32, count=int(keep.sum()))
...
for i, st in enumerate(stim):
...
for j in range(len(idx)):
```

iii. The notes claim behavior interpolation is vectorized and use `searchsorted`/`bincount` to reduce cost, but do not discuss these remaining Python loops.

## 10-c. What processing does the code repeat multiple times?

i. For every session it reloads trials/wheel/camera, cleans and interpolates each behavioral stream, loads every probe, and allocates repeated time and trial-level label rows. It calls garbage collection after each probe and again after every session. At assembly, repeated `list.index` calls remap subjects and regions. Validation then traverses every trial again.

ii.
```python
gc.collect()
...
for k,eid in enumerate(eids,1):
    ...
    gc.collect()
...
subject_idx=np.array([subject_list.index(x) for x in subjects],dtype=np.int64)
region_idx=[np.array([brain_regions.index(x) for x in r],dtype=np.int64) for r in region_names]
```

iii. The notes present repeated cleanup as a memory-control measure and repeated validation as a safety check; they do not explicitly identify redundant processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `channels` only to merge cluster metadata and then discards it; computes/stores extensive session audit metadata not consumed by decoder training; always computes `continuous = None` unless plotting, while optional plotting itself is diagnostic only. It also builds a fresh filesystem-derived ONE index and maps methods EIDs, work necessitated by its cohort/index strategy rather than downstream data. The loaded `feedbackType` and `feedback_times` are used only to exclude trials, not as decoder variables.

ii.
```python
spikes, clusters, channels = sl.load_spike_sorting()
...
info.update(dict(... raw_trials=len(tr), ... wheel_tertiles=wheel_thr, ...))
...
continuous = (wheel_cont, whisk_cont) if plot else None
```

iii. The agent justifies audit metadata, sanity plots, and rebuilding the index as safeguards against stale cache metadata. These do not affect decoder tensors but support reproducibility and debugging.
