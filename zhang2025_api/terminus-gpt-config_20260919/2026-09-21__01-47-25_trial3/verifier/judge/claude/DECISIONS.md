# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API with two separate ONE clients: one for the base `Brainwidemap` release tables and one for the `2025_Q3_IBL_et_al_BWM` update tables. It identifies candidate sessions by searching the cache datasets for sessions that have spike times, wheel timestamps, and at least one paired camera stream (left or right motion energy with timestamps). A `DATALIMIT_SUBSET.csv` is not used; instead, `candidate_eids()` filters from the full release index. Each session is loaded through ONE and brainbox loaders (`SpikeSortingLoader`, `load_object`, `load_dataset`), never by directly reading files.

ii.
```python
def make_ones():
    return (ONE(cache_dir=CACHE, tables_dir=BASE_TABLES, mode='local'),
            ONE(cache_dir=CACHE, tables_dir=UPDATE_TABLES, mode='local'))

def candidate_eids(base, update):
    bd, ud = base._cache['datasets'], update._cache['datasets']
    bp, up = bd.rel_path.astype(str), ud.rel_path.astype(str)
    def es(d, p, text):
        return set(d[p.str.contains(text, regex=False)].index.get_level_values('eid'))
    core = es(bd, bp, 'spikes.times') & es(bd, bp, '_ibl_wheel.timestamps.npy')
    left = es(ud, up, 'leftCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_leftCamera.times.npy')
    right = es(ud, up, 'rightCamera.ROIMotionEnergy.npy') & es(bd, bp, '_ibl_rightCamera.times.npy')
    return sorted(core & (left | right), key=str), left, right
```

iii. The AI explored the ONE cache structure in Step 2, found that the cache contains multiple release tables, and decided to use two separate ONE clients to handle revision discrepancies between tables. The base tables provide core ephys, wheel, and camera timing data, while the update tables provide the motion energy datasets.

## 1-b. How are the data split into subjects?

i. Subjects are identified using `one.eid2ref(eid)` which returns a reference containing the subject name. Sessions are grouped by subject in the final assembly step where unique sorted subject names form the `subjects` list and `subject_idx` maps each session.

ii.
```python
ref = base.eid2ref(eid)
out = dict(eid=str(eid), subject=str(ref.subject), ...)
# In build_output:
subjects = sorted({s['subject'] for s in sessions}); smap={x:i for i,x in enumerate(subjects)}
```

iii. The ONE API provides session metadata including the subject name, so no manual parsing is needed.

## 1-c. How are the data split into sessions?

i. Each session is identified by a unique `eid` from the candidate list. The `candidate_eids()` function returns all qualifying session EIDs from the release index. Each session is processed independently via `process_session()`.

ii.
```python
eids, left, right = candidate_eids(base, update)
# Each eid processed independently:
sess, raw = process_session(base, update, eid, left, right, ...)
```

iii. Sessions are the natural unit of organization in the ONE cache. No splitting is needed.

## 1-d. How are the data split into trials?

i. Trials are loaded from the aggregate trial table (`_ibl_trials.table.pqt`) via a custom `load_trial_table()` function that handles revision reconciliation. Each row is one trial.

ii.
```python
def load_trial_table(one, eid):
    ...
    return one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf',
                            revision=rev, check_hash=False)
```

iii. The trials table is one row per trial, so no splitting is required.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a `trial_mask()` function that requires: all key columns (`choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`) must be finite; choice must be -1 or +1 (excludes no-go); prior must be 0.2, 0.5, or 0.8; reaction time (firstMovement - stimOn) must be between 0.08 and 2.0 s. Additionally, trials are dropped if interpolated wheel or whisker values contain NaN (indicating incomplete behavioral coverage), and trials with all-zero neural data are removed.

ii.
```python
def trial_mask(trials):
    required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
                'stimOn_times', 'firstMovement_times']
    m = np.ones(len(trials), dtype=bool)
    for c in required:
        m &= np.isfinite(trials[c].to_numpy(dtype=float))
    choice = trials.choice.to_numpy(dtype=float)
    prior = trials.probabilityLeft.to_numpy(dtype=float)
    rt = trials.firstMovement_times.to_numpy(dtype=float) - trials.stimOn_times.to_numpy(dtype=float)
    m &= np.isin(choice, [-1., 1.])
    m &= np.isin(prior, [.2, .5, .8])
    m &= (rt >= .08) & (rt <= 2.0)
    return m
# Additional filtering:
complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
# And zero-neural filtering:
neural_valid = np.any(neural != 0, axis=(1, 2))
```

iii. The AI documented following data-paper trial curation rules (finite events, valid choice, valid prior, RT bounds). The additional `feedbackType` and `feedback_times` finite-ness checks go beyond the reference but ensure data completeness. The zero-neural trial removal addresses a validator warning.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe's pykilosort spike sorting, plus the `label` and `acronym` columns from the merged cluster table for quality/region filtering.

ii.
```python
ssl = SpikeSortingLoader(eid=eid, pname=pname, one=base)
spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
merged = ssl.merge_clusters(spikes, clusters, channels).to_df()
```

iii. Same spike sorting data as the reference code, loaded through brainbox loaders.

## 2-b. How is the `neural` data processed?

i. Spikes from good clusters are selected and binned into 20 ms bins over [-0.5, 1.5] s around stimulus onset using `brainbox.singlecell.bin_spikes2D`. Multiple probes are concatenated along the neuron axis. The result is stored as **spike counts** (not firing rates), cast to float32.

ii.
```python
binned, tscale = bin_spikes2D(np.asarray(spikes['times'])[selected_spikes],
                            spike_clusters[selected_spikes], ids, align,
                            pre_time=.5, post_time=1.5, bin_size=DT)
arrays.append(np.asarray(binned))
...
x = np.concatenate(arrays, axis=1)
return x.astype(np.float32), regions, uuids, spike_total
```

iii. The AI uses brainbox's built-in `bin_spikes2D` rather than manual binning. Spike counts are preserved rather than converted to firing rates, documented as "no information is lost" since the bin size is constant. The AI's CONVERSION_NOTES Step 5 states "Preserve counts (not rates or z-scores)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are retained (well-isolated neurons). Additionally, clusters whose Beryl-mapped acronym is in `{'void', 'root', 'nan', 'None', ''}` are excluded. This removes units outside the brain (`void`) AND units in non-specific regions (`root`).

ii.
```python
BAD_REGIONS = {'void', 'root', 'nan', 'None', ''}
acr = BrainRegions().acronym2acronym(allen_acr, mapping='Beryl').astype(str)
good = (label >= 1) & ~np.isin(acr, list(BAD_REGIONS))
```

iii. The AI documented following data-paper curation (75,708 well-isolated neurons at `label >= 1`). The exclusion of `root` goes beyond the reference, which only excludes `void`. The AI's CONVERSION_NOTES mention "Beryl root/void removal" and "62,761 in behavior-complete gray-matter cohort" vs 75,708 release-wide.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `stimOn_times` (stimulus onset) by passing the stimulus onset times as the `align` parameter to `bin_spikes2D`, which counts spikes in bins relative to those alignment times with `pre_time=0.5` and `post_time=1.5`.

ii.
```python
align = trials.stimOn_times.to_numpy(float)[mask]
binned, tscale = bin_spikes2D(..., align, pre_time=.5, post_time=1.5, bin_size=DT)
```

iii. The AI uses stimulus onset alignment as specified in both the instructions and reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial spanning [-0.5, 1.5] s. No rebinning is applied; spikes are directly counted into these bins.

ii.
```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
N_BINS = 100
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```

iii. Matches the reference code parameters (binsize=0.02, 100 time steps).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the bin centers of the temporal grid, computed as constants relative to `stimOn_times`. The bin centers are `[-0.49, -0.47, ..., 1.49]` seconds.

ii.
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```

iii. The time input is a deterministic grid defined by the temporal binning parameters, not derived from any raw data variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing. The bin centers are precomputed from the constants `OFF_START=-0.5`, `DT=0.02`, and `N_BINS=100`.

ii.
```python
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
ins.append(np.vstack((BIN_CENTERS, np.full(N_BINS, s['trial_number'][j], np.float32))))
```

iii. N/A

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centers represent the midpoints of the same 20 ms bins used for neural spike counting, so they are inherently aligned.

ii.
```python
# Same bin centers used for both neural binning and time input:
BIN_CENTERS = (OFF_START + DT / 2 + np.arange(N_BINS) * DT).astype(np.float32)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected by changes in `probabilityLeft`.

ii.
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
    return out

block_num = trial_number_in_block(trials.probabilityLeft.to_numpy(float))
```

iii. The trials table has no explicit block identifier, so blocks are inferred from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter that resets to 0 whenever `probabilityLeft` changes between consecutive trials. Computed on the full trial table before trial filtering, so the count reflects the animal's real position in the block.

ii.
```python
block_num = trial_number_in_block(trials.probabilityLeft.to_numpy(float))
mask = trial_mask(trials)
block_num = block_num[mask]
```

iii. Computing before filtering preserves the experimental block structure. The AI's CONVERSION_NOTES Step 5 confirm "Zero-based counter reset to 0 whenever prior changes; broadcast across 100 bins."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which takes values -1, 0, or +1.

ii.
```python
choice_raw = trials.choice.to_numpy(float)[mask]
choice = (choice_raw == 1).astype(np.uint8)
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == 1` to 1 and `choice == -1` to 0. No-go trials (choice=0) are excluded by the trial mask. In the standard IBL convention, `choice=-1` is left and `choice=+1` is right, so this maps left→0, right→1, matching the instructions.

ii.
```python
choice = (choice_raw == 1).astype(np.uint8)
```

iii. The AI's CONVERSION_NOTES Step 4 state "raw `choice=-1` (left) → 0 and `choice=+1` (right) → 1" but note the AI appears confused about which IBL value corresponds to which direction. The code `(choice_raw == 1)` maps IBL +1 to 1 and IBL -1 to 0. Under the standard IBL convention (-1=left, +1=right), this gives left=0, right=1, which matches the instructions. However, the reference code uses `{1.0: 0, -1.0: 1}` with the comment "+1 is a leftward choice," suggesting the reference author believed IBL +1 = left. The mapping is thus inverted between the two solutions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
prior_raw = trials.probabilityLeft.to_numpy(float)[mask]
prior = np.select([prior_raw == .2, prior_raw == .5, prior_raw == .8], [0, 1, 2]).astype(np.uint8)
```

iii. Same source and mapping as the reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A direct mapping: 0.2→0, 0.5→1, 0.8→2, as specified in the instructions.

ii.
```python
prior = np.select([prior_raw == .2, prior_raw == .5, prior_raw == .8], [0, 1, 2]).astype(np.uint8)
```

iii. Matches the instruction specification exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel `timestamps` and `position` arrays, loaded via `base.load_object(eid, 'wheel', collection='alf')`.

ii.
```python
def load_wheel_speed(base, eid, align):
    w = base.load_object(eid, 'wheel', collection='alf')
    ts = np.asarray(w['timestamps'], float)
    pos = np.asarray(w['position'], float)
```

iii. Same raw data as the reference, loaded through ONE.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is interpolated to 1 kHz using `wheellib.interpolate_position`, then a low-pass filtered velocity is computed with `wheellib.velocity_filtered`. Speed is the absolute value of velocity. The continuous speed is then interpolated onto trial bin centers using `np.interp` (vectorized across all trials). Finally, the speed values are discretized into 3 bins using **global** tertiles (1/3 and 2/3 quantiles computed across all sessions' wheel values).

ii.
```python
ipos, its = wheellib.interpolate_position(ts, pos, freq=1000)
vel, _ = wheellib.velocity_filtered(ipos, fs=1000)
return interp_trials(its, np.abs(vel), align), (ts, pos, its, vel)
# Discretization in build_output:
wheel_vals = np.concatenate([s['wheel'].ravel() for s in sessions])
thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float), ...}
wc = np.digitize(s['wheel'], thresholds['wheel']).astype(np.uint8)
```

iii. The AI's notes describe brainbox wheel processing and global tertile discretization. The wheel processing pipeline matches the reference (both use brainbox interpolation and filtering). The discretization uses global rather than per-session percentiles.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global 1/3 and 2/3 quantiles are computed over all retained wheel speed values from all sessions. Values are then classified using `np.digitize` into 3 bins (0=low, 1=medium, 2=high).

ii.
```python
wheel_vals = np.concatenate([s['wheel'].ravel() for s in sessions])
thresholds = {'wheel': np.quantile(wheel_vals, [1/3, 2/3]).astype(float), ...}
wc = np.digitize(s['wheel'], thresholds['wheel']).astype(np.uint8)
```

iii. The AI chose global tertiles for "comparable class meanings" across sessions. The reference uses per-session percentiles instead.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated at the same bin centers as the neural data, measured relative to stimulus onset, so both share the same time axis.

ii.
```python
def interp_trials(times, values, align):
    query = align[:, None] + BIN_CENTERS[None, :]
    out = np.interp(query.ravel(), times, values, left=np.nan, right=np.nan)
    return out.reshape(len(align), N_BINS).astype(np.float32)
```

iii. Interpolation at the neural bin centers ensures temporal alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The `ROIMotionEnergy` array from the left camera (preferred) or right camera (fallback), paired with the corresponding camera timestamps. Motion energy is loaded from the update ONE client, while timestamps are loaded from the base client.

ii.
```python
def load_whisker(base, update, eid, align, left_set, right_set):
    for camera, available in [('left', left_set), ('right', right_set)]:
        if uid not in available:
            continue
        me = load_motion_energy(update, eid, camera)
        times = base.load_dataset(eid, f'_ibl_{camera}Camera.times.npy', collection='alf')
```

iii. Left camera preferred, right as fallback, matching the reference code logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the trial bin centers using `np.interp`, then discretized into 3 bins using global tertiles (same approach as wheel speed).

ii.
```python
# Interpolation:
return interp_trials(times, me, align), camera, (times, me)
# Discretization in build_output:
whisk_vals = np.concatenate([s['whisker'].ravel() for s in sessions])
thresholds = {'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float), ...}
mc = np.digitize(s['whisker'], thresholds['whisker']).astype(np.uint8)
```

iii. No additional processing applied to the motion energy beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: global 1/3 and 2/3 quantiles over all sessions, classified via `np.digitize`.

ii.
```python
whisk_vals = np.concatenate([s['whisker'].ravel() for s in sessions])
thresholds = {'whisker': np.quantile(whisk_vals, [1/3, 2/3]).astype(float), ...}
mc = np.digitize(s['whisker'], thresholds['whisker']).astype(np.uint8)
```

iii. Global tertiles chosen for consistency across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker motion energy is interpolated at the same bin centers as neural data, aligned to stimulus onset.

ii.
```python
return interp_trials(times, me, align), camera, (times, me)
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) Trial mask requires all key columns to be finite. (2) Interpolation with `left=np.nan, right=np.nan` marks edge trials where behavior data doesn't cover the window. (3) Trials with any NaN in wheel or whisker are dropped. (4) Trials with all-zero neural data are dropped (39 trials affected). (5) Sessions with <2 valid trials or zero curated neurons raise exceptions and are skipped. (6) Probes with no good clusters are skipped. (7) Sessions failing to load are caught and logged as failures.

ii.
```python
# NaN-based coverage check:
complete = np.isfinite(wheel).all(1) & np.isfinite(whisk).all(1)
# Zero-neural removal:
neural_valid = np.any(neural != 0, axis=(1, 2))
# Minimum trial check:
if len(align) < 2:
    raise RuntimeError(f'fewer than two complete valid trials ({len(align)})')
```

iii. The AI's CONVERSION_NOTES document the 39 zero-neural trials removed and the session-level exception handling.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (tens of millions of spikes per session) and the brainbox `bin_spikes2D` binning operation. The AI estimated ~4.3 s/session serially.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting(revision='2024-05-06')
binned, tscale = bin_spikes2D(...)
```

iii. The AI's CONVERSION_NOTES Step 6 identify spike loading and binning as the dominant cost, consistent with the reference assessment.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_number_in_block` function uses a Python for-loop over all trials. This could be vectorized using cumulative operations (as the reference does with pandas `groupby().cumcount()`). However, this loop runs once per session over ~hundreds of trials and is negligible in cost.

ii.
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if prior[i] != prior[i - 1] else out[i - 1] + 1
    return out
```

iii. The loop is sequential by nature (each value depends on the previous) but could still be vectorized with cumsum tricks.

## 10-c. What processing does the code repeat multiple times?

i. In the parallel full-conversion mode, each worker process calls `make_ones()` and `candidate_eids()` independently, rebuilding the ONE clients and computing the full candidate list for every session. This is redundant; the candidate list could be computed once and passed to workers.

ii.
```python
def _process_worker(eid_str):
    base, update = make_ones()
    eids, left, right = candidate_eids(base, update)
    return process_session(base, update, base.to_eid(eid_str), left, right, False)[0]
```

iii. The redundancy is a consequence of using `ProcessPoolExecutor` where ONE clients can't be pickled, but the candidate computation is unnecessary to repeat.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and checks `feedbackType` and `feedback_times` as part of trial filtering, but these variables are not used in any decoder input or output. The AI also computes and stores `uuids`, `trial_indices`, `n_trials_raw`, `n_zero_neural`, and `nspikes` per session in metadata, which are diagnostic but not used by the decoder. The `validate()` function at the end performs a full integrity scan that duplicates what the external verifier does.

ii.
```python
# Unused in decoder but required finite:
required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
            'stimOn_times', 'firstMovement_times']
# Extra metadata:
out = dict(..., uuids=uuids, trial_indices=original_idx.astype(np.int32),
           n_trials_raw=len(trials), n_trials_valid=len(align), n_zero_neural=n_zero_neural, nspikes=spike_total)
```

iii. The extra filtering columns ensure data completeness but are stricter than necessary for the decoder task.
