# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `ONE` with a locally-built parquet index (via `make_parquet_db`) over the staged cache at `/app/data/one_cache`. It then identifies candidate sessions by intersecting dataset availability for trials, wheel, spikes, and camera motion energy. Critically, the AI restricts the session cohort to the 39-session methods-paper list from `/app/code/code_zhang2025/data/repro_ephys_release.txt`, mapping those Alyx EIDs to local path-hash EIDs via `(lab, subject, date, number)` identity matching. This results in only 22 sessions being processed rather than the full BWM release.

ii.
```python
def get_one():
    """Return local ONE backed by an index built by ONE itself."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    ses, dat = INDEX_DIR / 'sessions.pqt', INDEX_DIR / 'datasets.pqt'
    if not (ses.exists() and dat.exists() and ses.stat().st_size > 100):
        make_parquet_db(CACHE_ROOT, out_dir=INDEX_DIR, hash_ids=True, hash_files=False)
    one = ONE(cache_dir=CACHE_ROOT, mode='local')
    one.load_cache(tables_dir=INDEX_DIR)
    return one

def candidate_eids(one):
    ...
    if TARGET_EIDS.exists():
        targets = [x.strip() for x in TARGET_EIDS.read_text().splitlines() if x.strip()]
        release = ONE(cache_dir='/app/data/one_cache', mode='local')
        release.load_cache(tag='Brainwidemap')
        ...
        selected = [local_by_key[key(row)] for _, row in target_rows.iterrows() if key(row) in local_by_key]
        if selected:
            return selected
```

iii. The AI justified using the methods-paper cohort as a speed optimization (processing all 447 sessions would exceed 15 minutes) and as a way to match the methods paper's session list. The AI noted that only 22 of the 40 methods-paper EIDs mapped to complete local sessions.

## 1-b. How are the data split into subjects?

i. Subject names are extracted from the ONE session cache metadata for each processed EID. A sorted unique list of subjects is built, and `subject_idx` maps each session to its index.

ii.
```python
info = session_details(one, eid)
...
subjects.append(info['subject'] or 'unknown')
...
subject_list = sorted(set(subjects))
subject_idx = np.array([subject_list.index(x) for x in subjects], dtype=np.int64)
```

iii. Subject metadata comes directly from ONE session tables. Each of the 22 sessions maps to a unique subject (22 subjects total, 1 session each).

## 1-c. How are the data split into sessions?

i. Each EID from the candidate list is processed independently as one session. Sessions are iterated sequentially in a for-loop.

ii.
```python
for k, eid in enumerate(eids, 1):
    ...
    n, i, o, r, info, cont = process_session(one, eid, plot=...)
```

iii. No splitting needed; each EID is already a session.

## 1-d. How are the data split into trials?

i. Trials come from the ONE trials object loaded per session. The trials table has one row per trial. Valid trials are selected by a mask; each valid trial becomes one entry in the per-session lists.

ii.
```python
tr = trial_frame(one.load_object(eid, 'trials'))
...
mask = valid_trials(tr, wt, ct)
idx = np.flatnonzero(mask)
```

iii. The trials table is already one row per trial; no further splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the following trial filters: (1) all required columns (`choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times`) must be finite, (2) choice must be -1 or 1, (3) probabilityLeft must be close to 0.2, 0.5, or 0.8, (4) trial duration must be positive and <= 10 s (if intervals available), (5) the trial window [-0.5, 1.5] must be fully within wheel and camera timestamp coverage. Additionally, trials with non-finite interpolated wheel or whisker values are dropped. **Notably, no reaction time bounds are applied** (the reference uses 80 ms to 2 s bounds).

ii.
```python
def valid_trials(df, wheel_t, cam_t):
    ...
    required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
                'stimOn_times', 'firstMovement_times']
    for c in required:
        mask &= np.isfinite(df[c].to_numpy(float))
    choice = df.choice.to_numpy(float)
    prior = df.probabilityLeft.to_numpy(float)
    mask &= np.isin(choice, [-1, 1])
    mask &= np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(axis=1)
    if {'intervals_0', 'intervals_1'} <= set(df):
        duration = df.intervals_1.to_numpy(float) - df.intervals_0.to_numpy(float)
        mask &= np.isfinite(duration) & (duration <= 10) & (duration > 0)
    stim = df.stimOn_times.to_numpy(float)
    starts, ends = stim + OFF_START, stim + OFF_END
    mask &= (starts >= wheel_t[0]) & (ends <= wheel_t[-1])
    mask &= (starts >= cam_t[0]) & (ends <= cam_t[-1])
    return mask
```

iii. The AI based its trial filtering on the reference code's `load_trials_and_mask` which includes event finiteness and max trial duration (10 s). The AI noted this in CONVERSION_NOTES.md Step 4. The omission of RT bounds was not explicitly justified.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` loaded via `SpikeSortingLoader`, plus cluster metadata (`label`, `atlas_id`, `acronym`) for QC and anatomy.

ii.
```python
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
```

iii. Spike times and cluster assignments are loaded through the standard brainbox pathway, same as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over [-0.5, 1.5] s around stimulus onset (100 bins), yielding spike counts per unit per bin. **The counts are stored as-is (not divided by bin width)**, so the neural data is in units of spike counts rather than firing rates in Hz. Probes within a session are merged with unique cluster IDs.

ii.
```python
def bin_spikes(spike_t, spike_c, stim, n_units):
    out = np.zeros((len(stim), n_units, N_TIME), dtype=np.float32)
    for i, st in enumerate(stim):
        lo, hi = st + OFF_START, st + OFF_END
        a, b = np.searchsorted(spike_t, [lo, hi])
        relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
        clu = spike_c[a:b]
        ok = (relbin >= 0) & (relbin < N_TIME) & (clu >= 0) & (clu < n_units)
        flat = clu[ok] * N_TIME + relbin[ok]
        out[i] = np.bincount(flat, minlength=n_units * N_TIME).reshape(n_units, N_TIME)
    return out
```

```python
neural_arr = bin_spikes(spike_t, spike_c, stim, len(regions))
```

iii. The AI notes "Store counts, not firing-rate smoothing or z-scoring. This exactly matches the reference cache stage." in CONVERSION_NOTES Step 5. However, the reference solution divides by BIN to get firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` and `atlas_id > 0` are kept. This retains the best QC units that have resolved anatomical annotation. Clusters are mapped to Beryl atlas acronyms.

ii.
```python
label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
good = (label >= 1) & (atlas > 0)
```

iii. The AI follows the same QC threshold (`label >= 1`) as the reference and also excludes units with unresolved atlas IDs (atlas_id == 0). The reference solution excludes `void` Beryl acronyms but keeps `root`; the AI's `atlas_id > 0` filter differs slightly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are selected by searching for spikes within [stimOn + OFF_START, stimOn + OFF_END], then binned relative to the start of that window. This aligns data to stimulus onset.

ii.
```python
for i, st in enumerate(stim):
    lo, hi = st + OFF_START, st + OFF_END
    a, b = np.searchsorted(spike_t, [lo, hi])
    relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
```

iii. Stimulus onset alignment matches the reference code's `align_time='stimOn_times'` and the task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial spanning [-0.5, 1.5] s. No rebinning or smoothing is applied.

ii.
```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
```

iii. Matches the reference code's `binsize=0.02` and the papers' 20 ms bin size.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin center positions, which are defined by the time window and bin size parameters. The bin centers run from -0.49 to +1.49 s in 0.02 s increments.

ii.
```python
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
...
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. The time input is the bin-center grid itself, identical for every trial and session, matching the reference.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the bin center positions from the edge array. The same 100-element array is used for every trial.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

iii. N/A

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is the center of the same bins used to bin spikes, so alignment is by construction.

ii.
```python
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```
These same edges are used in `bin_spikes`:
```python
relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `trials.probabilityLeft`. A block change is detected whenever `probabilityLeft` differs from the previous trial. The trial number resets to 0 at each block boundary.

ii.
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
    return out
```

iii. The AI notes that the trials table carries no block identifier, so blocks are recovered from the prior sequence.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is computed on ALL trials (before filtering) then indexed to only the valid trials. It is zero-based, incrementing by 1 for each consecutive trial with the same `probabilityLeft`, resetting to 0 when it changes.

ii.
```python
prior_all = tr.probabilityLeft.to_numpy(float)
block_no_all = trial_number_in_block(prior_all)
...
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. Computing on all trials before filtering preserves the animal's real position in the block even when some trials are dropped.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which is +1 (left in IBL convention) or -1 (right).

ii.
```python
choice = tr.choice.to_numpy(float)[idx]
choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
```

iii. The AI's code comment says `-1 left, +1 right`, which is the **opposite** of the actual IBL convention (+1 = left, -1 = right). This means the mapping is inverted: left choices get value 1, and right choices get value 0, whereas the instructions specify left = 0, right = 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == 1` to 1, and `choice != 1` (i.e., -1) to 0. Due to the inverted understanding of IBL conventions, this results in left = 1 and right = 0, which is inverted from the instructions (left = 0, right = 1).

ii.
```python
choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
```

iii. The AI's comment indicates it believes -1 is left and +1 is right, which is the opposite of the IBL convention documented in the data paper and used by the reference code.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior = prior_all[idx]
prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)
```

iii. The AI maps to the nearest of {0.2, 0.5, 0.8} and takes the argmin index, producing 0, 1, 2 — matching the instruction mapping 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The argmin approach finds the closest value among [0.2, 0.5, 0.8] and returns its index (0, 1, or 2). The valid_trials function already ensures only values close to 0.2, 0.5, or 0.8 survive, so this rounding is effectively exact.

ii.
```python
prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)
```

iii. Functionally equivalent to a direct map of {0.2: 0, 0.5: 1, 0.8: 2}, since only those three values pass the trial filter.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps` and `_ibl_wheel.position`, loaded via ONE's wheel object.

ii.
```python
wheel = one.load_object(eid, 'wheel')
wt, ws = wheel_speed(wheel['timestamps'], wheel['position'])
```

iii. The raw wheel position and timestamps are the source data.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel speed as the absolute value of `np.gradient(pos, t)` after cleaning the timeseries (removing NaN, duplicate timestamps). This is a simple central-difference derivative without any filtering. The reference instead uses `SessionLoader.load_wheel()` which applies interpolation to 1000 Hz and a 20 Hz Butterworth low-pass filter before differentiation.

ii.
```python
def wheel_speed(t, pos):
    t, pos = clean_timeseries(t, pos)
    speed = np.abs(np.gradient(pos, t))
    speed[~np.isfinite(speed)] = np.nan
    return t, speed
```

Then sampled onto bin centers:
```python
def sample_trials(t, x, stim):
    q = stim[:, None] + CENTERS_REL[None, :]
    y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
    return y.astype(np.float32)
```

iii. The AI notes "Use absolute timestamp-derived angular velocity" in CONVERSION_NOTES Step 5. The lack of Butterworth filtering means the wheel speed signal will be noisier than the reference.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Within-session tertile discretization: the 1/3 and 2/3 quantiles of all finite wheel speed values across all valid trial time-bins are computed, then values are classified as 0 (<=q1), 1 ((q1,q2]), or 2 (>q2). A fallback handles degenerate cases where quantiles coincide.

ii.
```python
def discretize_tertiles(values):
    finite = np.isfinite(values)
    q1, q2 = np.nanquantile(values, [1/3, 2/3])
    ...
    out = np.zeros(values.shape, dtype=np.int64)
    out[values > q1] = 1
    out[values > q2] = 2
    out[~finite] = -1
    return out, (float(q1), float(q2)), method
```

iii. The AI splits at session-level quantiles to achieve approximately equal class sizes. This is conceptually the same as the reference's `np.percentile(trace, [33.3, 66.7])` approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same bin centers (`CENTERS_REL`) relative to stimulus onset using `np.interp`, matching the neural binning grid.

ii.
```python
def sample_trials(t, x, stim):
    q = stim[:, None] + CENTERS_REL[None, :]
    y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
    return y.astype(np.float32)
```

iii. Same time grid as neural data ensures alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy` (or `rightCamera` as fallback) and the corresponding camera timestamps, loaded via `one.load_object`.

ii.
```python
def load_camera(one, eid):
    for side in ('left', 'right'):
        try:
            cam = one.load_object(eid, f'{side}Camera', collection='alf')
            t, me = clean_timeseries(cam['times'], cam['ROIMotionEnergy'])
            if len(t) > 100 and len(t) == len(me):
                return side, t, me
        except Exception as exc:
            errors.append(f'{side}:{type(exc).__name__}')
```

iii. Left camera preferred, right as fallback, matching the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are cleaned (NaN/duplicate removal), then interpolated onto bin centers using `np.interp`. No additional filtering or normalization. Then discretized into 3 bins using session-level tertiles.

ii.
```python
camera, ct, me = load_camera(one, eid)
whisk_cont = sample_trials(ct, me, stim)
whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)
```

iii. The reference also uses the raw motion energy trace without additional processing, so this matches.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same tertile-based discretization as wheel speed: session-level 1/3 and 2/3 quantiles, values classified as 0, 1, or 2.

ii.
```python
whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Interpolated onto the same bin centers as neural data using `np.interp`, identical approach to wheel speed alignment.

ii.
```python
whisk_cont = sample_trials(ct, me, stim)
```

iii. Same time grid ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI applies several strategies: (1) `clean_timeseries` removes NaN and duplicate timestamps from wheel and camera data, (2) `valid_trials` requires all key trial columns to be finite, (3) trials with non-finite interpolated behavior are dropped, (4) sessions with fewer than 2 valid trials or no QC-passing units raise exceptions and are excluded, (5) probe insertions with no good clusters are skipped.

ii.
```python
def clean_timeseries(t, x):
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    order = np.argsort(t, kind='stable'); t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], x[keep]
```

```python
finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
idx, stim = idx[finite], stim[finite]
```

```python
if len(idx) < 2: raise RuntimeError(f'only {len(idx)} valid trials')
```

iii. Missing data is dropped rather than imputed, which is appropriate for neural decoder training.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk via `SpikeSortingLoader.load_spike_sorting()` is the dominant cost, as spike arrays can contain tens of millions of entries per probe.

ii.
```python
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
```

iii. The AI reports per-session processing times of 3-15 seconds, with spike loading being the main bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops run per-trial: (1) `bin_spikes` loops over trials to bin spikes, and (2) the main output construction loop builds per-trial input/output arrays. The `trial_number_in_block` function also loops element-by-element.

ii.
```python
def bin_spikes(spike_t, spike_c, stim, n_units):
    out = np.zeros((len(stim), n_units, N_TIME), dtype=np.float32)
    for i, st in enumerate(stim):
        ...
```

```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
    return out
```

iii. The `sample_trials` function is already vectorized. The per-trial binning loop and trial-number loop could theoretically be vectorized but the cost is small.

## 10-c. What processing does the code repeat multiple times?

i. The AI loads camera data through `load_camera` using `one.load_object`, and wheel data through `one.load_object`. These are loaded once per session, so there is no redundant processing.

ii. N/A

iii. No obvious repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `feedbackType` and `feedback_times` as part of trial validation (requiring them to be finite) even though they are not used in the final output. This adds filtering criteria not present in the reference solution.

ii.
```python
required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
            'stimOn_times', 'firstMovement_times']
for c in required:
    mask &= np.isfinite(df[c].to_numpy(float))
```

iii. The AI includes these based on the data paper's trial exclusion criteria, which mention excluding trials where these cannot be detected.
