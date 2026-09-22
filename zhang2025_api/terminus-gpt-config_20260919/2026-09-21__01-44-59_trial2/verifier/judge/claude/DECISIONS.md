# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All arrays are read through the ONE API and `brainbox`, never by opening ALF files directly. Because the staged release tables mark the trials table and cluster metrics as non-default revisions (so `one.load_object` refused them in local mode), the AI built its own ONE index over the read-only cache with ONE's own indexer, `one.alf.cache.make_parquet_db`, and pointed a `mode='local'` `ONE` client at it. Session/dataset discovery is then done by filtering ONE's in-memory `_cache['datasets']` table for the required dataset names (trials table, wheel position/timestamps, `spikes.times`, and left- or right-camera times + ROI motion energy), which yields 447 "complete" sessions out of the 461 indexed. Trials, wheel and camera are loaded per session with `one.load_object`; spikes/clusters are loaded per probe with `brainbox.io.one.SpikeSortingLoader`.

The decisive decision is what happens next: **the 447 complete sessions are then reduced to the 40-EID `repro_ephys_release.txt` cohort** shipped with the methods-paper repo. Because the locally built index uses deterministic path-hash EIDs rather than Alyx EIDs, the AI opened a *second* `ONE` client on `/app/data/one_cache` with the `Brainwidemap` tag and matched the 40 Alyx EIDs to local EIDs on `(lab, subject, date, number)`. Only 22 of the 40 matched, and the conversion ran on those 22 sessions / 22 subjects / 12,943 trials / 3,984 units. The 18 unmatched EIDs were logged as a count and not investigated further. There is no fallback to the full cohort once at least one EID matches.

ii.
```python
CACHE_ROOT = Path('/mnt/dataset/one_cache')
INDEX_DIR = Path('/app/cache/local_one_index')
TARGET_EIDS = Path('/app/code/code_zhang2025/data/repro_ephys_release.txt')

def get_one():
    """Return local ONE backed by an index built by ONE itself."""
    ...
    make_parquet_db(CACHE_ROOT, out_dir=INDEX_DIR, hash_ids=True, hash_files=False)
    one = ONE(cache_dir=CACHE_ROOT, mode='local')
    one.load_cache(tables_dir=INDEX_DIR)
    return one
```

```python
def dataset_eids(one, filename):
    d = one._cache['datasets']
    names = d.rel_path.astype(str).str.rsplit('/', n=1).str[-1]
    return set(d.index.get_level_values('eid')[names.eq(filename)])

def candidate_eids(one):
    core = (dataset_eids(one, '_ibl_trials.table.pqt') &
            dataset_eids(one, '_ibl_wheel.timestamps.npy') &
            dataset_eids(one, '_ibl_wheel.position.npy') &
            dataset_eids(one, 'spikes.times.npy'))
    left = dataset_eids(one, '_ibl_leftCamera.times.npy') & dataset_eids(one, 'leftCamera.ROIMotionEnergy.npy')
    right = dataset_eids(one, '_ibl_rightCamera.times.npy') & dataset_eids(one, 'rightCamera.ROIMotionEnergy.npy')
    complete = core & (left | right)
    if TARGET_EIDS.exists():
        targets = [x.strip() for x in TARGET_EIDS.read_text().splitlines() if x.strip()]
        release = ONE(cache_dir='/app/data/one_cache', mode='local')
        release.load_cache(tag='Brainwidemap')
        rs = release._cache['sessions']; ls = one._cache['sessions']
        target_rows = rs.loc[[i for i in rs.index if str(i) in set(targets)]]
        def key(row):
            return (str(row.get('lab')), str(row.get('subject')), str(row.get('date')), str(row.get('number')))
        local_by_key = {key(row): eid for eid, row in ls.iterrows() if eid in complete}
        selected = [local_by_key[key(row)] for _, row in target_rows.iterrows() if key(row) in local_by_key]
        if selected:
            print(f'Cohort: mapped {len(selected)}/{len(targets)} methods-paper EIDs to complete local BWM sessions')
            return selected
    print('WARNING: methods cohort unavailable; using all complete BWM sessions')
    return sorted(complete, key=str)
```

Per-session loading:
```python
tr = trial_frame(one.load_object(eid, 'trials'))
wheel = one.load_object(eid, 'wheel')
cam = one.load_object(eid, f'{side}Camera', collection='alf')
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
```

iii. For the index: "Important cache issue: several release-table records report `exists=True` but non-default/stale trial-table and cluster-metrics files are not loadable by ONE in this staged snapshot... the conversion must select sessions/objects that actually load and must not bypass ONE," resolved by "building an accurate index with ONE's `make_parquet_db` over the read-only staged ALF hierarchy."

For the cohort, the notes are self-contradictory. Step 2 and Step 4 explicitly reject the repro-ephys list: "Only 24 of those occur in the current broad BWM session table; therefore it is not identical to the data-paper BWM cohort and is used for processing guidance, not blindly as the data cohort", and the Step 4 resolution is "Start from the 459 BWM sessions and filter only for required loadable modalities/QC." Step 6 then reverses this under the heading **"Code speedups added"**: "Restrict full processing to locally available members of the 39-session methods-paper cohort, mapped via ONE metadata (22 complete sessions)." Step 10 states the reason plainly: "**Excessive prospective runtime**: Processing all 447 complete local sessions would exceed 15 minutes. Resolved by applying the methods-paper session cohort; 22 complete staged sessions convert in 159 s."

## 1-b. How are the data split into subjects?

i. The subject name is read from ONE's session table for each eid (`one._cache['sessions'].loc[eid]`), so nothing is parsed from paths. `subjects` is the sorted set of names over converted sessions and `subject_idx` is each session's index into that list, with `'unknown'` substituted if the field is null. The mechanism matches the reference; only the coverage differs (22 subjects, one session each, vs 136–138 available).

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
subject_list = sorted(set(subjects))
subject_idx = np.array([subject_list.index(x) for x in subjects], dtype=np.int64)
```

iii. The notes treat subject identity as metadata that ONE already supplies; Step 2 documents the subject counts per cohort ("Broad manifest: 143; complete requested-modality cohort: 136") from the same session table. No derivation was considered necessary.

## 1-c. How are the data split into sessions?

i. A session is the unit ONE indexes, one eid per row of the session table, so no splitting is done. The one wrinkle is that the locally built index assigns deterministic path-hash eids instead of the Alyx UUIDs, so the AI carries a `(lab, subject, date, number)` mapping to recover the methods-paper session identities (see 1-a). Each session is processed independently in a serial loop and appended to the session-level lists.

ii.
```python
one = get_one(); eids = candidate_eids(one)
if args.sample: eids = eids[:2]
for k, eid in enumerate(eids, 1):
    n, i, o, r, info, cont = process_session(one, eid, plot=args.show_processing and k <= 2)
    neural.append(n); inputs.append(i); outputs.append(o); region_names.append(r)
```

iii. "Local path-hash session IDs are mapped to methods-paper Alyx EIDs by `(lab, subject, date, number)` using two ONE session tables" — needed only because the AI replaced the release tables with its own index. Otherwise no decision: "A session is one decoder session."

## 1-d. How are the data split into trials?

i. The trials table has one row per trial, so the split is given by the data. `trial_frame` normalises whatever ALF returns into a DataFrame: if a `table` sub-object is present it is used and any equal-length 1-D attributes are attached as extra columns; otherwise all 1-D attributes are collected and `intervals` is unpacked into `intervals_0`/`intervals_1` so the reference code's duration criterion can be applied.

ii.
```python
def trial_frame(tr):
    """Normalize an ALF trials object to a DataFrame."""
    if 'table' in tr and isinstance(tr['table'], pd.DataFrame):
        df = tr['table'].copy()
        for k, v in tr.items():
            if k != 'table' and np.asarray(v).ndim == 1 and len(v) == len(df):
                df[k] = v
        return df
    cols = {}
    for k, v in tr.items():
        a = np.asarray(v)
        if a.ndim == 1:
            cols[k] = a
        elif k == 'intervals' and a.ndim == 2 and a.shape[1] == 2:
            cols['intervals_0'], cols['intervals_1'] = a[:, 0], a[:, 1]
    return pd.DataFrame(cols)
```

iii. The notes record that the reference builds its mask on a trials DataFrame via `load_trials_and_mask`/`SessionLoader`, and that the installed brainbox API differs from the one the methods code was written against ("API-version difference, not a processing decision"), so the AI reproduces the same DataFrame shape from the ONE object instead of importing the reference function.

## 1-e. How are trials filtered based on quality controls?

i. Six criteria, combined into one mask applied before any binning:
1. finiteness of `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, `firstMovement_times` (the data paper's NaN-exclusion list);
2. `choice ∈ {-1, +1}` (drops no-response trials);
3. `probabilityLeft ∈ {0.2, 0.5, 0.8}`;
4. trial duration `intervals_1 - intervals_0` in `(0, 10]` s (the reference code's `max_trial_len=10.0`);
5. the whole `[stimOn-0.5, stimOn+1.5]` window must lie between the **first and last** wheel sample and the first and last camera sample;
6. after interpolation, trials whose resampled wheel or whisker trace contains any non-finite value are dropped.

Sessions left with fewer than two trials raise and are logged as excluded. **The reaction-time window (0.08–2.00 s between stimulus onset and first wheel movement) is not applied**, even though the AI documented it as a requirement. Consequently 588 trials/session are retained on average, against 428/session in the reference conversion.

ii.
```python
def valid_trials(df, wheel_t, cam_t):
    n = len(df)
    mask = np.ones(n, dtype=bool)
    required = ['choice', 'probabilityLeft', 'feedbackType', 'feedback_times',
                'stimOn_times', 'firstMovement_times']
    for c in required:
        if c not in df:
            raise KeyError(f'missing trial column {c}')
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

```python
mask = valid_trials(tr, wt, ct)
idx = np.flatnonzero(mask)
if len(idx) < 2: raise RuntimeError(f'only {len(idx)} valid trials')
...
finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
idx, stim = idx[finite], stim[finite]
```

iii. The intended rule, from Step 3/Step 4, was the union of the paper's and the code's criteria: "Data paper excludes a trial when any of these cannot be detected: `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, or `firstMovement_times`"; "Reference `load_trials_and_mask` also applies maximum trial duration 10 s and finite event-time/reaction-time criteria"; "Trial QC | Reference mask checks event NaNs, **RT limits**, and max 10 s trial | ... | Apply the union of paper and code requirements to loaded tables; additionally require complete wheel/whisker support for the requested window." Step 5 then restates the plan without the RT bounds ("Require finite/detected choice, probabilityLeft, feedbackType/time, stimOn, firstMovement; duration <=10 s; supported probability and choice categories; and full neural/wheel/motion temporal coverage"), and Step 10's reference comparison asserts "Same required event finiteness and max duration; conversion adds required category and complete behavior-window support" — the RT criterion silently disappears between plan and implementation and is never flagged.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` per probe insertion, loaded by `SpikeSortingLoader`. The merged cluster table supplies `cluster_id`, `label` (QC) and `atlas_id` (anatomy), used only for selection and for `brain_region_idx`; the neural matrix itself is built from the two spike arrays.

ii.
```python
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
cid = np.asarray(clusters['cluster_id']).astype(int)
label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
...
all_t.append(np.asarray(spikes['times'])[keep].astype(np.float64))
all_c.append(mapped)
```

iii. Step 1 identifies `load_spiking_data`/`SpikeSortingLoader` as the reference's loading path and `merge_probes` as its pooling step; Step 4 concludes "Neural representation | Cache code histograms spike counts... Native data are spike event times and cluster IDs".

## 2-b. How is the `neural` data processed?

i. Probes are pooled into one session population: surviving clusters of each probe are renumbered contiguously with a running `offset`, the concatenated spike times are stably sorted, and the trial window is sliced with `searchsorted`. Within a trial each spike gets a bin index `floor((t - lo)/0.02)` and a single `bincount` on the flattened `unit * 100 + bin` index fills the (n_units, 100) matrix. **The counts are left as counts** — they are not divided by the bin width, so the stored values are spikes per 20 ms bin rather than Hz — and no smoothing or standardisation is applied. Arrays are stored as float32.

ii.
```python
lut = {int(c): i + offset for i, c in enumerate(good_ids)}
sc = np.asarray(spikes['clusters']).astype(int)
keep = np.isin(sc, good_ids)
mapped = np.fromiter((lut[int(c)] for c in sc[keep]), dtype=np.int32, count=int(keep.sum()))
...
offset += len(good_ids)
t = np.concatenate(all_t); c = np.concatenate(all_c)
order = np.argsort(t, kind='stable')
return t[order], c[order], np.asarray(all_reg, dtype=object), raw_units, probes
```

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

iii. "Spike counts are the neural representation at caching time. Unit-wise standardization is applied later by the model data loader; the target conversion should preserve neural activity rather than pre-standardize unless decoder requirements demand otherwise." Step 4: "Store raw integer-like spike counts as float32 matrices; do not unit-standardize converted neural data because the validator/decoder should see preserved counts." Probe merging follows the paper's non-independence argument: "All probes in it are merged with unique cluster IDs, matching the paper's non-independence rationale."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two per-cluster criteria on the merged cluster table: `label >= 1` (the IBL pipeline's composite spike-sorting score, whose values are {0, 1/3, 2/3, 1}, so this keeps units passing every metric) and `atlas_id > 0`, which discards clusters with an unresolved/out-of-brain atlas assignment (id 0, i.e. `void`). Units whose Beryl acronym is `root` are deliberately kept (429 of 3,984). Surviving Allen ids are remapped to Beryl for `brain_regions`. A probe with no surviving unit is skipped; a session with none raises and is excluded.

ii.
```python
good = (label >= 1) & (atlas > 0)
good_ids = cid[good]
if not len(good_ids):
    continue
...
beryl_ids = BR.remap(atlas[good], source_map='Allen', target_map='Beryl')
all_reg.extend(BR.id2acronym(beryl_ids).tolist())
...
if offset == 0:
    raise RuntimeError('no QC-passing units')
```

iii. "The reference code calls `load_spiking_data(..., qc=1.0)` and retains clusters with `label >= qc`; because labels are in {0, 1/3, 2/3, 1}, this means fully passing (`label == 1`) units." On `root`: "`root` is a valid positive atlas mapping and appears for 429 units. The reference unit loader filters by QC and atlas mapping but does not automatically discard this Beryl category. It is retained and explicitly named, avoiding an undocumented unit deletion." On the region-level criteria of the data paper: "For decoder-format preservation, retain all QC-passing units and their region IDs; do not discard a whole session solely because one region has fewer than five units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams are already on one session clock, so alignment is a subtraction: for each trial the window `[stimOn - 0.5, stimOn + 1.5]` is located in the sorted spike-time array with `searchsorted`, and bin indices are computed relative to the window start `lo = stimOn + OFF_START`. Spikes falling outside `[0, 100)` after the subtraction are masked out rather than clipped. Trials are indexed by `stim = tr.stimOn_times.to_numpy(float)[idx]`, i.e. the same `stimOn_times` used for the behavioural streams.

ii.
```python
OFF_START, OFF_END = -0.5, 1.5
...
for i, st in enumerate(stim):
    lo, hi = st + OFF_START, st + OFF_END
    a, b = np.searchsorted(spike_t, [lo, hi])
    relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)
    clu = spike_c[a:b]
    ok = (relbin >= 0) & (relbin < N_TIME) & (clu >= 0) & (clu < n_units)
```

iii. Step 4: "Neural and behavioral streams use identical 100-bin edges from -0.5 to +1.5 s relative to stimulus onset"; "Camera timestamps and wheel timestamps are in experiment time. Continuous behavior must be interpolated/binned against the same absolute trial-relative edges as spikes." Verified by an independent check: "Loaded original spike sorting with `SpikeSortingLoader`... used `np.histogram` on stimulus-relative edges. All 100 bins for session 0/trial 5/unit 0 pass `np.allclose`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins. The 2 s window (-0.5 to 1.5 s) is cut into 101 edges / 100 bins on a fixed grid shared by every trial and session; `metadata['time_bin_size'] = 20.0` ms. Spikes are histogrammed directly onto that grid, so there is no rebinning or resampling of the neural data; the behavioural traces are interpolated once onto the bin centres of the same grid.

ii.
```python
DT = 0.020
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = len(CENTERS_REL)
```

```python
metadata=dict(..., time_bin_size=20.0, ..., off_start=OFF_START, off_end=OFF_END, n_timepoints=N_TIME, ...)
```

iii. "Reference defaults in that script are `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)` s, `binsize=0.02` s, and `interval_len=2` s. Thus each retained trial has 100 common 20 ms bins around stimulus onset." Cross-checked against the data paper: "We averaged wheel values in nonoverlapping 20-ms bins… Spike counts were similarly binned." Step 4: "Use fixed 20 ms edges shared by all modalities (100 bins/trial)."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table — it is the alignment event, and the input is the vector of 100 bin centres of the window taken around it, so the same values (-0.49 … 1.49 s) serve every trial in every session. Nothing else in the raw data is consulted.

ii.
```python
stim_all = tr.stimOn_times.to_numpy(float)
stim = stim_all[idx]
```
```python
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

iii. "Use stimulus onset and -0.5/+1.5 s because both the task and methods caching code specify it, even though the data paper's dedicated wheel analysis was movement-aligned"; the Decoder Task section of the instructions also mandates stimulus-onset alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid: 101 edges from -0.5 to 1.5 in 0.02 steps, midpoints taken, cast to float32 and stacked as row 0 of every trial's input. The AI chose a continuous ramp of bin centres rather than the binary onset indicator the format notes mention, on the grounds that the variable is "time since" the event and every trial shares the same axis.

ii.
```python
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. Step 5: "both decoder inputs will be 2×100 time-varying arrays". Step 7 records the resulting range: "Time-since-stimulus range | [-0.49, 1.49] s at bin centers (validator rounds display to [-0.5,1.5])", and Step 10's structural check confirms "Every trial has exactly the same 100 centers from -0.49 to +1.49 s".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction it *is* the neural grid: `CENTERS_REL` are the midpoints of `EDGES_REL`, and the spike bin index is computed as `floor((t - (stimOn + OFF_START))/DT)` on those same edges. Bin *k* of the input therefore labels the centre of bin *k* of the neural matrix, for every trial, with no offset.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + DT / 2, DT)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```
```python
relbin = np.floor((spike_t[a:b] - lo) / DT).astype(np.int32)   # lo = stimOn + OFF_START
```

iii. "Bin edges are exactly [-0.5,1.5] with 100 half-open 20 ms bins"; the `--show-processing` plots were used as the visual check — "Plots show the stimulus at 0 s, spike-count raster over the exact -0.5/+1.5 s interval, aligned wheel and whisker continuous traces, and categorical overlays... no visible temporal shift or edge truncation."

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The table carries no block identifier, so blocks are recovered from the fact that the prior is constant within a block: a change of value starts a new block.

ii.
```python
prior_all = tr.probabilityLeft.to_numpy(float)
block_no_all = trial_number_in_block(prior_all)
```

iii. Step 5: "trial number in block resets when `probabilityLeft` changes"; the metadata records `trial_number_in_block='zero-based; resets when probabilityLeft changes'`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter: trial 0 is 0, and each subsequent trial is `previous + 1` if its prior equals the previous trial's (compared with `np.isclose`) or 0 if it changed. Crucially the counter is computed over the **unfiltered** trials table and only then indexed by the surviving trial indices, so a trial that is later dropped still advances the count and the value is the animal's real position in the block. The value is broadcast constant across the 100 bins as row 1 of the input. Observed range across the dataset is 0–98, consistent with the paper's 20–100-trial blocks and the 90-trial unbiased opening block.

ii.
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
    return out
```
```python
inp = np.vstack((CENTERS_REL.astype(np.float32),
                 np.full(N_TIME, block_no_all[idx[j]], np.float32)))
```

iii. The notes state the counter is "derived exactly from source prior transitions" and verify it independently: "Choice mapping, prior mapping, all 100 time centers, and repeated zero-based block-trial numbers pass `np.allclose`"; "trial-number-in-block is constant within a trial; source trial indices are strictly increasing."

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1, -1 or 0. Trials with 0 (no response) are removed by the trial mask (`np.isin(choice, [-1, 1])`), so only the two signed values reach the output.

ii.
```python
choice = tr.choice.to_numpy(float)[idx]
choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
```

iii. "Trial-level behaviors assembled by reference code include `choice`, `block` (`probabilityLeft`)... The requested conversion uses choice and prior directly." No-response trials are dropped as part of the paper's trial exclusions.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Only a recode to the two classes the instructions ask for, broadcast constant over the 100 bins and declared as `output_values[0] = ['left','right']`. The AI's mapping is `choice == +1 → 1 ("right")`, `choice == -1 → 0 ("left")`. **This is the opposite of the IBL convention**, where `choice == +1` is a *leftward* choice and `choice == -1` a *rightward* one (`brainbox/examples/plot_all_peths.py`: ":param choice: subject's choice (1=left, -1=right)"; `brainbox/behavior/training.py`: `rightward = trials.choice == -1`). The reference solution encodes `{+1: 0, -1: 1}`. The AI's class labels are therefore inverted with respect to the requested "left = 0, right = 1" specification; the reported 0.475/0.525 left/right split is the mirror of the true one. Because the variable is binary and the relabelling is symmetric, decoder accuracy is unaffected — which is why the internal checks did not catch it.

ii.
```python
choice_cls = (choice == 1).astype(np.int64)  # -1 left, +1 right
...
out = np.vstack((np.full(N_TIME, choice_cls[j], np.int64), ...)).astype(np.int64)
```
```python
output_values=[['left','right'], ['0.2','0.5','0.8'], ['low','medium','high'], ['low','medium','high']]
```

iii. The convention was asserted, not derived: Step 5 states "choice maps IBL -1/left to 0 and +1/right to 1", and the README repeats "Choice: left = 0, right = 1". Step 12's investigation of the below-1.5×-chance choice accuracy re-checked only that the mapping had been applied consistently, not that its polarity was right: "Raw choice values were independently checked for three specific trials against ONE and all matched (`-1` left -> 0, `+1` right -> 1)."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the three block values 0.2, 0.5 and 0.8. Trials whose value is not within 1e-6 of one of those three are dropped by the trial mask.

ii.
```python
mask &= np.isclose(prior[:, None], [0.2, 0.5, 0.8], atol=1e-6).any(axis=1)
...
prior = prior_all[idx]
```

iii. "Source prior has {0.2,0.5,0.8}"; the block prior is "held constant within a block" per the data paper, and the instructions specify the 0.2→0, 0.5→1, 0.8→2 mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A recode to {0, 1, 2} by nearest-value matching (`argmin` of the absolute distance to `[0.2, 0.5, 0.8]`), which is robust to float representation of the stored values, then broadcast constant across the 100 bins. Resulting distribution 0.417 / 0.147 / 0.436, matching the reference conversion's 0.418 / 0.141 / 0.442 closely.

ii.
```python
prior_cls = np.argmin(np.abs(prior[:, None] - np.array([.2, .5, .8])), axis=1).astype(np.int64)
```
```python
out = np.vstack((np.full(N_TIME, choice_cls[j], np.int64),
                 np.full(N_TIME, prior_cls[j], np.int64),
                 wheel_cls[j], whisk_cls[j])).astype(np.int64)
```

iii. "prior maps 0.2/0.5/0.8 to 0/1/2" (Step 5), matching the Decoder Task specification. Step 9: "Prior distribution | ... | [0.41660,0.14718,0.43622] | Exact supported values". The 0.5 class being much rarer is expected because it is only the 90-trial unbiased opening block.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps` and `_ibl_wheel.position`, loaded as the ONE `wheel` object. Speed is the magnitude of the time derivative of angular position, in rad/s.

ii.
```python
wheel = one.load_object(eid, 'wheel')
wt, ws = wheel_speed(wheel['timestamps'], wheel['position'])
```

iii. "Native wheel provides position and timestamps"; "**Wheel speed**: Use absolute timestamp-derived angular velocity. Speed, not signed velocity, is requested."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps. (1) `clean_timeseries` drops non-finite samples, stably sorts by timestamp and removes non-increasing duplicates. (2) `np.gradient(pos, t)` takes a timestamp-aware central difference of the raw encoder trace, and the absolute value is taken. (3) The resulting session-wide trace is linearly interpolated at the 100 bin centres of each trial.

Notably, the AI did **not** use `SessionLoader.load_wheel()`, which is what the reference code's `load_target_behavior('wheel-speed')` calls and which internally interpolates position onto a uniform 1000 Hz grid (`interpolate_position`) and differentiates it through a 20 Hz Butterworth low-pass (`velocity_filtered`) before `np.abs` is applied. Differentiating the raw, irregularly-sampled encoder ticks instead yields a much noisier and quantised speed trace than the reference's filtered velocity.

ii.
```python
def clean_timeseries(t, x):
    t, x = np.asarray(t, float), np.asarray(x, float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    order = np.argsort(t, kind='stable'); t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], x[keep]

def wheel_speed(t, pos):
    t, pos = clean_timeseries(t, pos)
    # Timestamp-aware central differences; native position is angular radians.
    speed = np.abs(np.gradient(pos, t))
    speed[~np.isfinite(speed)] = np.nan
    return t, speed
```
```python
def sample_trials(t, x, stim):
    """Linear interpolation at common bin centers (reference behavior method)."""
    q = stim[:, None] + CENTERS_REL[None, :]
    y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
    return y.astype(np.float32)
```

iii. The notes allow either route and the code took the second branch: "Wheel definition | Reference behavior loader uses wheel velocity/speed interpolation | Native wheel provides position and timestamps | Paper defines dynamic wheel speed/velocity and averages values in 20 ms bins | **Compute velocity with brainbox wheel processing (or gradient after timestamp QC)**, take absolute value for speed, and average/interpolate consistently into task bins." The mapping table likewise says "Compute timestamp-aware velocity with brainbox wheel utilities". No reason is given for preferring the gradient, and the absence of the 1000 Hz resample and 20 Hz filter is never noted as a divergence from the reference — Step 10's comparison table records only "same interpolation concept".

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three within-session classes at the 1/3 and 2/3 quantiles of that session's own speed values, pooled over all retained trials *and* all 100 bins (`np.nanquantile` on the full (n_trials, 100) array). Assignment is `<= q1 → 0`, `(q1, q2] → 1`, `> q2 → 2`; non-finite values get -1 (those trials are already removed). Degenerate cases are handled explicitly: if `q2 == q1` (long runs of zero speed), thresholds are recomputed from the *distinct* values, or collapsed for two/one unique values, and the method and thresholds are recorded per session in `metadata['session_info']`. The realised distribution is 0.333/0.333/0.333 in every session, matching the reference's balanced tertiles.

ii.
```python
def discretize_tertiles(values):
    finite = np.isfinite(values)
    q1, q2 = np.nanquantile(values, [1/3, 2/3])
    method = 'quantile'
    if not q2 > q1:
        # Preserve equal values in one class; derive thresholds from distinct values.
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
    out[~finite] = -1
    return out, (float(q1), float(q2)), method
```

iii. "**Discretization**: Compute 1/3 and 2/3 quantiles separately within each retained session over all finite valid trial-time bins. Session-level thresholds are necessary for camera motion energy because values are arbitrary units and camera scale differs; applying the same approach to wheel makes categorical balance and interpretation consistent. Values `<=q1`, `(q1,q2]`, and `>q2` map to 0/1/2. If quantiles coincide because of prolonged zeros, fall back to rank-based tertiles with stable ordering; store thresholds/method in session metadata." Also: "balanced classes aid decoding".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated at `stimOn + CENTERS_REL` — exactly the bin centres of the neural grid, measured from the same `stimOn_times` — so the two share a time axis bin for bin. The query points for all trials are built as one matrix and interpolated in a single vectorised `np.interp` call over the session-wide trace. Query points outside the stream return NaN, which removes the trial.

ii.
```python
q = stim[:, None] + CENTERS_REL[None, :]
y = np.interp(q.ravel(), t, x, left=np.nan, right=np.nan).reshape(len(stim), N_TIME)
```
```python
wheel_cont = sample_trials(wt, ws, stim)
```

iii. "Camera timestamps and wheel timestamps are in experiment time. Continuous behavior must be interpolated/binned against the same absolute trial-relative edges as spikes." Checked independently: "Loaded original wheel timestamps/position through ONE, independently cleaned timestamps, computed timestamp-aware absolute gradient, interpolated at bin centers, and applied stored tertiles. All 100 classes pass `np.allclose`."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, one value per video frame, used as released. The left camera is tried first and the right used only if the left is missing or unusable (fewer than 100 samples, or a times/values length mismatch); all 22 converted sessions used the left camera, and the choice is recorded per session as `motion_camera`.

ii.
```python
def load_camera(one, eid):
    errors = []
    for side in ('left', 'right'):
        try:
            cam = one.load_object(eid, f'{side}Camera', collection='alf')
            t, me = clean_timeseries(cam['times'], cam['ROIMotionEnergy'])
            if len(t) > 100 and len(t) == len(me):
                return side, t, me
        except Exception as exc:
            errors.append(f'{side}:{type(exc).__name__}')
    raise RuntimeError('no valid whisker motion stream (' + ','.join(errors) + ')')
```

iii. "Whisker motion energy is the mean absolute adjacent-frame difference over rectangular whisker-pad ROIs anchored using DLC nose-tip and eye estimates. Left and right cameras both provide this signal; use one camera consistently per session (prefer left when valid, otherwise right) rather than averaging asynchronous cameras." Step 4: "Prefer left camera because it directly views the whisker pad and is available in most sessions; if unavailable/invalid, use right. Never average asynchronous streams. Record selected camera/session." This mirrors the reference code's `bin_behaviors`, which tries `left-whisker-motion-energy` and falls back to right.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None beyond timestamp hygiene and resampling: `clean_timeseries` drops non-finite samples and non-increasing timestamps, and the released trace is then linearly interpolated onto the same 100 bin centres per trial by the same `sample_trials` used for the wheel. No filtering, smoothing or normalisation is applied before discretisation.

ii.
```python
camera, ct, me = load_camera(one, eid)
...
whisk_cont = sample_trials(ct, me, stim)
```

iii. The AI treats the released ROI motion energy as the finished signal ("Camera objects have timestamps and ROI motion energy"), matching the reference, which takes `SessionLoader.motion_energy[...]['whiskerMotionEnergy']` as-is. Arbitrary units are handled at the discretisation step instead: "Session-level thresholds are necessary for camera motion energy because values are arbitrary units and camera scale differs."

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. With the identical rule used for the wheel: within-session 1/3 and 2/3 quantiles over all retained trials × 100 bins, `<= q1 → 0`, `(q1, q2] → 1`, `> q2 → 2`, with the same degenerate-threshold fallbacks and per-session recording of the thresholds and method. Realised distribution 0.333/0.333/0.333 per session.

ii.
```python
whisk_cls, whisk_thr, whisk_method = discretize_tertiles(whisk_cont)
...
info.update(dict(..., whisker_tertiles=whisk_thr, whisker_discretization=whisk_method, ...))
```

iii. Same justification as 7-c — session-level tertiles because motion energy is in arbitrary camera-dependent units, plus "balanced classes aid decoding, with deterministic handling of duplicate thresholds".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera trace is evaluated at `stimOn + CENTERS_REL`, i.e. the neural bin centres measured from the same stimulus onset on the shared session clock, in one vectorised `np.interp`. Out-of-range queries return NaN and drop the trial. No resampling of the 60 Hz camera stream onto an intermediate grid is done — it is interpolated straight onto the 50 Hz bin centres.

ii.
```python
whisk_cont = sample_trials(ct, me, stim)   # q = stim[:, None] + CENTERS_REL[None, :]
```

iii. "Continuous behavior must be interpolated/binned against the same absolute trial-relative edges as spikes, then common validity masks applied." Independently verified: "Loaded the selected original camera timestamps/ROI motion energy through ONE, independently interpolated and thresholded. All 100 classes pass `np.allclose`."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is dropped rather than imputed, at four levels.
- **Sample level**: `clean_timeseries` removes non-finite samples and non-monotonic/duplicate timestamps from the wheel and camera streams before use.
- **Trial level**: required trial fields must be finite; `choice`/`probabilityLeft` must be in the supported value sets; duration must be in (0, 10] s; the trial window must lie within the wheel and camera stream extents; and any trial whose interpolated wheel or whisker trace contains a NaN is discarded after resampling.
- **Session level**: the whole per-session call is wrapped in `try/except`; any failure (no QC-passing unit, no usable camera, fewer than two valid trials, a missing trial column) excludes the session and is recorded in `metadata['excluded_sessions']` with the exception text. Zero sessions failed in the full run.
- **Degenerate statistics**: coincident tertile thresholds fall back to distinct-value quantiles, and the chosen method is stored per session.

A final `validate()` pass asserts shapes, finiteness of neural and input arrays, and that every output value is inside its declared category set — which is what guarantees the `-1` "missing" code from `discretize_tertiles` never survives into the pickle.

One gap relative to the reference: coverage is tested only against the **first and last** sample of each stream, so a trial sitting inside a long interior camera or wheel dropout still passes and is silently interpolated across the gap. The reference instead requires a real sample within one bin of each window edge (as does the reference code's `get_behavior_per_interval`, which skips intervals whose "target data starts too late"/"ends too early").

ii.
```python
mask &= (starts >= wheel_t[0]) & (ends <= wheel_t[-1])
mask &= (starts >= cam_t[0]) & (ends <= cam_t[-1])
```
```python
finite = np.isfinite(wheel_cont).all(1) & np.isfinite(whisk_cont).all(1)
idx, stim = idx[finite], stim[finite]
wheel_cont, whisk_cont = wheel_cont[finite], whisk_cont[finite]
if len(idx) < 2: raise RuntimeError('fewer than two finite behavior trials')
```
```python
except Exception as exc:
    failures.append({'eid':str(eid),'error':f'{type(exc).__name__}: {exc}'})
    print('  EXCLUDED:',failures[-1]['error'],flush=True)
```
```python
def validate(data):
    ...
    assert np.isfinite(x).all() and np.isfinite(i).all()
    assert set(np.unique(o[0])) <= {0,1}; assert set(np.unique(o[1])) <= {0,1,2}
```

iii. "For this stimulus-aligned task, require complete support from -0.5 to +1.5 s for neural, wheel, and selected whisker streams; **exclude trials with missing/nonfinite required behavior rather than imputing outputs**"; "Missing methods EIDs or sessions lacking a required stream are not fabricated"; "Offline-cache loadability is an unavoidable additional curation criterion and will be quantified. Missing outputs will not be fabricated or imputed"; "Require at least two retained trials per session, as mandated by the target format."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk. Each probe contributes spike arrays with tens of millions of entries, and per-session times of 3–15 s (159 s total for 22 sessions) track the number of probes and units rather than the number of trials; the 468-unit, 3-probe session was the slowest at 14.6 s. The one-off ONE index build (`make_parquet_db`) costs 21.7 s. Behavioural loading, interpolation and spike binning are comparatively negligible. Per-session timings are printed and stored as `processing_seconds` in the metadata.

ii.
```python
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
```
```python
info.update(dict(..., processing_seconds=round(time.time()-t0, 3)))
```

iii. "Full spike sorting arrays can contain tens of millions of spikes/probe; redundant reloads must be avoided"; "Load each probe once and immediately discard large raw arrays | Limits repeated I/O and memory"; Step 7 records "ONE index creation | 21.7 s once | Reused for sample/full runs" and "Sample conversion | 7.45 s mean".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain, and one of them is on the hot path:
- `load_neural` maps cluster ids to compact unit indices with a **Python generator over every retained spike** (`np.fromiter((lut[int(c)] for c in sc[keep]), ...)`), i.e. one dictionary lookup and one int cast per spike, for tens of millions of spikes per probe. The reference does the same renumbering with a vectorised `np.cumsum(is_good) - 1` gather. `np.isin(sc, good_ids)` is likewise a sort-based membership test where a boolean flag indexed by cluster would be O(n).
- `bin_spikes` loops once per trial; this is the same structure the reference uses and costs little.
- `trial_number_in_block` is a scalar Python loop over trials; the reference vectorises it as `(prior != prior.shift()).cumsum()` + `groupby(...).cumcount()`.
- The per-trial assembly loop in `process_session` builds `np.full`/`np.vstack` arrays one trial at a time.

More consequentially, **the session loop is entirely serial**. The reference runs sessions in a `ProcessPoolExecutor` with 10 workers, which is what makes a 444-session conversion tractable; the AI instead used the projected serial runtime as the reason to shrink the dataset to 22 sessions.

ii.
```python
lut = {int(c): i + offset for i, c in enumerate(good_ids)}
sc = np.asarray(spikes['clusters']).astype(int)
keep = np.isin(sc, good_ids)
mapped = np.fromiter((lut[int(c)] for c in sc[keep]), dtype=np.int32, count=int(keep.sum()))
```
```python
def trial_number_in_block(prior):
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = 0 if not np.isclose(prior[i], prior[i-1]) else out[i-1] + 1
    return out
```
```python
for k, eid in enumerate(eids, 1):
    ...
    n,i,o,r,info,cont = process_session(one, eid, plot=args.show_processing and k<=2)
```

iii. The notes claim more vectorisation than the code contains: "Code speedups added: ... Slice spike arrays to QC units once, sort once, and use `searchsorted` + `bincount` for trial binning. Interpolate all behavior trial bins in vectorized arrays and release probe arrays between sessions", and Step 7 lists "Vectorized behavior interpolation and `bincount` spike binning | **Avoids per-unit/per-sample Python loops**". The per-spike loop in `load_neural` and the per-trial loop in `trial_number_in_block` are not mentioned anywhere, and parallel processing is never considered despite the instructions suggesting it.

## 10-c. What processing does the code repeat multiple times?

i. Little of consequence, and nothing per-trial or per-spike is recomputed. Minor repetitions: a **second `ONE` client is constructed and a second cache (`/app/data/one_cache`, tag `Brainwidemap`) is loaded** inside `candidate_eids` purely to translate 40 EIDs, downloading the release index in the process (visible at the top of `conversion_full_out.txt`); `dataset_eids` rebuilds the same `rel_path → basename` string split six times over the full 18,482-row dataset table; `load_camera` re-enters the load/clean path for the right camera whenever the left one fails; and assembly uses `list.index()` inside comprehensions (`subject_list.index`, `brain_regions.index`) instead of a dict, which is O(n_regions) per neuron. `validate()` walks every trial of every session a second time after assembly. None of these is measurable against the spike I/O.

ii.
```python
release = ONE(cache_dir='/app/data/one_cache', mode='local')
release.load_cache(tag='Brainwidemap')
```
```python
def dataset_eids(one, filename):
    d = one._cache['datasets']
    names = d.rel_path.astype(str).str.rsplit('/', n=1).str[-1]
    return set(d.index.get_level_values('eid')[names.eq(filename)])
```
```python
brain_regions = sorted(set(x for r in region_names for x in r.tolist()))
region_idx = [np.array([brain_regions.index(x) for x in r], dtype=np.int64) for r in region_names]
```

iii. The notes identify avoiding repetition as a goal — "redundant reloads must be avoided", "Load each probe once and immediately discard large raw arrays", "Slice spike arrays to QC units once, sort once" — and do not claim any remaining repeated processing. The per-session timings (3–15 s, dominated by probe count) support the claim that nothing significant is repeated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest case is in the bottleneck itself: `sl.load_spike_sorting()` is called with default attributes, so brainbox loads `spikes.amps` and `spikes.depths` in addition to `spikes.times` and `spikes.clusters`, then both are discarded. Those arrays are the same length as `spikes.times` (tens of millions of float entries per probe) and are staged for all 699–701 insertions, so this roughly doubles the I/O in the step the AI itself identified as dominant. The reference explicitly prevents it with `bio.SPIKES_ATTRIBUTES = ['clusters', 'times']`. `merge_clusters` additionally attaches the full metrics and histology tables when only `label`, `cluster_id` and `atlas_id` are read.

Smaller items: `valid_trials` requires `feedbackType`/`feedback_times` to be finite but neither value is carried into the output (this is deliberate — it is the paper's exclusion list — but the columns are otherwise unused); `raw_units`, `source_trial_indices`, the per-session tertile thresholds and the discretisation method are computed and stored in metadata but never read by the decoder; the continuous `wheel_cont`/`whisk_cont` traces are returned from `process_session` only when plotting; and `validate()` is a full extra pass over the assembled dataset.

ii.
```python
spikes, clusters, channels = sl.load_spike_sorting()          # also reads spikes.amps, spikes.depths
clusters = sl.merge_clusters(spikes, clusters, channels)       # full metrics + histology
cid = np.asarray(clusters['cluster_id']).astype(int)
label = np.asarray(clusters.get('label', np.zeros(len(cid))), float)
atlas = np.asarray(clusters.get('atlas_id', np.zeros(len(cid))), int)
```
```python
info.update(dict(eid=str(eid), probes=probes, motion_camera=camera,
                 raw_trials=len(tr), initially_valid_trials=int(mask.sum()), retained_trials=len(idx),
                 raw_units=int(raw_units), retained_units=len(regions),
                 wheel_tertiles=wheel_thr, wheel_discretization=wheel_method,
                 whisker_tertiles=whisk_thr, whisker_discretization=whisk_method,
                 source_trial_indices=idx.tolist(), processing_seconds=round(time.time()-t0, 3)))
```
```python
continuous = (wheel_cont, whisk_cont) if plot else None
```

iii. The metadata extras are deliberate and justified as audit trail: "store thresholds/method in session metadata", "`dynamic_discretization`='within-session behavioral tertiles; thresholds in session_info'", "logs exclusions/timings", "this reduction is explicit and auditable". The AI also states it minimised probe memory — "Load each probe once and immediately discard large raw arrays | Limits repeated I/O and memory" — but it never identifies the unused `amps`/`depths` reads, so the unnecessary I/O in its own acknowledged bottleneck went unnoticed.
