# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything goes through the ONE API against the local IBL cache in `/app/data/one_cache`, in `mode='local'`. The session/probe inventory is not obtained from `one.search`, but from the reference repository's own freeze file `code/code_zhang2025/data/bwm_release.csv` (699 probe-insertion rows → 459 unique `eid`s, 139 subjects), exactly the file `0_data_caching.py` reads. Rows are grouped by `eid` so each job carries `(eid, [pids], [probe_names], subject, lab)`; sessions are then processed in a 32-worker `multiprocessing` pool, each worker building its own ONE client.

Before any loading, the AI added a repair step (`build_patched_tables`) because the shipped ONE cache tables are stale relative to the file tree: the tables list e.g. `alf/_ibl_trials.table.pqt`, whereas on disk the file only exists under a revision folder (`alf/#2025-03-03#/_ibl_trials.table.pqt`). The three shipped table directories are merged, every dataset row is re-pointed at the newest revision of that `(collection, filename)` actually present on disk, `file_size` is re-stat'ed, `hash` dropped, and `default_revision` forced to True. A `DATALIMIT_SUBSET.csv`, if present, restricts the `eid` list (it is absent in this environment, so all 459 sessions were attempted).

Per session, `SessionLoader` loads trials / wheel / motion energy and `SpikeSortingLoader` loads one probe insertion at a time.

ii.
```python
REPO = Path('/app/code/code_zhang2025')
BWM_RELEASE = REPO / 'data' / 'bwm_release.csv'
CACHE_DIR = Path('/app/data/one_cache')
...
def get_one(force_rebuild=False):
    """A local-mode ONE pointed at the patched tables."""
    from one.api import ONE
    tables_dir = build_patched_tables(force=force_rebuild)
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
               cache_dir=str(CACHE_DIR), tables_dir=str(tables_dir), mode='local')
```

```python
    bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
    subset_file = Path('/app/data/DATALIMIT_SUBSET.csv')
    if subset_file.exists():
        subset = pd.read_csv(subset_file)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        bwm_df = bwm_df[bwm_df.eid.isin(subset[col].astype(str))]
    jobs = []
    for eid, grp in bwm_df.groupby('eid', sort=False):
        jobs.append((eid, list(grp.pid), list(grp.probe_name),
                     grp.subject.iloc[0], grp.lab.iloc[0]))
```

```python
    sess_loader = SessionLoader(one=one, eid=eid)
    sess_loader.load_trials()
    ...
    for pid, pname in zip(pids, probe_names):
        ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. From the final summary: the freeze file is what `0_data_caching.py` itself uses to enumerate the brain-wide map, so using it reproduces the reference's session set. On the cache repair the AI wrote: *"The shipped ONE cache tables are stale relative to the file tree … ONE in local mode silently fails to find them — 459 of 461 sessions would have loaded an empty trials table."* It further noted it preferred the table row that describes the on-disk file because *"the superseded rows carry the `CRITICAL` QC flag that caused the replacement, which would otherwise make ONE skip the good wheel data in 37 sessions."*

## 1-b. How are the data split into subjects?

i. No parsing is needed: `bwm_release.csv` carries a `subject` column (and a `lab` column), which is attached to every job and carried through to the result dict. At assembly, `subjects` is built in first-encounter order over the eid-sorted results, and `subject_idx` records each session's index into that list. Result: 136 subjects over the 440 surviving sessions.

ii.
```python
        jobs.append((eid, list(grp.pid), list(grp.probe_name),
                     grp.subject.iloc[0], grp.lab.iloc[0]))
```

```python
        if res['subject'] not in subjects:
            subjects.append(res['subject'])
        subject_idx.append(subjects.index(res['subject']))
```

iii. No explicit justification is given; the release freeze file already provides a unique subject id per session, so nothing has to be derived.

## 1-c. How are the data split into sessions?

i. The session is the natural unit. `bwm_release.csv` has one row per probe insertion, so it is grouped by `eid` to give one job per session (459 sessions from 699 insertions); the jobs are then sorted by `eid` for deterministic output ordering, and each session is converted independently in a worker process. Probes belonging to the same session are merged into a single population rather than treated as separate sessions.

ii.
```python
    for eid, grp in bwm_df.groupby('eid', sort=False):
        jobs.append((eid, list(grp.pid), list(grp.probe_name),
                     grp.subject.iloc[0], grp.lab.iloc[0]))
    jobs.sort(key=lambda j: j[0])
```

```python
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
```

iii. In the final message: *"Probes from the same session share the same behaviour and are not statistically independent, so the reference merges them into one population"* (mirroring `merge_probes` in `ibl_data_utils.py`).

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split is given by the data. Each trial becomes a 2 s window around its own `stimOn_times`, from −0.5 s to +1.5 s, and every stream (spikes, wheel, whisker) is cut to that same window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

```python
    sess_loader.load_trials()
    trials = sess_loader.trials
    keep = trials_mask(trials)
    ...
    align_times = trials[ALIGN_TIME].to_numpy()
```

iii. The module docstring states the window and bin size are *"the `params` block of `0_data_caching.py`"* — i.e. `{'interval_len': 2, 'binsize': 0.02, 'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`, copied verbatim from the reference.

## 1-e. How are trials filtered based on quality controls?

i. Two stages.

Stage 1 reproduces the reference `ibl_data_utils.load_trials_and_mask(..., max_trial_len=10.0)` exactly — the same pandas `eval` query string is rebuilt: reaction time (`firstMovement_times - stimOn_times`) must lie in [0.08, 2.0] s; trial length (`feedback_times - goCue_times`) must be ≤ 10 s; none of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` may be NaN; and `choice == 0` (no response) is dropped. `exclude_unbiased` is left False, so the initial 50:50 block is retained.

Stage 2 is a behavioural-coverage requirement: a trial is kept only if both the wheel trace and the whisker motion-energy trace span the full 2 s window to within one bin at each edge (the reference's `get_behavior_per_interval` "starts too late"/"ends too early" checks). A session with fewer than 2 trials at either stage is dropped entirely.

ii.
```python
MIN_RT, MAX_RT = 0.08, 2.0
MAX_TRIAL_LEN = 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

def trials_mask(trials):
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return (~trials.eval(query)).to_numpy()
```

```python
    keep = trials_mask(trials)
    if keep.sum() < 2:
        return {'eid': eid, 'skip': f'only {int(keep.sum())} trials pass the criteria'}
```

```python
    mask = i1 > i0
    first = times[np.clip(i0, 0, len(times) - 1)]
    last = times[np.clip(i1 - 1, 0, len(times) - 1)]
    mask &= np.abs(begs - first) <= BINSIZE
    mask &= np.abs(ends - last) <= BINSIZE
    mask &= np.isfinite(align_times)
```

```python
    ok = wheel_mask & whisker_mask
    if ok.sum() < 2:
        return {'eid': eid, 'skip': f'only {int(ok.sum())} trials have complete behaviour'}
```

iii. The docstring of `trials_mask` says it is the *"Reference `load_trials_and_mask` criteria"* and notes *"The unbiased 50:50 block at the start of the session is kept (`exclude_unbiased=False` in the reference), which is what makes the prior a three-valued target."* The final message adds: *"Trials — reference `load_trials_and_mask(max_trial_len=10.0)` … Verified identical to the reference mask."* Result: 187,667 trials over 440 sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, one call per probe insertion listed in `bwm_release.csv`. The merged cluster table (`merge_clusters(...).to_df()`) supplies `label` (the QC score) and `acronym` (the anatomical location), used only to decide which clusters are kept and to fill `brain_region_idx`; the array itself is built from the two spike arrays.

ii.
```python
    for pid, pname in zip(pids, probe_names):
        ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        spikes, clusters, channels = ssl.load_spike_sorting()
        if len(spikes) == 0:
            continue
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        spikes_list.append(spikes)
        clusters_list.append(clusters)
```

```python
    spike_times = np.concatenate(merged_times)
    spike_clusters = np.concatenate(merged_ids)
```

iii. Implicit — this is the same pair of arrays `ibl_data_utils.prepare_data` puts into `neural_dict` as `'spike_times'` / `'spike_clusters'`.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 non-overlapping 20 ms bins per trial. The counts are **not** divided by bin width — the stored values are raw spike counts per 20 ms bin (stored as float32), matching what the reference `bin_spiking_data`/`bincount2D` produces. When a session has two probes their cluster tables are concatenated with an index offset and the spike arrays merged, then sorted by spike time, so the session is one pooled population. Non-finite spike times are dropped. A spike falling exactly on the window edge is clipped into the valid bin range. No smoothing or normalisation is applied.

ii.
```python
    offset = 0
    merged_clusters, merged_times, merged_ids = [], [], []
    for spikes, clusters in zip(spikes_list, clusters_list):
        merged_times.append(spikes['times'])
        merged_ids.append(spikes['clusters'].astype(np.int64) + offset)
        offset += int(clusters.index.max()) + 1
        merged_clusters.append(clusters)
    clusters = pd.concat(merged_clusters, ignore_index=True)
    spike_times = np.concatenate(merged_times)
    spike_clusters = np.concatenate(merged_ids)
    order = np.argsort(spike_times, kind='stable')
    spike_times, spike_clusters = spike_times[order], spike_clusters[order]
    finite = np.isfinite(spike_times)
    spike_times, spike_clusters = spike_times[finite], spike_clusters[finite]
```

```python
    out = np.zeros((len(align_times), n_clusters, N_BINS), dtype=np.float32)
    for k in range(len(align_times)):
        ...
        b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(c * N_BINS + b, minlength=n_clusters * N_BINS)
        out[k] = counts.reshape(n_clusters, N_BINS)
```

iii. The `bin_spikes` docstring: *"Bin i of a trial covers `[align + TIME_WINDOW[0] + i*BINSIZE, ... + BINSIZE)`, the same half-open bins that `bincount2D` produces inside the reference `get_spike_data_per_interval`."* The final message reports the binner was checked against the reference on a full session and was ***"bit-identical (4,012,340 spikes)"***. Merging probes is justified as *"Probes from the same session share the same behaviour and are not statistically independent, so the reference merges them into one population."* The metadata records `'neural_data': 'spike counts per 20 ms bin, Kilosort 2.5 / pykilosort units'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Four cuts, all at the neuron level, taken from the **data paper's** inclusion criteria rather than from the reference code (which passes `qc=None` and keeps every sorted cluster):

1. `clusters.label >= 1.0` — the unit passed all three RIGOR metrics (amplitude, noise cut-off, refractory-period violation).
2. Beryl acronym not in `{'root', 'void'}` — i.e. restricted to Allen grey-matter summary structures.
3. At least 5 surviving units of that Beryl region **within the session**.
4. At assembly, the region must be present in **at least 2 sessions** across the whole dataset; neurons in regions failing this are dropped, and a session left with no neurons is dropped.

Result: 60,483 neurons across 440 sessions (mean 137, min 5, max 508) in 209 Beryl regions; 4 sessions were skipped for having no neuron pass.

ii.
```python
QC_LABEL = 1.0
NON_GREY = {'root', 'void'}
MIN_NEURONS_PER_REGION = 5
MIN_SESSIONS_PER_REGION = 2
```

```python
    beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
    beryl = np.asarray(beryl, dtype=object)
    good = (clusters['label'].to_numpy() >= QC_LABEL) & ~np.isin(beryl, list(NON_GREY))
    # at least five well-isolated neurons per region in this session
    regions, counts = np.unique(beryl[good], return_counts=True)
    enough = set(regions[counts >= MIN_NEURONS_PER_REGION])
    good &= np.array([r in enough for r in beryl])
    cluster_ids = np.nonzero(good)[0]
    if len(cluster_ids) == 0:
        return {'eid': eid, 'skip': 'no neurons pass the inclusion criteria'}
```

```python
    keep_regions = sorted(r for r, n in n_sessions_with_region.items()
                          if n >= MIN_SESSIONS_PER_REGION)
    ...
        keep = np.array([r in region_index for r in res['regions']])
        if keep.sum() == 0:
            skipped.append((res['eid'], 'no neurons in regions recorded in >= 2 sessions'))
            continue
        spikes = res['neural'][:, keep, :]
```

iii. Explicit and detailed. In-code comments quote the data paper: *"Neurons … were excluded if they failed one of the three criteria … amplitude, noise cut-off and refractory period violation"* and *"restricted to regions that were designated grey matter … contained at least five well-isolated neurons per session and were recorded from in at least two such sessions"*. The final message: *"here I followed the data paper rather than the Zhang code, which passes `qc=None` and keeps all ~1350 Kilosort units per session … Rationale: those criteria are the only explicit low-quality-neuron filter in the references, and the decoder here trains jointly over all sessions — the unfiltered version is a ~110 GB dataset, which `decoder.py`'s own comments flag as having OOM-killed a prior run."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `trials.stimOn_times`. All IBL streams are already expressed in seconds on one synchronised session clock, so alignment is just subtraction: for each trial the window `[stimOn - 0.5, stimOn + 1.5)` is located in the sorted spike-time array by `searchsorted`, and the bin index of each spike is `floor((t - (stimOn - 0.5)) / 0.02)`. The same `align_times` vector drives the behavioural grids, so every stream shares the trial's time origin.

ii.
```python
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    ...
        b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)
```

```python
'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
'off_start': TIME_WINDOW[0],
'off_end': TIME_WINDOW[1],
```

iii. The module docstring: *"Trials are aligned to stimulus onset and span -0.5 s to +1.5 s around it"* — the instructions' "Temporally align based on stimulus onset" and the reference `params['align_time'] = 'stimOn_times'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 20.0` (ms). Spikes are binned once, directly at 20 ms, from raw spike times — there is no rebinning or resampling of an intermediate representation. The behavioural traces are resampled (by linear interpolation) onto the same 20 ms grid.

ii.
```python
BINSIZE = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
...
'time_bin_size': BINSIZE * 1000,
'n_timepoints': N_BINS,
```

iii. The docstring: *"binned into 100 non-overlapping 20 ms bins — the `params` block of `0_data_caching.py`"*. The verify summary confirms `T_min = T_max = 100`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Nothing in the raw data beyond the alignment event itself — it is the fixed bin-centre grid of the (−0.5, 1.5) s window relative to `stimOn_times`, so the same 100 values (−0.49 … +1.49 s) are broadcast to every trial of every session.

ii.
```python
    bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
    inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = bin_centres
```

iii. The docstring lists input 0 as *"time since stimulus onset (s), bin centres, -0.49 .. 1.49"*, and the metadata as *"signed time of the bin centre relative to stimulus onset, in seconds"*.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None — it is constructed analytically as `TIME_WINDOW[0] + (arange(100) + 0.5) * 0.02`, i.e. the centre of each 20 ms bin. It is stored as a continuous float32 time series rather than as a binary onset indicator (the instructions ask for a continuous "Time since stimulus onset" input).

ii.
```python
    bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
```

iii. No separate justification; the value is defined by the chosen window and bin size. Verified in the stats: `input_range['time_from_stimulus_onset'] = [-0.49, 1.49]`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural binning grid. Neural bin `i` covers `[stimOn − 0.5 + 0.02·i, stimOn − 0.5 + 0.02·(i+1))`, and input 0 at index `i` is the centre of that same interval, so the two arrays are aligned bin-for-bin by construction and identical across trials and sessions.

ii.
```python
        b = np.floor((t - begs[k]) / BINSIZE).astype(np.int64)     # neural bin index
```
```python
    bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE   # input 0
```

iii. No explicit statement beyond the docstring; the shared `TIME_WINDOW`/`BINSIZE` constants make the correspondence exact.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. `trials.probabilityLeft` alone. The trials table carries no block identifier, so blocks are recovered as maximal runs of constant `probabilityLeft`.

ii.
```python
    block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Implicit in the function docstring: *"Index of each trial within its block of constant `probabilityLeft`"*.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A new block is declared wherever `probabilityLeft` changes (and at index 0); a running maximum of the block-start index gives, for each trial, the index at which its block began; the 0-based position is `arange - block_start`. Crucially this is computed on the **full, unfiltered** trials table and only then indexed by the trial mask, so trials that are later excluded still advance the counter and the value reflects the animal's true position in the block. The value is constant within a trial and broadcast across all 100 bins as a continuous float32. Observed range 0–98.

ii.
```python
def trial_number_in_block(probability_left):
    """Index of each trial within its block of constant probabilityLeft (0-based).

    Computed on the full trials table so that the count is unaffected by trials that
    are later excluded -- the block the mouse experienced does not skip trials.
    """
    p = np.asarray(probability_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    new_block[1:] = p[1:] != p[:-1]
    block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(p)), 0))
    return np.arange(len(p)) - block_start
```

```python
    block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())
    trials = trials[keep]
    block_idx = block_idx[keep]
    ...
    inputs[:, 1, :] = block_idx[:, None]
```

iii. The docstring states the reason directly: *"the block the mouse experienced does not skip trials."*

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1 (leftward report), −1 (rightward) or 0 (no response). The 0 entries never reach this point — they are removed by `trials_mask`.

ii.
```python
    # trials.choice is +1 when the mouse reported the left side and -1 for the right.
    choice = ((1 - trials['choice'].to_numpy()) / 2).astype(np.int8)
```

iii. The final message: *"Choice coding — verified empirically across sessions that `trials.choice == +1` ⟺ left report."*

## 5-b. What processing is involved in computing `output` *Choice*?

i. A pure recoding, `(1 − choice)/2`: +1 → 0 (left), −1 → 1 (right), matching the instructions' "left = 0, right = 1". The per-trial scalar is then broadcast to all 100 bins and stored as int8, so the output is formally time-varying but constant within a trial. `output_values[0] = ['left', 'right']`. Observed class balance 0.508 / 0.492.

ii.
```python
    outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int8)
    outputs[:, 0, :] = choice[:, None]
```

iii. The instructions specify the encoding; the docstring records *"choice  left=0, right=1  (constant within a trial)"*, and the metadata *"reported stimulus side: left=0, right=1 (trials.choice +1/-1)"*.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table (the uncued block prior). NaN values are already excluded by `trials_mask`.

ii.
```python
PRIOR_VALUES = np.array([0.2, 0.5, 0.8])
...
    prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
                   - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
```

iii. Implicit; the metadata describes it as *"block prior probability that the stimulus appears on the left: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2"*.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Each value is mapped to the index of the nearest of `[0.2, 0.5, 0.8]` (rather than a strict dictionary lookup), giving 0/1/2 as the instructions require. The unbiased 50:50 block at the start of each session is deliberately retained, which is what makes this a 3-class rather than 2-class target. The per-trial scalar is broadcast to all 100 bins as int8. Observed fractions: 0.417 / 0.140 / 0.442.

ii.
```python
    prior = np.abs(trials['probabilityLeft'].to_numpy()[:, None]
                   - PRIOR_VALUES[None, :]).argmin(axis=1).astype(np.int8)
    outputs[:, 1, :] = prior[:, None]
```

```python
    """... The unbiased 50:50 block at the start of the session is
    kept (``exclude_unbiased=False`` in the reference), which is what makes the
    prior a three-valued target.
    """
```

iii. As quoted above: keeping the unbiased block follows the reference's `exclude_unbiased=False` default and is needed for a three-valued target.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `sess_loader.wheel['times']` and `sess_loader.wheel['velocity']` from `SessionLoader.load_wheel()`, i.e. the wheel derived from `_ibl_wheel.position` / `_ibl_wheel.timestamps`. Speed is the absolute value of velocity.

ii.
```python
    sess_loader.load_wheel()
    wheel_speed, wheel_mask = bin_behavior(
        sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. Implicit — this is how the reference builds `'wheel-speed'` (`np.abs` of the `SessionLoader` velocity).

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` does the upstream work: interpolate the irregularly-sampled wheel position onto a 1000 Hz grid and differentiate it with a 20 Hz Butterworth low-pass, yielding velocity in rad/s. The AI then takes `|velocity|`, drops non-finite samples, sorts by time, and linearly interpolates the trace onto the per-trial grid (100 points running from `interval_beg + BINSIZE` to `interval_end`, i.e. the right edge of each bin — the reference's `np.linspace(beg + binsize, end, n_bins)`). Interpolation is done against the **full** session trace rather than the within-interval slice, so edge bins are interpolated rather than extrapolated. Finally the per-session tertile discretisation is applied.

ii.
```python
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]

    finite = np.isfinite(values)
    times, values = times[finite], values[finite]
    order = np.argsort(times, kind='stable')
    times, values = times[order], values[order]
    ...
    binned = np.interp(grid.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. The `bin_behavior` docstring: *"The grid is the reference one from `get_behavior_per_interval`… Interpolated from the full trace rather than the within-interval slice, so the edge bins are interpolated instead of extrapolated."* And on NaNs: *"NaN samples are dropped before the coverage check rather than tolerated as the reference's `allow_nans=True` does, because the decoder needs a finite class label in every bin."* The final message adds that against the reference the values *"differ only in the final bin per trial … (tertile labels agree 100% / 99.6%)."*

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 equal-occupancy bins (tertiles) computed from **that session's own** kept trials: the 1/3 and 2/3 quantiles of the pooled (trials × bins) speed values form the two edges, and `searchsorted(..., side='right')` assigns 0/1/2 (`['low','medium','high']`). Because the thresholds are per-session and the pool is all bins of all kept trials, the classes come out almost exactly balanced (0.3333 / 0.3333 / 0.3333 in the verify stats).

ii.
```python
N_OUTPUT_BINS = 3  # tertiles for the two continuous behaviours

def tertile_labels(values):
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)
...
    outputs[:, 2, :] = tertile_labels(wheel_speed)
```

iii. The docstring: *"Session-level quantiles rather than global ones: whisker motion energy is in camera-dependent arbitrary units and the wheel is used at very different rates by different mice, so a common threshold would mean 'low'/'high' referred to different behaviour in different sessions, while the decoder's output heads are shared across sessions."*

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The grid is built from the same `align_times` (`stimOn_times`) and the same `TIME_WINDOW`/`BINSIZE` constants as the spike binning, giving 100 samples, one per neural bin. The samples sit at the **right edge** of each 20 ms bin (`beg + 0.02·(i+1)`) rather than the centre — a half-bin (10 ms) offset relative to the neural bin centre, adopted to match the reference `get_behavior_per_interval`. Trials where the wheel does not cover the window to within one bin at either edge are dropped from *all* streams, so the neural, input and output arrays always have the same trial count.

ii.
```python
    grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
```

```python
    ok = wheel_mask & whisker_mask
    binned_spikes = binned_spikes[ok]
    wheel_speed, whisker = wheel_speed[ok], whisker[ok]
    trials, block_idx = trials[ok], block_idx[ok]
```

iii. As above — the grid is *"the reference one from `get_behavior_per_interval`: N_BINS points running from `interval_start + BINSIZE` to `interval_end`, i.e. the right edge of each bin"*, and the coverage rejection is *"the same check the reference applies."*

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy` with its frame times `_ibl_<side>Camera.times`, loaded via `SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy` column. The left camera is tried first and the right used as a fallback; a session with neither is dropped (14 sessions were skipped for this reason).

ii.
```python
    whisker, whisker_mask = None, None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            me = sess_loader.motion_energy[cam]
            whisker, whisker_mask = bin_behavior(
                me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
            break
        except Exception:
            continue
    if whisker is None:
        return {'eid': eid, 'skip': 'no whisker motion energy available'}
```

iii. In-code comment: *"The reference prefers the left camera and falls back to the right one."* Metadata: *"whisker pad motion energy (left camera, right camera as fallback)"*.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is (no filtering or normalisation) — it goes through the identical `bin_behavior` path as the wheel: non-finite samples dropped, sorted by time, linearly interpolated onto the 100-point per-trial grid at bin right edges, with the same one-bin coverage requirement, then per-session tertiles.

ii.
```python
            whisker, whisker_mask = bin_behavior(
                me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy(), align_times)
```

```python
    outputs[:, 3, :] = tertile_labels(whisker)
```

iii. Same justification as the wheel (shared `bin_behavior` docstring); no additional processing is argued for.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: 3 equal-occupancy classes at the 1/3 and 2/3 quantiles of that session's own pooled (trials × bins) values, labelled `['low','medium','high']`. Observed fractions 0.3326 / 0.3326 / 0.3348 (slightly off exactly balanced because motion energy has ties/repeated values).

ii.
```python
    outputs[:, 3, :] = tertile_labels(whisker)
```
```python
def tertile_labels(values):
    edges = np.quantile(values, np.arange(1, N_OUTPUT_BINS) / N_OUTPUT_BINS)
    return np.searchsorted(edges, values, side='right').astype(np.int8)
```

iii. The `tertile_labels` docstring makes the camera-specific argument explicitly: *"whisker motion energy is in camera-dependent arbitrary units … a common threshold would mean 'low'/'high' referred to different behaviour in different sessions, while the decoder's output heads are shared across sessions."*

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: built from the same `align_times` (`stimOn_times`) and `TIME_WINDOW`/`BINSIZE`, 100 samples at the right edge of each neural bin (a 10 ms offset from the bin centre), with camera frame times already on the shared session clock. Trials whose camera trace does not span the window to within one bin are dropped from all streams via the combined `ok` mask.

ii.
```python
    grid = begs[:, None] + (np.arange(1, N_BINS + 1) * BINSIZE)[None, :]
    ...
    mask &= np.abs(begs - first) <= BINSIZE
    mask &= np.abs(ends - last) <= BINSIZE
```
```python
    ok = wheel_mask & whisker_mask
```

iii. Same as 7-d.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Layered and defensive:
- **Stale cache tables** — repaired wholesale by `build_patched_tables` (re-pointing rows at on-disk revisions, re-stat'ing sizes, dropping stale hashes, forcing `default_revision`), including preferring the table row that describes the file actually on disk so a superseded `CRITICAL` QC flag does not suppress good wheel data.
- **Missing trial events** — the six NaN-excluded columns in `trials_mask` drop those trials.
- **Missing/short behavioural coverage** — the one-bin coverage check drops those trials; NaN samples are removed from the trace before the check.
- **Non-finite spike times** — filtered out after merging probes.
- **Edge-of-window spikes** — bin index clipped into `[0, N_BINS-1]`.
- **A probe with no spike sorting** — skipped (`if len(spikes) == 0: continue`); a session with no probe at all is skipped.
- **No camera / no neurons / <2 usable trials** — the session is skipped and the reason recorded.
- **Any other failure** — caught per session with a traceback, the session skipped, and the run continues; all 19 skip reasons are written into `metadata['skipped_sessions']`.

ii.
```python
def process_session(args):
    eid, pids, probe_names, subject, lab = args
    try:
        return _process_session(eid, pids, probe_names, subject, lab)
    except Exception as e:  # a session that cannot be loaded is reported and skipped
        return {'eid': eid, 'skip': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}
```

```python
    finite = np.isfinite(spike_times)
    spike_times, spike_clusters = spike_times[finite], spike_clusters[finite]
```

```python
    finite = np.isfinite(values)
    times, values = times[finite], values[finite]
    ...
    mask &= np.isfinite(binned).all(axis=1)
```

```python
            'skipped_sessions': skipped,
```

iii. From the docstrings/message: NaNs in behaviour are dropped rather than tolerated *"because the decoder needs a finite class label in every bin"*; the cache repair is justified because otherwise *"459 of 461 sessions would have loaded an empty trials table"*; and the final message enumerates the 19 dropped sessions (*"14 have no camera data at all, 4 have no neurons passing QC, 1 has <2 trials with complete behaviour"*).

## 10-a. What are the most time-consuming steps of the code?

i. Measured: the whole conversion ran in **160 s** with 32 worker processes. The dominant costs are (1) reading the spike sorting off disk — `ssl.load_spike_sorting()` per probe, hundreds of MB of `spikes.times`/`spikes.clusters` per insertion; (2) `build_patched_tables`, which `os.walk`s every one of ~480 session directories and re-`stat`s 16,607 files (one-off, ~seconds, but serial before the pool starts); (3) writing the 10.8 GB pickle at the end, which is single-threaded; and (4) the per-trial spike-binning loop in `bin_spikes`.

ii.
```python
        spikes, clusters, channels = ssl.load_spike_sorting()
```

```python
    for root, _dirs, files in os.walk(session_dir):
```

```python
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Not discussed explicitly in the trajectory. The AI did address throughput by parallelising across sessions with a 32-process pool and by not passing `check_hash`-style re-reads; it flagged dataset *size* (rather than time) as the binding constraint: *"the unfiltered version is a ~110 GB dataset, which `decoder.py`'s own comments flag as having OOM-killed a prior run."*

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, none of them hot enough to matter at 160 s total:
- `bin_spikes` loops over trials. Since spikes are sorted and every trial has the same 100-bin grid, this could be one `np.bincount` over a flat `(trial, unit, bin)` index for the whole session. (The AI already vectorised *within* a trial.)
- `good &= np.array([r in enough for r in beryl])` and `keep = np.array([r in region_index for r in res['regions']])` are Python membership loops over every cluster; `np.isin` would do both.
- `build_patched_tables` loops over sessions and over `(collection, filename)` keys in Python; the `file_size` list comprehension `stat()`s 16,607 files one at a time.
- `neural.append([spikes[k] for k in range(spikes.shape[0])])` (and the same for inputs/outputs) unpacks a 3-D array into a Python list of 2-D slices — unavoidable given the required output format, but it is a per-trial Python loop.

ii.
```python
    out = np.zeros((len(align_times), n_clusters, N_BINS), dtype=np.float32)
    for k in range(len(align_times)):
```

```python
    good &= np.array([r in enough for r in beryl])
```

```python
    datasets['file_size'] = [
        (session_dirs[eid] / rel).stat().st_size
        for eid, rel in zip(datasets['eid'], datasets['rel_path'])]
```

iii. The AI describes its binner as *"my vectorised spike binner"* and verified it against the reference as bit-identical, so it considered the remaining per-trial loop acceptable; no further vectorisation is discussed.

## 10-c. What processing does the code repeat multiple times?

i.
- `build_patched_tables()` is called once in `main()` and then again inside `get_one()` in **every** worker process (32 workers × 459 sessions' worth of calls). It short-circuits on `PATCHED_DIR.exists()`, so the expensive rebuild happens once, but the check and the ONE client construction are repeated per session.
- `get_one()` is called inside `_process_session`, i.e. once per session rather than once per worker — the ONE client and its cache tables are re-loaded 459 times instead of 32.
- `BrainRegions()` is instantiated fresh inside every session (it loads the Allen atlas tables each time) instead of once per process.
- `bin_behavior` interpolates **all** trials, including those that failed the coverage mask, and `bin_spikes` bins **all** mask-stage-1 trials, including those later removed by the behavioural mask.
- The Beryl mapping is applied to the whole cluster table, and the region-membership scan over clusters is done twice (per-session `enough` test, then again at assembly for `keep_regions`).

ii.
```python
def get_one(force_rebuild=False):
    tables_dir = build_patched_tables(force=force_rebuild)
    return ONE(...)
```

```python
def _process_session(eid, pids, probe_names, subject, lab):
    from brainbox.io.one import SessionLoader, SpikeSortingLoader
    from iblatlas.regions import BrainRegions
    one = get_one()
```

```python
    beryl = BrainRegions().acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
```

iii. Not discussed. The AI did consolidate the cache fix from a separate script into `convert_data.py` (*"Let me consolidate the cache fix into `convert_data.py` so the conversion is one self-contained script"*), which is what introduced the per-worker re-entry into `build_patched_tables`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Spikes binned for trials that are then thrown away.** `bin_spikes` runs over all `keep`-stage trials; only afterwards is the behavioural mask `ok` applied and `binned_spikes = binned_spikes[ok]`. The same holds for the interpolation in `bin_behavior`, which fills the grid for every trial including ones whose `mask` is already False.
- **Neurons binned and then discarded.** `bin_spikes` is given only `cluster_ids`, so the per-session QC is applied first — but the *cross-session* region filter (`>= 2 sessions`) is applied at assembly, so those neurons' spike counts were binned, held in memory for the whole run, and then dropped by `res['neural'][:, keep, :]`.
- **Full `merge_clusters`** computes/attaches all cluster metrics and histology per probe, while only `label` and `acronym` are ever used.
- **Continuous behaviour values** are computed at float precision and then collapsed to 3 classes; the continuous traces are not retained.
- **Neural counts stored as float32** — the values are small non-negative integers, so int16 would halve the 10.8 GB file with no information loss.
- **Unused metadata** carried through the pipeline: `lab`, `n_probes`, `n_trials_in_season`/`n_trials_total`, `traceback` strings, and the full `skipped_sessions` list — harmless but not used by the decoder.

ii.
```python
    binned_spikes = bin_spikes(spike_times, spike_clusters, cluster_ids, align_times)
    ...
    ok = wheel_mask & whisker_mask
    ...
    binned_spikes = binned_spikes[ok]
```

```python
        keep = np.array([r in region_index for r in res['regions']])
        ...
        spikes = res['neural'][:, keep, :]
```

```python
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
    out = np.zeros((len(align_times), n_clusters, N_BINS), dtype=np.float32)
```

iii. Not discussed in the trajectory. The AI's only size-related reasoning is the neuron-inclusion argument (*"the unfiltered version is a ~110 GB dataset"*), which is about how many neurons to keep rather than about redundant work.
