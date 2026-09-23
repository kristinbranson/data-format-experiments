# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the session list from the reference code's `bwm_release.csv` file (a CSV shipped with `code_zhang2025`), not from the ONE API release index. It restricts to sessions in `DATALIMIT_SUBSET.csv` when that file exists. For each session, it uses the reference utility functions from `utils.ibl_data_utils` (the vendored reference code) to load spikes, trials, and behavior. An `ONE` client is created pointing at the local cache directory. Each session is processed sequentially in a loop.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
if os.path.exists(DATALIMIT):
    subset = pd.read_csv(DATALIMIT)
    col = 'eid' if 'eid' in subset.columns else subset.columns[0]
    allowed = set(subset[col].astype(str))
    eids = [e for e in eids if e in allowed]
```

```python
one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir='/app/data/one_cache')
```

iii. The agent stated: "Sessions: all sessions of the BWM public release freeze shipped with the reference code (data/bwm_release.csv; 459 sessions, 139 mice), in order of first appearance in that freeze." The agent explicitly chose to reuse the reference code's utilities rather than writing data loading from scratch.

## 1-b. How are the data split into subjects?

i. Subject names are extracted from `bwm_release.csv` via `eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))`. Subjects are accumulated in a list as sessions are processed, and `subject_idx` maps each session to its position in the list.

ii.
```python
eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))
...
sub = eid2subject[eid]
if sub not in subjects:
    subjects.append(sub)
data['subject_idx'].append(subjects.index(sub))
```

iii. The subject names come directly from the CSV; the AI did not need to parse paths or filenames.

## 1-c. How are the data split into sessions?

i. Each row in the `bwm_release.csv` corresponds to a probe insertion; sessions are identified by unique `eid` values. The AI deduplicates via `dict.fromkeys(bwm_df.eid.tolist())` to get one entry per session.

ii.
```python
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
```

iii. Sessions are already the unit the release is organized by; the AI just deduplicates the per-probe rows.

## 1-d. How are the data split into trials?

i. Trials are loaded via the reference utility `U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)`, which returns a trials DataFrame with one row per trial and a quality mask.

ii.
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
```

iii. The trials table already has one row per trial; no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Two stages. First, the reference utility `load_trials_and_mask(max_trial_len=10.0)` is called, which filters: reaction time (firstMovement_times - stimOn_times) must be in [0.08, 2.0] s, trial length (feedback_times - goCue_times) must be <= 10 s, NaN exclusion on 6 fields (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), and no-choice trials (choice == 0) are excluded. Second, the AI additionally drops trials whose wheel-speed or whisker-motion-energy behavioral trace is None, has the wrong number of bins, or contains NaN values. Sessions with fewer than 2 surviving trials are skipped.

ii.
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
trials_mask = np.asarray(trials_mask).astype(bool)
...
good = trials_mask.copy()
for beh in BEH_NAMES:
    traces = binned_beh[beh]
    ok = np.array([
        (tr is not None) and (np.asarray(tr).size == NBINS)
        and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
        for tr in traces])
    good &= ok
keep = np.flatnonzero(good)
if len(keep) < MIN_TRIALS:
    raise RuntimeError(f'only {len(keep)} trials survive curation')
```

iii. The agent stated: "Trial curation: load_trials_and_mask(max_trial_len=10.0), exactly as reference prepare_data." The additional NaN check was justified: "a NaN cannot be assigned a category, and the decoder rejects non-finite values."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` and `spikes.clusters`, loaded via `SpikeSortingLoader.load_spike_sorting()`. Cluster metadata (including `acronym` for brain regions) is obtained via `SpikeSortingLoader.merge_clusters()`.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
```

iii. Spike times and cluster assignments are the standard raw variables for constructing neural activity matrices.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms bins over the 2 s trial window (-0.5 to 1.5 s around stimulus onset), giving spike counts per neuron per bin. The data is stored as spike counts (NOT converted to firing rates in Hz). When a session has multiple probes, they are merged via `U.merge_probes`. The binning is done by the reference utility `U.bin_spiking_data`.

ii.
```python
spikes, clusters = U.merge_probes(spikes_list, clusters_list)
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
...
neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
```

iii. The agent stated the metadata records `'neural_units': 'spike counts per 20 ms bin'`. The agent explicitly chose spike counts rather than firing rates, following the reference code's `bin_spiking_data` which outputs counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. NO quality filtering is applied to clusters. The AI uses `qc=None` (all clusters kept regardless of quality label), matching the reference code's `prepare_data` which calls `load_spiking_data` without a `qc` argument. The agent's custom `load_clusters_and_spikes` function does not filter by cluster label. Clusters labeled `void` or `root` are NOT specifically excluded at this stage (Beryl mapping is applied for region names but not used to filter out void clusters).

ii.
```python
def load_clusters_and_spikes(one, pid, eid, pname):
    """...Same as utils.ibl_data_utils.load_spiking_data with qc=None..."""
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    ...
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
    return spikes, clusters_labeled
```

iii. The agent explicitly justified: "Neural: pykilosort spike-sorted spikes of ALL clusters (reference prepare_data calls load_spiking_data with qc=None, i.e. no unit-quality threshold; methods paper: 'we bin spike counts using all neurons ... from each session')."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The spike data is aligned to stimulus onset (`stimOn_times`) as specified in the PARAMS configuration. The reference utility `bin_spiking_data` takes `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)` and bins spikes relative to each trial's stimulus onset time.

ii.
```python
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
```

iii. The agent stated: "Alignment: stimOn_times, window (-0.5, +1.5) s, non-overlapping 20 ms bins -> T = 100. These are exactly the params of 0_data_caching.py."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, producing 100 bins over the 2 s window. No rebinning or resampling is applied; the spikes are binned directly into 20 ms bins by the reference utility.

ii.
```python
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    ...
}
NBINS = int(round((PARAMS['time_window'][1] - PARAMS['time_window'][0])
                  / PARAMS['binsize']))  # 100
```

iii. The agent noted: "The 20 ms bin size (rather than the 50 ms the paper uses for the purely static choice/prior targets) is required here because this task also asks for time-varying wheel-speed and whisker-motion-energy outputs, for which paper and code use 20 ms."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time grid is computed as the right edge of each 20 ms bin relative to stimulus onset.

ii.
```python
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
```

iii. The agent stated the time grid represents "signed time (s) of the end of each 20 ms bin relative to stimulus onset: -0.48 ... +1.50."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time grid is constructed as the right edge of each bin: `(np.arange(1, NBINS+1) * 0.02) + (-0.5)`, giving values from -0.48 to +1.50 s. This is the same for every trial.

ii.
```python
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
```

iii. The agent described this as "the grid the reference code interpolates the behavioural traces onto."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time grid represents the right edges of the same bins that the neural spikes are counted into, so they share the same temporal structure by construction.

ii.
```python
inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. The agent described alignment as inherent since both use the same bin structure.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where consecutive `probabilityLeft` values differ.

ii.
```python
def trial_number_in_block(prob_left):
    p = np.asarray(prob_left, dtype=float)
    idx = np.zeros(len(p), dtype=np.float32)
    counter = 0
    for i in range(len(p)):
        if i > 0 and np.isclose(p[i], p[i - 1]):
            counter += 1
        else:
            counter = 0
        idx[i] = counter
    return idx
```

iii. The agent noted that `probabilityLeft` is constant within a block, so a change marks a new block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter increments within each block. The counter resets to 0 when `probabilityLeft` changes (detected via `np.isclose`). The computation is done on the full (unfiltered) trial array, then indexed to retained trials, so the count reflects the animal's true position in the block.

ii.
```python
tinb = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())[keep]
```

iii. The agent computed block number before filtering so that dropped trials still advance the count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table. IBL convention: +1 = leftward choice, -1 = rightward choice, 0 = no response.

ii.
```python
choice = trials_df['choice'].to_numpy()[keep]
choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
```

iii. The agent stated: "IBL convention: choice == +1 -> leftward turn, choice == -1 -> rightward turn."

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded from IBL convention to decoder format: +1 (left) -> 0, -1 (right) -> 1. No-choice trials (choice == 0) were already filtered out by `load_trials_and_mask`.

ii.
```python
choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
```

iii. Matches the instruction specification "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
...
pleft = trials_df['probabilityLeft'].to_numpy()[keep]
prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft], dtype=np.int64)
```

iii. Matches the instruction specification "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping from the three probability values to integers 0, 1, 2. A `round(float(p), 1)` is applied to handle floating point imprecision.

ii.
```python
prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft], dtype=np.int64)
```

iii. No additional processing beyond the mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps (`_ibl_wheel.position` and `_ibl_wheel.timestamps`), loaded by the reference utility via `U.bin_behaviors(one, eid, ['wheel-speed'], ...)`. Internally, `SessionLoader` interpolates the position to 1000 Hz, applies a Butterworth low-pass filter, and differentiates to get velocity. The speed is the absolute value of this velocity.

ii.
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
...
wheel = np.stack([np.asarray(binned_beh['wheel-speed'][i], dtype=float).ravel()
                  for i in keep])
```

iii. The agent relied on the reference utility's wheel-speed implementation.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel speed is loaded and binned by the reference utility `bin_behaviors`, which interpolates it onto the trial time grid. Then it is discretized into 3 bins using within-session tertiles (33.3rd and 66.7th percentiles of all retained samples).

ii.
```python
def discretize_tertiles(values):
    edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    return np.searchsorted(edges, values, side='right').astype(np.int64)
...
wheel_out = discretize_tertiles(wheel)
```

iii. The agent justified session-level tertiles: "Both signals are heavy-tailed and in session-specific units... so fixed global thresholds would collapse whole sessions into one class."

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel speed is discretized into 3 categories {0, 1, 2} (labeled low/medium/high) at the within-session 33.3rd and 66.7th percentiles using `np.quantile` and `np.searchsorted`.

ii.
```python
edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
return np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. Session-level tertiles ensure roughly equal class sizes within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The reference utility `bin_behaviors` interpolates the wheel speed onto the same time grid as the neural data (same bin edges, same alignment to stimulus onset), so the two are aligned by construction.

ii.
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
```

iii. Both neural and behavioral data use the same PARAMS configuration for alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from a side camera ROI (`leftCamera.ROIMotionEnergy` or `rightCamera.ROIMotionEnergy`), loaded by the reference utility via `U.bin_behaviors(one, eid, ['whisker-motion-energy'], ...)`. The utility internally selects between left and right camera.

ii.
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
...
whisk = np.stack([np.asarray(binned_beh['whisker-motion-energy'][i],
                             dtype=float).ravel() for i in keep])
```

The session-level check for camera availability:
```python
for beh, targets in (('wheel-speed', ['wheel-speed']),
                     ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                'right-whisker-motion-energy'])):
    if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
        raise RuntimeError(f'{beh} is not available for this session')
```

iii. The agent noted that 14 of 16 skipped sessions lacked whisker motion energy entirely.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy trace is loaded and interpolated onto the trial time grid by the reference utility `bin_behaviors`. It is then discretized into 3 bins using within-session tertiles, the same method as wheel speed.

ii.
```python
whisk_out = discretize_tertiles(whisk)
```

iii. Same discretization approach as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 categories {0, 1, 2} at the within-session 33.3rd and 66.7th percentiles.

ii.
```python
edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
return np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. Consistent with the wheel speed approach for interpretability.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: the reference utility `bin_behaviors` interpolates the motion energy onto the same time grid as the neural data.

ii.
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
```

iii. Both share the same PARAMS configuration for temporal alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Sessions without whisker motion energy (neither left nor right camera) are skipped entirely. (2) Sessions where a probe has no spike sorting raise an error and are skipped. (3) Trials whose behavioral traces are None, wrong length, or contain NaN are dropped. (4) Sessions with fewer than 2 surviving trials are skipped. (5) Any unexpected exception during session processing is caught and the session is skipped with a log message. Final tally: 443 sessions kept, 16 skipped out of 459.

ii.
```python
try:
    sess = load_session(one, eid, args.n_workers)
except Exception as exc:
    print(f'  SKIPPED {eid}: {exc!r}', flush=True)
    skipped.append({'eid': eid, 'reason': repr(exc)})
    continue
```

```python
if clusters is None:
    raise RuntimeError(f'no spike sorting for probe insertion {pid} ({pname})')
```

```python
good = trials_mask.copy()
for beh in BEH_NAMES:
    traces = binned_beh[beh]
    ok = np.array([
        (tr is not None) and (np.asarray(tr).size == NBINS)
        and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
        for tr in traces])
    good &= ok
```

iii. The agent noted: "16 skipped: 14 lack whisker motion energy entirely, 1 has zero trials passing curation, and 1 failed only because of the unused raw-ephys sampling-frequency lookup."

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting data from disk (large arrays of spike times and cluster assignments), and the spike binning step via `U.bin_spiking_data` which uses multiprocessing (`n_workers`). Sessions are processed sequentially in the main loop (not parallelized across sessions).

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
```

iii. The spike sorting files are hundreds of megabytes per probe, making I/O the bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_number_in_block` function uses a Python for-loop over all trials to compute the block index. This could be vectorized using cumulative sum over a boolean mask of block boundaries (as the reference solution does with pandas `groupby.cumcount`). The trial-level loop for constructing input/output arrays could also potentially be vectorized.

ii.
```python
def trial_number_in_block(prob_left):
    p = np.asarray(prob_left, dtype=float)
    idx = np.zeros(len(p), dtype=np.float32)
    counter = 0
    for i in range(len(p)):
        if i > 0 and np.isclose(p[i], p[i - 1]):
            counter += 1
        else:
            counter = 0
        idx[i] = counter
    return idx
```

iii. The loop is over a small number of trials per session so performance impact is negligible.

## 10-c. What processing does the code repeat multiple times?

i. The reference utility `list_brain_regions` and `select_brain_regions` are called to get cluster IDs, and then `bin_spiking_data` is called, which internally re-processes the spike data. The behavioral availability check via `load_target_behavior` is called separately from `bin_behaviors`, meaning the behavior loading code may run twice for each behavioral variable.

ii.
```python
# First call to check availability:
if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
    raise RuntimeError(...)
# Second call that actually bins:
binned_beh, _ = U.bin_behaviors(one, eid, BEH_NAMES, ...)
```

iii. The duplicate behavioral loading is a minor inefficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `cluster_regions` for all clusters including `void` and `root` regions, which some downstream analyses would exclude. The metadata includes extensive session info, skipped session details, and other fields that are not used by the decoder. The `n_workers` parameter for multiprocessing within each session adds overhead for what could be a simpler sequential computation.

ii.
```python
data['metadata'] = {
    ...
    'skipped_sessions': skipped,
    'session_info': session_info,
    ...
}
```

iii. The extra metadata is informational rather than strictly necessary for decoding.
