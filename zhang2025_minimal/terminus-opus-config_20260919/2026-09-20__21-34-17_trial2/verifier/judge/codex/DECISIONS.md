# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not enumerate sessions by querying the local ONE release index for required datasets. Instead, it reads the reference release manifest `bwm_release.csv`, keeps the first occurrence of each `eid`, optionally intersects that list with `DATALIMIT_SUBSET.csv`, and then loads each session on demand through `ONE`, `SpikeSortingLoader`, and helper functions in `utils.ibl_data_utils`.

ii. 
```python
BWM_RELEASE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT = '/app/data/DATALIMIT_SUBSET.csv'
...
one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir='/app/data/one_cache')

bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
if os.path.exists(DATALIMIT):
    subset = pd.read_csv(DATALIMIT)
    ...
    eids = [e for e in eids if e in allowed]
```

```python
for k, eid in enumerate(eids):
    ...
    sess = load_session(one, eid, args.n_workers)
```

iii. In the trajectory the AI said the task-specific loading should follow the vendored reference code, but that it would use the public release freeze in `bwm_release.csv` as the session list and load sessions lazily through ONE. It justified the optional subset restriction as needed when `DATALIMIT_SUBSET.csv` is present.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `subject` column of `bwm_release.csv`. Sessions are assigned to subjects by `eid2subject`, and the final `subjects` list is built in first-seen order while iterating through kept sessions.

ii. 
```python
eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))
...
subjects, region_names, session_info, skipped = [], [], [], []
...
sub = eid2subject[eid]
if sub not in subjects:
    subjects.append(sub)
...
data['subject_idx'].append(subjects.index(sub))
...
data['subjects'] = subjects
```

iii. The trajectory says the AI preferred using the release freeze table because it already contains subject names aligned to the selected `eid`s, so no path parsing or further lookup was needed.

## 1-c. How are the data split into sessions?

i. A session is one unique `eid` from `bwm_release.csv`. The code deduplicates repeated `eid`s by preserving the first occurrence from the release freeze and then processes sessions one by one.

ii. 
```python
bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
# session order = order of first appearance in the public release freeze
eids = list(dict.fromkeys(bwm_df.eid.tolist()))
...
for k, eid in enumerate(eids):
    ...
    sess = load_session(one, eid, args.n_workers)
```

iii. The AI justified this in the trajectory as matching the reference release freeze ordering, while still using ONE-based loaders to fetch the session contents.

## 1-d. How are the data split into trials?

i. Trials come from the session trials table returned by `U.load_trials_and_mask`. The code treats each row as one trial, then keeps a subset of row indices in `keep` and slices neural and behavioral arrays trialwise with those indices.

ii. 
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
trials_mask = np.asarray(trials_mask).astype(bool)
...
keep = np.flatnonzero(good)
...
spk = binned_spikes[keep]
...
choice = trials_df['choice'].to_numpy()[keep]
pleft = trials_df['probabilityLeft'].to_numpy()[keep]
```

iii. The trajectory treats this as straightforward: the session trial table already defines the trial split, and the remaining work is filtering and alignment.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is delegated to `U.load_trials_and_mask(..., max_trial_len=10.0)`, which the AI describes as dropping short/long reaction times, long trials, NaNs in several task variables, and no-choice trials. It then adds another filter: wheel-speed and whisker-motion-energy traces must both exist, have exactly 100 bins, and contain only finite values. Sessions with fewer than two surviving trials are dropped.

ii. 
```python
trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
...
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
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

iii. The AI repeatedly justified this in the trajectory as “exactly as reference `prepare_data`” for the trial mask, then added an extra decoder-driven justification: NaN-containing behavior traces cannot be discretized into categories, so such trials must also be dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from probe-level spike-sorting outputs: spike times, spike cluster assignments, and cluster acronyms. The actual neural matrix is built from spike times and cluster IDs; cluster acronyms are used to track brain-region labels.

ii. 
```python
spikes, clusters, channels = loader.load_spike_sorting()
...
spikes, clusters = U.merge_probes(spikes_list, clusters_list)
neural_dict = {
    'spike_times': spikes['times'],
    'spike_clusters': spikes['clusters'],
    'cluster_regions': clusters['acronym'].to_numpy(),
}
```

iii. In the trajectory the AI explicitly stated it would use pykilosort spike-sorted data from all clusters and merge probes within a session because that matches the vendored decoding pipeline.

## 2-b. How is the `neural` data processed?

i. The AI merges all probes within a session, maps cluster acronyms to Beryl regions through the helper utilities, bins spikes into a `(n_trials, T, n_neurons)` tensor with `U.bin_spiking_data`, and finally transposes each trial to `(n_neurons, n_timepoints)`. It keeps spike counts per 20 ms bin rather than converting counts to firing rates.

ii. 
```python
spikes, clusters = U.merge_probes(spikes_list, clusters_list)
...
regions, beryl_reg = U.list_brain_regions(neural_dict, **PARAMS)
reg_clu_ids = U.select_brain_regions(neural_dict, beryl_reg, regions[0], **PARAMS)
...
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
...
for i in range(len(keep)):
    neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
```

```python
'neural_units': 'spike counts per 20 ms bin',
```

iii. The trajectory says the AI followed the methods-paper pipeline parameters exactly and deliberately kept spike counts, citing the vendored helpers and the methods text about 20 ms trial bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI intentionally applies no neuron-quality threshold. It keeps “ALL clusters,” uses `qc=None`, and only drops sessions that end up with zero neurons or have a probe insertion with no spike sorting at all.

ii. 
```python
Neural: pykilosort spike-sorted spikes of ALL clusters (reference prepare_data calls
  load_spiking_data with qc=None, i.e. no unit-quality threshold; methods paper: "we bin
  spike counts using all neurons ... from each session").
```

```python
"""Spike sorting of one probe insertion, all clusters (no quality threshold).
...
Same as utils.ibl_data_utils.load_spiking_data with qc=None
```

```python
if clusters is None:
    raise RuntimeError(f'no spike sorting for probe insertion {pid} ({pname})')
...
if binned_spikes.shape[2] == 0:
    raise RuntimeError('no neurons')
```

iii. The trajectory explicitly argues that the reference decoding code used `qc=None`, so the AI chose all clusters and rejected a quality-label filter even though the data paper itself used a stricter unit curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times` using a fixed window from `-0.5` to `+1.5` seconds around stimulus onset, matching the decoder task’s requested alignment event.

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

iii. In the trajectory the AI said the methods text presents multiple alignments for different decoder targets, but because this task explicitly requires stimulus-onset alignment for all outputs, it chose the `stimOn_times` configuration from the vendored reference pipeline.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms non-overlapping bins, with 100 bins over a 2 s window. No additional temporal rebinning is applied after helper-based spike and behavior binning.

ii. 
```python
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    ...
    'time_window': (-0.5, 1.5),
}
NBINS = int(round((PARAMS['time_window'][1] - PARAMS['time_window'][0])
                  / PARAMS['binsize']))
```

```python
'time_bin_size': PARAMS['binsize'] * 1000.0,
'n_timepoints': NBINS,
```

iii. The AI repeatedly justified 20 ms bins in the trajectory as the common setting that makes the time-varying wheel and whisker outputs possible while still using stimulus-onset alignment.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI derives this input from the chosen alignment event and the fixed time-window parameters rather than from a standalone raw column. It is implicitly tied to `stimOn_times`, since that is the event the whole session is aligned to.

ii. 
```python
PARAMS = {
    ...
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
...
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
```

iii. The trajectory says the AI chose stimulus onset because the task explicitly requested it and because the vendored caching config already exposed that alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a deterministic 100-sample time grid from the 20 ms bin size and the `(-0.5, 1.5)` window. Importantly, it encodes the end of each bin (`-0.48` to `1.50`) rather than the bin centers.

ii. 
```python
# time grid = end of each 20 ms bin relative to stimulus onset (-0.48 ... 1.50 s),
# the grid the reference code interpolates the behavioural traces onto.
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
```

iii. The trajectory frames this as using the same grid that the reference code uses for behavior interpolation, and the docstring explicitly documents the choice of bin ends rather than centers.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is broadcast on the same 100-bin trial grid used for `spk`, so the input and neural arrays are synchronized bin-by-bin within each trial.

ii. 
```python
spk = binned_spikes[keep]
...
tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
         + PARAMS['time_window'][0]).astype(np.float32)
...
inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. The trajectory justification is that all streams are being binned by the same vendored helper pipeline under one shared `PARAMS` dictionary.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`. A new block starts when `probabilityLeft` changes, and the count within the block is then accumulated.

ii. 
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft."""
    p = np.asarray(prob_left, dtype=float)
    ...
    for i in range(len(p)):
        if i > 0 and np.isclose(p[i], p[i - 1]):
            counter += 1
        else:
            counter = 0
        idx[i] = counter
```

```python
tinb = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())[keep]
```

iii. The trajectory notes that `probabilityLeft` is the block variable exposed by the trials table, so the block structure must be reconstructed from it.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the session’s `probabilityLeft` vector, resets a counter whenever the value changes, and stores a 0-based count for each trial. After trial filtering, the surviving trial numbers are broadcast across all 100 bins of each kept trial.

ii. 
```python
idx = np.zeros(len(p), dtype=np.float32)
counter = 0
for i in range(len(p)):
    if i > 0 and np.isclose(p[i], p[i - 1]):
        counter += 1
    else:
        counter = 0
    idx[i] = counter
```

```python
inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
```

iii. The trajectory treats this as a simple derived variable and does not give a separate deeper justification beyond wanting a per-trial contextual regressor.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from the trials table column `choice`.

ii. 
```python
choice = trials_df['choice'].to_numpy()[keep]
...
choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
```

iii. The trajectory explicitly states that IBL’s sign convention is `choice == +1` for left and `choice == -1` for right, and that no-choice trials are already removed by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The only processing is recoding the kept trial values from IBL’s sign convention into the task’s requested coding: left `0`, right `1`. The scalar per-trial value is then broadcast across the 100 time bins.

ii. 
```python
choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
...
outputs.append(np.stack([
    np.full(NBINS, choice_out[i], dtype=np.int64),
    ...
]))
```

iii. The trajectory says this recoding was chosen after checking the IBL convention explicitly.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials table column `probabilityLeft`.

ii. 
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep]
...
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft], dtype=np.int64)
```

iii. The trajectory and docstring both state that the mapping was chosen directly from the decoder task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounds each retained `probabilityLeft` value to one decimal place and remaps it from `{0.2, 0.5, 0.8}` to `{0, 1, 2}`. It then broadcasts the per-trial category across time.

ii. 
```python
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}
prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft], dtype=np.int64)
...
outputs.append(np.stack([
    np.full(NBINS, choice_out[i], dtype=np.int64),
    np.full(NBINS, prior_out[i], dtype=np.int64),
    ...
]))
```

iii. The AI’s justification is simply that this is the target coding required by the prompt.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI derives wheel speed through the vendored behavior loader under the target name `'wheel-speed'`. In its own documentation it describes this as absolute wheel velocity, which in the IBL stack ultimately comes from wheel position and timestamps.

ii. 
```python
BEH_NAMES = ['wheel-speed', 'whisker-motion-energy']
...
for beh, targets in (('wheel-speed', ['wheel-speed']),
                     ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                'right-whisker-motion-energy'])):
    if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
        raise RuntimeError(f'{beh} is not available for this session')
...
wheel = np.stack([np.asarray(binned_beh['wheel-speed'][i], dtype=float).ravel()
                  for i in keep])
```

iii. The trajectory says the AI wanted to reuse the reference behavior-processing utilities rather than rebuild wheel preprocessing itself, while documenting the signal semantically as `|wheel velocity|`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI delegates wheel processing to `U.bin_behaviors(..., allow_nans=True, **PARAMS)`, then keeps only finite 100-bin traces, pools all retained wheel samples within a session, and discretizes them into three within-session tertile classes.

ii. 
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
...
wheel = np.stack([np.asarray(binned_beh['wheel-speed'][i], dtype=float).ravel()
                  for i in keep])
...
wheel_out = discretize_tertiles(wheel)
```

```python
def discretize_tertiles(values):
    """Map a (ntrials, T) float array to {0,1,2} using this session's pooled tertiles."""
    edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    return np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. The trajectory justifies this by saying wheel traces are heavy-tailed and session-specific, so fixed global thresholds would collapse some sessions into one class.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is thresholded into `low`, `medium`, and `high` by computing the 33.3rd and 66.7th percentiles across all retained wheel samples in the session, then applying `np.searchsorted` against those two thresholds.

ii. 
```python
OUTPUT_VALUES = [
    ['left', 'right'],
    ['0.2', '0.5', '0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]
...
edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
return np.searchsorted(edges, values, side='right').astype(np.int64)
```

iii. The trajectory explicitly calls these “within-session tertiles” and argues they produce roughly balanced categories despite heavy tails and session-to-session unit changes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is binned by the same helper pipeline and parameter set as spikes, so the retained wheel trace for each trial has the same 100-bin stimulus-aligned time axis as the neural array.

ii. 
```python
PARAMS = {
    'binsize': 0.02,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
...
binned_spikes, clusters_used = U.bin_spiking_data(..., **PARAMS)
...
binned_beh, _ = U.bin_behaviors(..., **PARAMS)
```

iii. The AI’s justification in the trajectory is that all streams are processed through the same reference parameterization, so wheel and neural data share the same temporal grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI derives whisker motion energy through the vendored behavior loader under the targets `'left-whisker-motion-energy'` and `'right-whisker-motion-energy'`, then uses the consolidated `'whisker-motion-energy'` trace returned by `U.bin_behaviors`.

ii. 
```python
for beh, targets in (('wheel-speed', ['wheel-speed']),
                     ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                'right-whisker-motion-energy'])):
    if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
        raise RuntimeError(f'{beh} is not available for this session')
...
whisk = np.stack([np.asarray(binned_beh['whisker-motion-energy'][i],
                             dtype=float).ravel() for i in keep])
```

iii. The trajectory says the AI discovered that some sessions have no whisker motion energy at all and therefore made the availability check explicit before attempting conversion.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI delegates whisker processing to `U.bin_behaviors`, then removes trials with missing or non-finite traces, pools all retained whisker samples within a session, and discretizes them with the same tertile rule as wheel speed.

ii. 
```python
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
...
whisk = np.stack([np.asarray(binned_beh['whisker-motion-energy'][i],
                             dtype=float).ravel() for i in keep])
...
whisk_out = discretize_tertiles(whisk)
```

iii. The trajectory justification is the same as for wheel speed: heavy-tailed, session-specific values motivate within-session discretization rather than global thresholds.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded into `low`, `medium`, and `high` using sessionwise pooled 33.3rd and 66.7th percentile cutoffs, exactly like wheel speed.

ii. 
```python
edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
return np.searchsorted(edges, values, side='right').astype(np.int64)
...
whisk_out = discretize_tertiles(whisk)
```

iii. The AI explicitly documents this as a within-session tertile scheme intended to avoid degenerate categories across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is binned under the same `stimOn_times`, `(-0.5, 1.5)`, 20 ms parameterization as spikes, so each kept whisker trace is aligned bin-by-bin to the neural trial matrix.

ii. 
```python
PARAMS = {
    'binsize': 0.02,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
...
binned_beh, _ = U.bin_behaviors(..., **PARAMS)
```

iii. The trajectory repeatedly states that the same helper configuration is used for all aligned streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by skipping it rather than imputing it. It skips sessions with no probe insertions, no spike sorting on any required insertion, no neurons after binning, missing wheel or whisker traces, or fewer than two usable trials. At the trial level it drops any trial whose wheel or whisker trace is missing, wrong-length, or non-finite.

ii. 
```python
if len(pids) == 0:
    raise RuntimeError('no probe insertions')
...
if clusters is None:
    raise RuntimeError(f'no spike sorting for probe insertion {pid} ({pname})')
...
if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
    raise RuntimeError(f'{beh} is not available for this session')
...
ok = np.array([
    (tr is not None) and (np.asarray(tr).size == NBINS)
    and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
    for tr in traces])
good &= ok
...
if len(keep) < MIN_TRIALS:
    raise RuntimeError(f'only {len(keep)} trials survive curation')
```

iii. The trajectory explicitly says these choices were made because the reference code already skips missing streams, and because NaNs cannot be left inside categorical decoder targets.

## 10-a. What are the most time-consuming steps of the code?

i. The code’s expensive work is session-by-session loading and binning: loading spike sorting for each probe insertion, binning spikes across all trials, and binning behavioral traces. The AI also singled out extra raw-ephys metadata reads as unnecessary overhead and removed them.

ii. 
```python
for pid, probe_name in zip(pids, probe_names):
    sp, cl = load_clusters_and_spikes(one, str(pid), eid, probe_name)
```

```python
binned_spikes, clusters_used = U.bin_spiking_data(
    reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
...
binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
```

iii. In the trajectory the AI reported that a single session took about 9 seconds end-to-end, then optimized away one unused raw-ephys metadata lookup because it was not needed for conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still contains explicit Python loops that could be vectorized or avoided: the pass over `probabilityLeft` in `trial_number_in_block`, the per-probe loading loop, the loop over behaviors when constructing `good`, the list comprehensions that stack behavioral traces, and the per-trial loop that assembles `neural`, `input`, and `output`.

ii. 
```python
for i in range(len(p)):
    if i > 0 and np.isclose(p[i], p[i - 1]):
        counter += 1
    else:
        counter = 0
    idx[i] = counter
```

```python
for beh in BEH_NAMES:
    traces = binned_beh[beh]
    ok = np.array([
        (tr is not None) and (np.asarray(tr).size == NBINS)
        and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
        for tr in traces])
    good &= ok
```

```python
for i in range(len(keep)):
    neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
    inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
    outputs.append(np.stack([
        np.full(NBINS, choice_out[i], dtype=np.int64),
        np.full(NBINS, prior_out[i], dtype=np.int64),
        wheel_out[i],
        whisk_out[i],
    ]))
```

iii. The trajectory does not contain a separate optimization discussion here; the code itself shows that the AI prioritized clarity and reuse of helper utilities over aggressive vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work. It pre-checks wheel and whisker availability with `U.load_target_behavior` and then loads/bins behaviors again with `U.bin_behaviors`. It also performs repeated linear membership/index operations when building the `subjects` list.

ii. 
```python
for beh, targets in (('wheel-speed', ['wheel-speed']),
                     ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                'right-whisker-motion-energy'])):
    if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
        raise RuntimeError(f'{beh} is not available for this session')

binned_beh, _ = U.bin_behaviors(
    one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
    n_workers=n_workers, **PARAMS)
```

```python
if sub not in subjects:
    subjects.append(sub)
...
data['subject_idx'].append(subjects.index(sub))
```

iii. The trajectory justifies the availability pre-check as a way to turn an opaque downstream crash into an explicit skip reason for missing whisker data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra work that is not needed to construct decoder-ready arrays: it builds `eid2lab` only for metadata, pre-checks behavior availability before doing the actual behavior loading, stores extensive per-session metadata and skip logs, and computes region-name summaries for `session_info` that are not used by the downstream decoder.

ii. 
```python
eid2lab = dict(zip(bwm_df.eid, bwm_df.lab))
...
session_info.append({
    'eid': eid, 'subject': sub, 'lab': eid2lab[eid],
    'n_probes': sess['n_probes'], 'n_neurons': sess['n_neurons'],
    'n_trials_total': sess['n_trials_total'],
    'n_trials_kept': sess['n_trials_kept'],
    'regions': sorted(set(sess['cluster_regions'].tolist())),
})
...
'session_info': session_info,
'skipped_sessions': skipped,
```

iii. The trajectory does not defend these extras as analytically necessary; the only explicit justification is that better skip reasons and metadata make the conversion easier to validate.
