# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the bundled BWM release CSV, optionally restricts it using `DATALIMIT_SUBSET.csv`, groups insertions by session, and processes every selected session. It initializes ONE from the local Brainwidemap cache, uses `SpikeSortingLoader` for each probe and `SessionLoader` for wheel data, but reads consolidated trials parquet and camera arrays directly from each session's ALF directory. Per-session results are cached and the final run assembled 444 usable sessions from 459 release sessions.

ii.
```python
bwm = pd.read_csv(RELEASE, index_col=0)
...
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid),
                 list(group.pid.astype(str)), list(group.probe_name.astype(str))))
...
one = ONE(cache_dir=CACHE, mode='local', silent=True)
one._cache = load_tables(CACHE/'Brainwidemap')
```

iii. The trajectory says the release CSV defines the release universe and that all sessions, rather than the initial 10-session smoke-test sample, were required. The agent used local cache tables and resumable session shards to make the full conversion feasible.

## 1-b. How are the data split into subjects?

i. Subject IDs come from the release CSV. Sessions retain their subject string; unique subjects are sorted and each session receives an index into that list.

ii.
```python
subjects = sorted(set(s['subject'] for s in sessions))
smap = {s:i for i,s in enumerate(subjects)}
subject_idx=np.array([smap[s['subject']] for s in sessions], np.int32)
```

iii. The trajectory treats the release's `subject` field as the authoritative identifier, requiring no filename or path parsing.

## 1-c. How are the data split into sessions?

i. Rows in the release CSV are grouped by `eid`; all probe IDs and probe names for an EID are merged into one session record.

ii.
```python
for eid, group in bwm.groupby('eid', sort=False):
    rows.append((str(group.subject.iloc[0]), str(eid),
                 list(group.pid.astype(str)), list(group.probe_name.astype(str))))
```

iii. The trajectory identifies `eid` as the session unit and explicitly merges probes because they share behavior and are not independent sessions.

## 1-d. How are the data split into trials?

i. The consolidated trials table supplies one row per trial. Binned neural data and interpolated behavior are indexed by retained trial indices to create per-trial matrices.

ii.
```python
trials = load_local_trials(one, eid)
...
idx = np.flatnonzero(valid)
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
```

iii. The agent reasoned that the trial table and the reference binning utility already establish trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. A trial must have nonmissing stimulus, choice, feedback, prior, first-movement, and feedback-type fields; reaction time must be 0.08–2 s; go-cue-to-feedback duration must be at most 10 s; choice must be nonzero; and wheel and whisker streams must cover the full trial window. Sessions with fewer than two jointly valid trials are excluded.

ii.
```python
valid = trials[required].notna().all(axis=1).to_numpy(dtype=bool, copy=True)
valid &= (rt >= 0.08) & (rt <= 2.0)
valid &= duration <= 10.0
valid &= trials.choice.to_numpy() != 0
...
valid = paper_mask & wheel_good & whisk_good
```

iii. The trajectory says these criteria reproduce the paper repository's trial mask, while the coverage checks are necessary because wheel and whisker are required outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each probe's `spikes.times` and `spikes.clusters`; the cluster table provides the complete unit list and anatomy.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
neural_df = {'spike_times': spikes['times'], 'spike_clusters': spikes['clusters']}
```

iii. The trajectory identified these as the reference repository's direct inputs to spike binning.

## 2-b. How is the `neural` data processed?

i. Probes are merged within a session, spikes are counted in 20 ms bins from -0.5 to 1.5 s, and each retained trial is transposed to neuron-by-time. The agent saves integer spike counts; it does not divide by bin width to obtain Hz.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
binned, used = bin_spiking_data(cluster_ids, neural_df, trials_df=trials,
                                n_workers=4, **PARAMS)
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
```

iii. The trajectory states that the Zhang reference utility bins all Kilosort clusters and that the common representation is stimulus-aligned 20 ms data. It interpreted that representation as spike counts rather than firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality or anatomical filter is applied: all cluster rows are passed to binning, including clusters not labeled good and clusters mapped to `void`.

ii.
```python
cluster_ids = np.arange(len(clusters), dtype=np.int64)
binned, used = bin_spiking_data(cluster_ids, neural_df, ...)
```

iii. The trajectory explicitly concluded that the methods repository used `qc=None` and therefore chose all Kilosort clusters. It did not follow the human solution's data-paper QC choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. `bin_spiking_data` aligns each trial to `stimOn_times` and extracts -0.5 to +1.5 s.

ii.
```python
PARAMS = dict(interval_len=2, binsize=BIN, single_region=False,
              align_time='stimOn_times', time_window=(OFF0, OFF1))
```

iii. The trajectory found this exact alignment and window in the provided Zhang code and used it as the shared representation for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms, yielding 100 bins over two seconds. Spikes are binned once; there is no later neural rebinning or smoothing.

ii.
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
```

iii. The trajectory says the provided repository specifies 20 ms for this jointly aligned decoder representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the stimulus-aligned window parameters and `stimOn_times`, rather than from a sampled raw trace.

ii.
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1,
                 dtype=np.float32)*BIN + OFF0
```

iii. The agent justified the axis from the common -0.5-to-1.5 s, 20 ms stimulus-aligned grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates 100 values from -0.48 through 1.50 s, i.e. bin right edges.

ii.
```python
time = np.arange(1, int(round((OFF1-OFF0)/BIN))+1,
                 dtype=np.float32)*BIN + OFF0
```

iii. The trajectory chose the reference utility's interpolation convention, described as interpolation to bin right edges.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The generated 100-point axis is paired positionally with the 100 neural bins. It labels bins by right edges rather than by the human solution's bin centers.

ii.
```python
T = s['neural'][j].shape[1]
inp = np.vstack((time[:T], np.full(T, s['block_trial'][j], np.float32)))
```

iii. The agent regarded both arrays as sharing the same stimulus-relative 20 ms grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive values of the trials table's `probabilityLeft`; a change starts a new block.

ii.
```python
block_trial = trial_number_in_block(trials.probabilityLeft.to_numpy())[idx]
```

iii. The trajectory notes that no explicit block ID is needed because prior probability is constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code counts trials starting at 1 and resets to 1 whenever the prior changes. It computes this before filtering, so excluded trials still advance the true within-block position, then broadcasts the value over time.

ii.
```python
if i == 0 or value != previous:
    k = 1
else:
    k += 1
...
np.full(T, s['block_trial'][j], np.float32)
```

iii. The trajectory supports computing the count before filtering to preserve the animal's actual block position; it did not discuss the one-based versus zero-based departure from the human solution.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from `trials.choice`; zero/no-response trials are removed first.

ii.
```python
choice_raw = trials.choice.to_numpy()[idx]
choice = (choice_raw == 1).astype(np.int8)
```

iii. The trajectory recognized raw values -1, 0, and +1 and the required binary output, but did not catch that IBL +1 denotes left and -1 denotes right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps raw +1 to category 1 and raw -1 to category 0, then broadcasts it across all time bins. Given the declared `['left', 'right']` ordering, this reverses left and right.

ii.
```python
choice = (choice_raw == 1).astype(np.int8)
out = np.vstack((np.full(T, s['choice'][j], np.int8), ...))
```

iii. The trajectory only verified that choices belonged to `[-1, 1]`; it supplied no valid justification for the reversed mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft`.

ii.
```python
pleft_raw = trials.probabilityLeft.to_numpy()[idx]
```

iii. The trajectory identifies this field as the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to categories 0, 1, and 2, then repeated across time.

ii.
```python
pmap = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([pmap[round(float(x), 1)] for x in pleft_raw], np.int8)
```

iii. The trajectory follows the mapping stated explicitly in the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()` loads wheel position/timestamps and provides filtered velocity; the code takes its absolute value.

ii.
```python
sl.load_wheel()
wheel_t = sl.wheel.times.to_numpy()
wheel_v = np.abs(sl.wheel.velocity.to_numpy())
```

iii. The trajectory found that the reference behavior is absolute, smoothed wheel velocity produced by the IBL loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Loader-derived velocity is made absolute, coverage-checked, linearly interpolated to 100 stimulus-relative right-edge times, and later discretized. No additional normalization is applied.

ii.
```python
xi = np.linspace(beg+BIN, end, nbin)
out.append(np.interp(xi, tx, vx).astype(np.float32))
```

iii. The trajectory says this follows the provided repository's behavior-loading/interpolation path.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are computed from all retained wheel samples pooled across every converted session; `np.digitize` maps samples into global low/medium/high tertiles.

ii.
```python
values = np.concatenate([np.concatenate(s[key]) for s in sessions])
edges = np.quantile(values, [1/3, 2/3])
...
wb = np.digitize(s['wheel'][j], wheel_edges).astype(np.int8)
```

iii. The trajectory chose global pooled tertiles to provide consistent category meanings across sessions. This differs from the human solution's session-specific percentile thresholds.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated on the same stimulus-relative number of samples, but at bin right edges (-0.48 to 1.50 s), whereas the human neural-bin representation uses centers.

ii.
```python
beg, end = onset + OFF0, onset + OFF1
xi = np.linspace(beg+BIN, end, nbin)
```

iii. The agent believed the repository's right-edge samples were the matching neural grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera ROI motion-energy arrays and matching camera timestamps, falling back to the right camera when necessary.

ii.
```python
for view in ('left', 'right'):
    tf = sorted(alf.glob(f'#*#/_ibl_{view}Camera.times.npy'))
    mf = sorted(alf.glob(f'#*#/{view}Camera.ROIMotionEnergy.npy'))
```

iii. The trajectory says this reproduces the provided reference logic and handles dataset revisions by matching lengths.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI motion-energy values are coverage-checked and linearly interpolated onto the 100 right-edge times; no filtering or normalization is applied before discretization.

ii.
```python
whisk, whisk_good = interpolate(motion_t, motion_v)
```

iii. The agent found no additional whisker processing in the reference repository.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. All retained whisker samples from all sessions are pooled, global 1/3 and 2/3 quantiles are calculated, and `np.digitize` creates three categories.

ii.
```python
whisk_edges = category_edges(sessions, 'whisk')
mb = np.digitize(s['whisk'][j], whisk_edges).astype(np.int8)
```

iii. As for wheel speed, the trajectory favors globally consistent categories; this differs from the human solution's per-session tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera values are interpolated relative to the same `stimOn_times` at 20 ms right-edge samples and paired positionally with neural bins.

ii.
```python
beg, end = onset + OFF0, onset + OFF1
xi = np.linspace(beg+BIN, end, nbin)
```

iii. The agent relied on the shared IBL session clock and stimulus onset, but used right edges rather than the human solution's bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Trials lacking required fields or full behavior coverage are dropped. A session is skipped and its exception cached if data are unavailable, behavior arrays cannot be matched, or fewer than two valid trials remain. Writes use temporary files and atomic replacement. Fifteen of 459 sessions were excluded (14 missing/mismatched whisker streams, one with fewer than two valid trials).

ii.
```python
if len(idx) < 2:
    raise RuntimeError('fewer than two valid trials')
...
fail_file.write_text(f'{type(exc).__name__}: {exc}\n')
os.replace(tmp_file, cache_file)
```

iii. The trajectory explicitly reconciled all release EIDs and judged these exclusions necessary because every retained example needs all requested outputs and at least two trials.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike-sorting arrays, binning spikes for every session/trial, and serializing/loading the roughly 53 GB assembled pickle dominate. Behavior interpolation and metadata construction are comparatively small.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
binned, used = bin_spiking_data(..., n_workers=4, **PARAMS)
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory observed that raw spike products are very large and used 16 resumable shards; final assembly and verification each required loading/scanning the full dataset.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial interpolation loop, the Python loop mapping prior values, within-block counter loop, per-trial final assembly loop, and region-index list comprehension could be vectorized. Some remain reasonable because trial windows slice irregular portions of continuous streams.

ii.
```python
for i, onset in enumerate(trials.stimOn_times.to_numpy()):
    ...
    out.append(np.interp(xi, tx, vx).astype(np.float32))
...
for j in range(len(s['neural'])):
```

iii. The trajectory focused optimization on parallel session processing and reference multiprocessing rather than rewriting these small or irregular loops.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches ALF directories and loads/interpolates behavior per session, repeatedly constructs constant per-trial time and label vectors, and loads each session cache once during assembly after writing it during conversion. Camera filename searches are repeated across views and revisions.

ii.
```python
for view in ('left', 'right'):
    tf = sorted(alf.glob(...))
    mf = sorted(alf.glob(...))
...
np.full(T, s['choice'][j], np.int8)
```

iii. The trajectory accepted repetition in exchange for revision-safe file matching, resumability, and simple per-session isolation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Neural binning is performed for every trial before the combined quality/behavior mask is applied, so bins for rejected trials are computed and discarded. Camera matching may load multiple arrays before finding a length match. It also retains source trial indices and detailed session metadata that the decoder does not use.

ii.
```python
binned, used = bin_spiking_data(..., trials_df=trials, ...)
...
valid = paper_mask & wheel_good & whisk_good
idx = np.flatnonzero(valid)
neural = [np.asarray(binned[i].T, dtype=np.int16) for i in idx]
```

iii. The trajectory prioritized using the provided binning utility unchanged and preserving provenance, despite the wasted work and unused metadata.
