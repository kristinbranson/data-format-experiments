# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent initializes local ONE, activates the Brainwidemap cache, reads `bwm_release.csv`, groups it by session `eid`, and resolves each session path. Trials and camera arrays are opened directly; wheel and spike sorting use ONE/IBL loaders. `MAX_SESSIONS` optionally limits a test run.

ii. ```python
one = ONE(mode='local', cache_dir=CACHE)
one.load_cache(CACHE / 'Brainwidemap')
release = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    spath = one.eid2path(eid)
    tr = load_trials(spath)
```

iii. The trajectory says the release contains the intended full data (no `DATALIMIT_SUBSET.csv` was present), local ONE resolves session paths, and the release CSV supplies the curated sessions/probes.

## 1-b. How are the data split into subjects?

i. The subject attached to each release-CSV session group is saved with the session. Unique subjects are later collected in first-seen order and each retained session gets an integer `subject_idx`.

ii. ```python
sessions.append(dict(eid=str(eid), subject=str(probes.subject.iloc[0]), ...))
subjects = list(dict.fromkeys(s['subject'] for s in sessions))
subj_map = {x:i for i,x in enumerate(subjects)}
subject_idx=np.array([subj_map[s['subject']] for s in kept],np.int32)
```

iii. The trajectory treated the release metadata as the authoritative subject identifier, avoiding filename/path parsing.

## 1-c. How are the data split into sessions?

i. Each distinct `eid` in the release CSV is one session; all listed probes for that `eid` are processed and merged into one session population.

ii. ```python
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    ...
    for r in probes.itertuples(index=False):
```

iii. The trajectory identified `eid` as ONE's unique session identifier and the release CSV as the curated session/probe index.

## 1-d. How are the data split into trials?

i. The trials parquet table supplies one row per trial. Indices passing the validity mask are used to slice trial variables, and continuous streams are windowed separately around each retained trial's stimulus onset.

ii. ```python
ti = np.flatnonzero(valid)
onsets = tr.stimOn_times.to_numpy(float)[ti]
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
```

iii. The agent reasoned that the ALF trials table already defines trial boundaries/events, so no inferred trial segmentation is required.

## 1-e. How are trials filtered based on quality controls?

i. A trial must have finite choice, prior, feedback type/time, stimulus onset, and first movement; reaction time must be 0.08–2.00 s; choice must be ±1; rounded prior must be 0.2/0.5/0.8. After behavior binning/gap filling, wheel and whisker must be finite in every bin. Sessions need at least two retained trials.

ii. ```python
needed = ['choice','probabilityLeft','feedbackType','feedback_times',
          'stimOn_times','firstMovement_times']
for c in needed: valid &= np.isfinite(tr[c].to_numpy(float))
valid &= (rt >= .08) & (rt <= 2.00)
valid &= np.isin(tr.choice.to_numpy(float), [-1, 1])
valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
```

iii. The trajectory cites the paper's 80-ms to 2-s reaction-time curation and no-choice exclusion. Complete behavior was required for fixed-size decoder outputs; finite feedback fields were added as required events.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays come from spike times and spike cluster assignments. Cluster IDs, quality labels, and atlas IDs determine which spike clusters become output neurons and their brain-region labels.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
ids = np.asarray(clusters['cluster_id'], int)
... probe_data.append((np.asarray(spikes.times), np.asarray(spikes.clusters), gids, acr))
```

iii. The trajectory recognized `spikes.times` and `spikes.clusters` as the direct neural observations, with cluster metadata used only for curation/anatomy.

## 2-b. How is the `neural` data processed?

i. Good neurons from every probe in a session are concatenated. Their spikes are histogrammed into neuron-by-100 arrays of raw spike counts; no smoothing or division by bin width is applied. Counts are stored as `uint16`.

ii. ```python
neural = np.zeros((len(ti), n_neurons, len(CENTERS)), dtype=np.uint16)
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. The agent read the methods as specifying 20-ms spike counts and deliberately used integer storage to keep the 185,891-trial pickle manageable; the trajectory notes the validator converts it for training.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps `clusters.label == 1`, valid positive atlas IDs, and Beryl gray-matter acronyms excluding `root`, `void`, and `fiber tracts`. A region needs at least five good neurons in a session; after all sessions are processed it must occur in at least two sessions.

ii. ```python
good = (labels == 1) & np.isfinite(atlas_ids) & (atlas_ids > 0)
keep = ~np.isin(acr, ['root','void','fiber tracts'])
allowed = set(vals[cnt >= 5])
allowed_global = {r for r,n in prevalence.items() if n >= 2}
```

iii. The trajectory calls label 1 the RIGOR well-isolated-unit criterion and attributes the ≥5-neuron/session and ≥2-session prevalence rules to the paper's regional analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are selected on the shared session clock from 0.5 s before through 1.5 s after each `stimOn_times` value, then stimulus onset is subtracted when assigning relative bins.

ii. ```python
onsets = tr.stimOn_times.to_numpy(float)[ti]
lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
```

iii. The trajectory says IBL synchronizes neural and behavioral streams upstream, so alignment only requires using the same session-clock stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms. Each two-second window has 100 bins. Spikes are binned once at that resolution, with no later neural resampling.

ii. ```python
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
```

iii. The trajectory explicitly found the paper configuration of two-second trials, 20-ms bins, and 100 time points.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from each trial's `stimOn_times` alignment and the fixed bin edges; the saved values are the relative bin centers and are identical across trials.

ii. ```python
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
np.vstack((CENTERS.astype(np.float32), ...))
```

iii. The agent used the reference window and stimulus-onset event discovered in the paper code.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Adjacent -0.5-to-1.5-s bin edges are averaged to produce centers from -0.49 through 1.49 s; no raw signal is transformed.

ii. ```python
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The trajectory treats this as a decoder coordinate defined by the analysis grid rather than a measured stream.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Each time value is the center of the exact edge interval used to histogram the corresponding neural column.

ii. ```python
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
input_out.append([np.vstack((CENTERS.astype(np.float32), ...)) ...])
```

iii. The same stimulus onset, offsets, and 20-ms grid are used for both arrays.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table's `probabilityLeft`; a changed or previously nonfinite probability begins a new block.

ii. ```python
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
```

iii. The trajectory inferred blocks from the prior because the trials table has no separate block identifier.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. All original trials are traversed before filtering. The counter is zero-based, resets when prior changes, increments otherwise, and the retained values are broadcast over all 100 time bins.

ii. ```python
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
np.full(len(CENTERS),s['block_trial'][j],np.float32)
```

iii. Counting before filtering preserves the animal's true position even when intervening trials are later excluded.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `tr.choice`, whose valid values are +1 for left and -1 for right.

ii. ```python
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)
```

iii. The trajectory caught and corrected an initial reversal after confirming IBL's sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Valid +1 values map to 0 (left), and -1 values map to 1 (right); the scalar category is repeated over the trial's 100 bins.

ii. ```python
choice = (... == -1).astype(np.int8)
np.full(len(CENTERS),s['choice'][j],np.int8)
```

iii. This directly implements the requested left=0/right=1 coding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trials table's `probabilityLeft` column.

ii. ```python
pleft = np.round(tr.probabilityLeft.to_numpy(float)[ti], 1)
```

iii. The agent identified this column as the experiment's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal and mapped 0.2→0, 0.5→1, 0.8→2, then repeated over time.

ii. ```python
prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
np.full(len(CENTERS),s['prior'][j],np.int8)
```

iii. The mapping is explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from the wheel object's timestamps and angular position.

ii. ```python
wheel = one.load_object(eid, 'wheel', collection='alf')
wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
```

iii. The trajectory found that standard IBL wheel velocity is computed from timestamped position.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, passed through `velocity_filtered` (including its default low-pass processing), converted to absolute velocity, averaged within each 20-ms trial bin, and empty bins are interpolated from other bins in that trial.

ii. ```python
ws, _ = velocity_filtered(wpos, 1000)
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
```

iii. The trajectory says interpolation/filtering follow the standard brainbox/SessionLoader wheel procedure; bin means were chosen to aggregate all samples in each neural bin.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All finite wheel-bin values in a session are split at session-wide 1/3 and 2/3 quantiles with `np.digitize`, producing 0/1/2. A tiny deterministic perturbation is used only if both thresholds coincide.

ii. ```python
q = np.quantile(finite, [1/3, 2/3])
if q[0] == q[1]:
    q = np.quantile(finite + np.linspace(0, 1e-7, finite.size), [1/3, 2/3])
return np.digitize(x, q, right=False).astype(np.int8), q.tolist()
```

iii. Tertiles were required to make the continuous behavior a three-class output while keeping classes approximately balanced per session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel samples on the shared clock are averaged between the same stimulus-relative edges used for neural bins, yielding one category per neural column.

ii. ```python
ix = np.searchsorted(times, onset + EDGES)
sm = cs[ix[1:]] - cs[ix[:-1]]
np.divide(sm, n, out=out[i], where=n > 0)
```

iii. The shared session clock and identical bin edges provide binwise temporal alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses the left camera's released ROI motion-energy array and left-camera frame timestamps, truncated to their common length.

ii. ```python
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
tf = newest(alf, '_ibl_leftCamera.times.npy')
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
n = min(len(me), len(mt))
```

iii. The trajectory chose the high-frame-rate left side view as the whisker-pad signal; sessions lacking it were skipped.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released values receive no filtering or normalization. They are averaged per 20-ms trial bin, missing bins are interpolated within each row, and the result is discretized by session tertiles.

ii. ```python
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
whisk_c, whisk_q = tertiles(whisk_b)
```

iii. The trajectory regarded the released ROI motion energy as already processed and applied only time aggregation and task-required categorization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-wide 1/3 and 2/3 quantile routine as wheel speed, yielding categories 0, 1, and 2.

ii. ```python
whisk_c, whisk_q = tertiles(whisk_b)
q = np.quantile(finite, [1/3, 2/3])
```

iii. The agent sought approximately equal-frequency low/medium/high classes within each session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera samples are grouped by the same absolute `stimOn_times + EDGES` intervals used for neural bins.

ii. ```python
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
ix = np.searchsorted(times, onset + EDGES)
```

iii. Camera times and spike times share the synchronized session clock, so common trial edges align them.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Required nonfinite trial fields are excluded. Behavior arrays are length-matched; empty behavior bins are interpolated when a row has at least two finite bins, otherwise incomplete trials are dropped. Missing/invalid probes are effectively skipped, and any session exception is recorded and skipped. Sessions with fewer than two trials or no qualifying neural region are excluded.

ii. ```python
n = min(len(me), len(mt))
row[~ok] = np.interp(q[~ok], q[ok], row[ok])
except Exception as exc:
    failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
```

iii. The trajectory prioritized a fixed complete decoder tensor, preserving small internal gaps by interpolation while logging sessions that cannot be made usable.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/merging large spike-sorting arrays and iterating through every session/trial to bin spikes dominate runtime; full conversion took a long monitored run over 459 sessions. Serialization/verification of the 5.2-GB pickle is also substantial.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
for j, onset in enumerate(onsets):
    ...
    np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. The trajectory repeatedly identified spike-file I/O and large spike vectors as the expensive work and monitored the full run session by session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `binned_mean`, `fill_short_gaps`, spike binning, block counting, and final per-trial assembly could potentially be vectorized or batched. The spike loop already limits work with `searchsorted`, avoiding a full-vector scan per trial.

ii. ```python
for i, onset in enumerate(onsets):
for row in x:
for j, onset in enumerate(onsets):
for j in range(len(tr)):
```

iii. The trajectory explicitly chose indexed trial slices for clarity and efficiency on irregular windows, while using lookup maps and `searchsorted` to remove the worst repeated scans.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches/slices by trial for spikes and both behavior streams, applies gap filling and tertile logic separately to wheel and whisker, and later loops over every trial again to build list-based output arrays.

ii. ```python
wheel_b = fill_short_gaps(binned_mean(...))
whisk_b = fill_short_gaps(binned_mean(...))
neural_out.append([s['neural'][j,nk] for j in range(len(s['trials']))])
```

iii. The trajectory did not offer a separate justification beyond keeping stream-specific transformations explicit and satisfying the nested target format.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges cluster channels and computes metadata fields that are not directly decoded; it bins neurons from regions that may later fail the across-session prevalence filter; and it carries feedback fields only for filtering although feedback is not an input/output. Test-only `MAX_SESSIONS` handling is irrelevant to the full run.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
needed = [..., 'feedbackType','feedback_times', ...]
nk = np.isin(s['regions'], list(allowed_global))
```

iii. The trajectory viewed merged cluster metadata as necessary for QC/anatomy and deferred global prevalence filtering because prevalence cannot be known until all sessions have been surveyed.
