# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the local ONE cache plus the Zhang repository's curated `/app/code/code_zhang2025/data/bwm_release.csv` as the master index. It loads the `Brainwidemap` cache, groups the release CSV by `eid`, resolves each session path with `one.eid2path`, reads the newest trials parquet directly from disk, loads wheel data through `one.load_object`, loads whisker motion energy by directly `np.load`-ing the left-camera files, and loads spikes per probe with `SpikeSortingLoader`.

ii. ```python
one = ONE(mode='local', cache_dir=CACHE)
one.load_cache(CACHE / 'Brainwidemap')
release = pd.read_csv(RELEASE).drop(columns=['Unnamed: 0'], errors='ignore')

for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    spath = one.eid2path(eid)
    tr = load_trials(spath)
```

```python
wheel = one.load_object(eid, 'wheel', collection='alf')
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
tf = newest(alf, '_ibl_leftCamera.times.npy')
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
```

iii. In the trajectory, the AI explicitly decided to work offline from the staged ONE cache, concluded that `bwm_release.csv` was the curated paper/repository session list, and chose direct local loading because Alyx/network access was unreliable in the environment.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `subject` column of the grouped `bwm_release.csv` rows. Each processed session stores a single subject string, then the final output builds `subjects` from first appearance order and constructs `subject_idx` from that list.

ii. ```python
sessions.append(dict(eid=str(eid), subject=str(probes.subject.iloc[0]),
    trials=ti, neural=neural, regions=np.asarray(region_names),
    ...))
```

```python
subjects = list(dict.fromkeys(s['subject'] for s in sessions))
subj_map = {x:i for i,x in enumerate(subjects)}
...
subject_idx=np.array([subj_map[s['subject']] for s in kept],np.int32)
```

iii. The trajectory rationale was that the release CSV already provided per-session subject IDs, so there was no need to derive subjects from paths or separate metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each unique `eid` as one session and groups the probe-level release table by `eid`. All probe insertions belonging to that `eid` are merged into one session-level example.

ii. ```python
for si, (eid, probes) in enumerate(release.groupby('eid', sort=False), 1):
    ...
    for r in probes.itertuples(index=False):
        ssl = SpikeSortingLoader(eid=eid, pname=r.probe_name, one=one)
```

iii. In the trajectory, the AI inferred that `bwm_release.csv` has one row per probe insertion and that sessions therefore had to be reconstructed by grouping rows with the same `eid`.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the session's trials table parquet. The script creates a Boolean validity mask over rows, then uses the surviving row indices `ti` as the trial split for neural and behavioral extraction.

ii. ```python
tr = load_trials(spath)
valid = np.ones(len(tr), bool)
...
ti = np.flatnonzero(valid)
...
onsets = tr.stimOn_times.to_numpy(float)[ti]
```

iii. The trajectory states that the full trial table was available and already contained the per-trial variables needed, so trial boundaries came directly from that table.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with finite `choice`, `probabilityLeft`, `feedbackType`, `feedback_times`, `stimOn_times`, and `firstMovement_times`; first-movement latency between 0.08 and 2.00 s after stimulus onset; binary choices in `{-1, 1}`; and rounded `probabilityLeft` in `{0.2, 0.5, 0.8}`. After behavior is extracted, it further drops trials whose wheel or whisker traces still contain NaNs after gap filling, and it drops sessions left with fewer than two valid trials.

ii. ```python
needed = ['choice','probabilityLeft','feedbackType','feedback_times',
          'stimOn_times','firstMovement_times']
...
for c in needed:
    valid &= np.isfinite(tr[c].to_numpy(float))
rt = tr.firstMovement_times.to_numpy(float) - tr.stimOn_times.to_numpy(float)
valid &= (rt >= .08) & (rt <= 2.00)
valid &= np.isin(tr.choice.to_numpy(float), [-1, 1])
valid &= np.isin(np.round(tr.probabilityLeft.to_numpy(float), 1), [.2, .5, .8])
```

```python
good_trials = np.isfinite(wheel_b).all(1) & np.isfinite(whisk_b).all(1)
if good_trials.sum() < 2:
    raise ValueError('fewer than two trials with complete behavior')
```

iii. The trajectory says the AI adopted the paper's 80 ms to 2 s first-movement filter and valid choice/prior constraints, and then added complete-behavior requirements so every retained trial would have decoder outputs in all bins.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is built from spike timestamps and spike cluster assignments for each probe. Cluster metadata fields `label`, `cluster_id`, and `atlas_id` are used to decide which units survive and which brain region label each neuron gets.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
labels = np.asarray(clusters['label'])
ids = np.asarray(clusters['cluster_id'], int)
atlas_ids = np.asarray(clusters['atlas_id'])
```

```python
probe_data.append((np.asarray(spikes.times), np.asarray(spikes.clusters), gids, acr))
```

iii. In the trajectory, the AI identified `spikes.times` and `spikes.clusters` as the core neural signals and used cluster metadata only for QC and anatomical labeling.

## 2-b. How is the `neural` data processed?

i. For each retained session, the AI bins spikes into 100 bins of 20 ms over a -0.5 s to 1.5 s window around stimulus onset. It merges all probes from the session into one neuron population, remaps cluster IDs to contiguous output rows, and accumulates spike counts with `np.add.at`. It stores raw spike counts as `uint16`; it does not divide by bin width to convert to firing rates.

ii. ```python
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

```python
neural = np.zeros((len(ti), n_neurons, len(CENTERS)), dtype=np.uint16)
...
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
inside = (bb >= 0) & (bb < len(CENTERS))
np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. The trajectory says the AI followed the repository's stimulus-aligned 2 s window and 20 ms bins, then intentionally kept spike counts in `uint16` to keep the 459-session export tractable in memory and on disk.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `clusters['label'] == 1`, finite positive `atlas_id`, and Beryl acronyms not in `{'root', 'void', 'fiber tracts'}`. It then applies an additional per-session region filter requiring at least 5 surviving neurons in a Beryl region, and a later cross-session region filter requiring a region to appear in at least 2 sessions.

ii. ```python
good = (labels == 1) & np.isfinite(atlas_ids) & (atlas_ids > 0)
...
mapped = atlas.remap(atlas_ids[good].astype(int), source_map='Allen', target_map='Beryl')
acr = np.asarray(atlas.get(mapped)['acronym']).astype(str)
keep = ~np.isin(acr, ['root','void','fiber tracts'])
```

```python
vals, cnt = np.unique(all_regions, return_counts=True)
allowed = set(vals[cnt >= 5])
...
allowed_global = {r for r,n in prevalence.items() if n >= 2}
```

iii. In the trajectory, the AI justified this by citing the methods text and paper curation rules: use well-isolated units, keep only grey-matter Beryl regions, require at least 5 good neurons in a session-region, and retain only regions observed in at least 2 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to `stimOn_times`. For each onset, the script slices spikes in `[onset-0.5, onset+1.5)` and converts spike times to onset-relative bin indices, so time zero is visual stimulus onset.

ii. ```python
onsets = tr.stimOn_times.to_numpy(float)[ti]
...
lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
```

iii. The trajectory explicitly notes that the original continuous wheel decoder used a different alignment, but this task overrode that and required stimulus-onset alignment for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 20 ms bins over a 2 s window, giving 100 time bins per trial. There is no later rebinning or resampling of the neural signal.

ii. ```python
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
EDGES = np.arange(OFF0, OFF1 + DT/2, DT)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The trajectory repeatedly cites the Zhang code and methods text for the exact `[-0.5, 1.5]` window and 20 ms bin size.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The input is defined from the stimulus-onset alignment event in the trials table and the fixed bin grid. The per-trial raw event is `stimOn_times`; the output values themselves are the shared bin centers `CENTERS`.

ii. ```python
onsets = tr.stimOn_times.to_numpy(float)[ti]
DT = 0.020
OFF0, OFF1 = -0.5, 1.5
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

iii. The trajectory rationale was that the decoder input should mirror the neural bin grid around stimulus onset rather than recover another raw variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No additional raw-data processing is done. The AI defines this input as the 100 bin centers spanning -0.49 s to 1.49 s relative to stimulus onset and reuses the same vector for every trial.

ii. ```python
input_out.append([np.vstack((CENTERS.astype(np.float32),
                    np.full(len(CENTERS),s['block_trial'][j],np.float32)))
                  for j in range(len(s['trials']))])
```

iii. In the trajectory, the AI treated this as a task-defined decoder covariate rather than a measured signal needing preprocessing.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The time-since-stimulus input uses exactly the same `CENTERS` array that the neural bins imply, so it is aligned bin-for-bin with the onset-relative neural counts.

ii. ```python
bb = np.floor((st[lo:hi][ok] - onset - OFF0) / DT).astype(np.int32)
...
input_out.append([np.vstack((CENTERS.astype(np.float32),
                    np.full(len(CENTERS),s['block_trial'][j],np.float32)))
                  for j in range(len(s['trials']))])
```

iii. The AI's trajectory explicitly describes this as a single common stimulus-aligned grid reused for neural data and decoder inputs/outputs.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` sequence. The AI infers block boundaries by detecting when the rounded `probabilityLeft` value changes across consecutive trials.

ii. ```python
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
```

iii. The trajectory rationale was that no explicit block ID was available, so block membership had to be reconstructed from the prior-probability schedule.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI walks through the full session trial table once, resets a counter at the first trial or whenever `probabilityLeft` changes, increments otherwise, and stores a zero-based within-block count. Only after computing the full-session count does it index by retained trials `ti`.

ii. ```python
block_trial = np.zeros(len(tr), np.float32); k = 0
probs = np.round(tr.probabilityLeft.to_numpy(float), 1)
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
    else: k += 1
    block_trial[j] = k
...
block_trial=block_trial[ti]
```

iii. The trajectory says this was meant to reflect the animal's original block progression, not the renumbered order after QC filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from `tr.choice` in the trials table.

ii. ```python
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)  # IBL: +1 left, -1 right
```

iii. The trajectory shows the AI eventually corrected its first implementation after realizing IBL uses `+1` for left and `-1` for right.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The processing is a binary recode: retained trials with IBL `choice == -1` become decoder label `1` (right), while retained trials with `choice == +1` become `0` (left). This scalar is then repeated across all 100 bins of the trial.

ii. ```python
choice = (tr.choice.to_numpy(float)[ti] == -1).astype(np.int8)
...
output_out.append([np.vstack((np.full(len(CENTERS),s['choice'][j],np.int8),
                     np.full(len(CENTERS),s['prior'][j],np.int8),
                     s['wheel'][j],s['whisker'][j]))
                   for j in range(len(s['trials']))])
```

iii. The trajectory explains that this recoding was required to satisfy the task's explicit label convention `left = 0, right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `tr.probabilityLeft` in the trials table.

ii. ```python
pleft = np.round(tr.probabilityLeft.to_numpy(float)[ti], 1)
prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
```

iii. The trajectory treated this as a direct categorical remapping of the task's block-prior variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounds the raw values to one decimal place and maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. Like choice, the resulting categorical value is then broadcast across all 100 bins of each retained trial.

ii. ```python
pleft = np.round(tr.probabilityLeft.to_numpy(float)[ti], 1)
prior = np.array([{.2:0,.5:1,.8:2}[float(x)] for x in pleft], np.int8)
```

```python
np.full(len(CENTERS),s['prior'][j],np.int8)
```

iii. The trajectory justification was that the decoder task explicitly requested this 3-class encoding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel object's `timestamps` and `position` arrays.

ii. ```python
wheel = one.load_object(eid, 'wheel', collection='alf')
wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
ws, _ = velocity_filtered(wpos, 1000)
```

iii. The trajectory explicitly says it wanted the standard IBL wheel-processing path: interpolate position, then compute filtered velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to 1 kHz, computes low-pass filtered velocity with `velocity_filtered`, takes the absolute value, averages that speed within each stimulus-aligned 20 ms bin using `binned_mean`, fills short NaN gaps within a trial by interpolation, and later discretizes the result session-wise into tertiles.

ii. ```python
wpos, wt = interpolate_position(np.asarray(wheel.timestamps), np.asarray(wheel.position), freq=1000)
ws, _ = velocity_filtered(wpos, 1000)
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
```

iii. In the trajectory, the AI justified this as the closest available standard IBL processing after discovering that the installed brainbox version lacked the exact helper it first expected.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The wheel-speed bins are session-wise tertiles. The AI computes the 1/3 and 2/3 quantiles over all finite wheel-speed bins in that session and applies `np.digitize`. If both quantiles coincide, it adds a tiny deterministic ramp before recomputing thresholds.

ii. ```python
def tertiles(x):
    finite = x[np.isfinite(x)]
    q = np.quantile(finite, [1/3, 2/3])
    if q[0] == q[1]:
        q = np.quantile(finite + np.linspace(0, 1e-7, finite.size), [1/3, 2/3])
    return np.digitize(x, q, right=False).astype(np.int8), q.tolist()

wheel_c, wheel_q = tertiles(wheel_b)
```

iii. The trajectory says this was the AI's chosen task-specific discretization because the original paper decoded continuous wheel variables, while the current task required three categorical bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is expressed on the same 100 onset-relative bins as the neural data. For each trial onset, the AI computes 20 ms bin averages over the `[onset-0.5, onset+1.5)` window.

ii. ```python
ix = np.searchsorted(times, onset + EDGES)
...
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
```

iii. The trajectory repeatedly states that all decoder streams should share the single stimulus-aligned 20 ms grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI derives whisker motion energy from `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy` only. It does not fall back to right-camera motion energy when left-camera data are missing.

ii. ```python
mef = newest(alf, 'leftCamera.ROIMotionEnergy.npy')
tf = newest(alf, '_ibl_leftCamera.times.npy')
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
```

iii. The trajectory says the AI chose the left camera because it is the high-frame-rate side view used for whisker-pad motion, and later accepted session drops when those files were missing.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace and timestamps, truncates them to the shared minimum length, computes mean motion energy within each stimulus-aligned 20 ms bin, fills short NaN gaps within trials by interpolation, and later discretizes the resulting trace session-wise into tertiles.

ii. ```python
me, mt = np.load(mef, mmap_mode='r'), np.load(tf, mmap_mode='r')
n = min(len(me), len(mt))
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
...
whisk_c, whisk_q = tertiles(whisk_b)
```

iii. The trajectory did not give a separate whisker-specific rationale beyond reusing the same time-grid and tertile strategy as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: session-wise tertiles computed over all finite whisker-motion-energy bins, with the same tie-breaking fallback if both thresholds are equal.

ii. ```python
whisk_c, whisk_q = tertiles(whisk_b)
```

iii. The trajectory presents this as a consistent task-driven discretization for the two continuous time-varying behavioral outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to stimulus onset and summarized on the same 100 onset-relative 20 ms bins as the neural data.

ii. ```python
ix = np.searchsorted(times, onset + EDGES)
...
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
```

iii. The trajectory justification matches the wheel signal: use one common stimulus-aligned binning grid for all streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed inputs are mostly handled by dropping data. Missing required trial columns or too few surviving trials raise exceptions that skip the session. Missing left-camera files also skip the session. Missing/invalid behavior samples within a trial are averaged into bins, then short NaN gaps are interpolated; trials still containing NaNs afterward are dropped. Missing or unusable probes are handled by continuing past them, but a session with no surviving neurons or no qualifying region is skipped.

ii. ```python
def newest(path, pattern):
    fs = list(path.rglob(pattern))
    if not fs:
        raise FileNotFoundError(f'{pattern} below {path}')
```

```python
if not probe_data:
    raise ValueError('no well-isolated grey-matter neurons')
...
if n_neurons == 0:
    raise ValueError('no region has >=5 well-isolated neurons')
...
except Exception as exc:
    failures.append((str(eid), f'{type(exc).__name__}: {exc}'))
```

```python
def fill_short_gaps(x):
    ...
    if ok.sum() >= 2:
        row[~ok] = np.interp(q[~ok], q[ok], row[ok])
```

iii. The trajectory explicitly describes this as pragmatic failure handling for a large staged dataset: keep going across sessions, record failures, and only retain trials/sessions with complete decoder-ready data.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant work is loading spike-sorting outputs for each probe and binning large spike trains into per-trial neural tensors. The trajectory first identified full-vector per-neuron scans as too slow, then replaced them with a more efficient per-trial accumulation scheme.

ii. ```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)
```

```python
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
    ...
    np.add.at(neural[j], (cc[ok][inside], bb[inside]), 1)
```

iii. In the trajectory, the AI repeatedly called spike loading the expensive part and explicitly optimized away the original per-neuron repeated scans.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious loops are per-trial loops in `binned_mean`, the per-trial spike-binning loop over `onsets`, and the per-trial block-counter loop over `len(tr)`. These loops are straightforward but not fully vectorized.

ii. ```python
for i, onset in enumerate(onsets):
    ix = np.searchsorted(times, onset + EDGES)
    ...
```

```python
for j, onset in enumerate(onsets):
    lo, hi = np.searchsorted(st, [onset + OFF0, onset + OFF1])
    ...
```

```python
for j in range(len(tr)):
    if j == 0 or probs[j] != probs[j-1] or not np.isfinite(probs[j-1]): k = 0
```

iii. The trajectory acknowledges exactly this tradeoff: some loops could be vectorized further, but it prioritized clarity first and only optimized the clearly expensive spike path after a smoke test.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several small pieces of work: `binned_mean` sorts and cumulative-sums each behavioral stream separately every session; the allowed-region membership check is recomputed probe by probe; and onset-relative search/binner logic is rerun independently for wheel, whisker, and spikes. It also repeatedly constructs broadcast vectors for trial-level variables when assembling inputs and outputs.

ii. ```python
order = np.argsort(times); times, values = times[order], values[order]
cs = np.r_[0., np.cumsum(values)]
...
wheel_b = fill_short_gaps(binned_mean(wt, np.abs(ws), onsets))
...
whisk_b = fill_short_gaps(binned_mean(mt[:n], me[:n], onsets))
```

```python
use = np.isin(acr, list(allowed)); gids, acr = gids[use], acr[use]
...
np.full(len(CENTERS),s['choice'][j],np.int8)
np.full(len(CENTERS),s['prior'][j],np.int8)
```

iii. The trajectory only called out the repeated spike scanning explicitly; the remaining repetition is visible in the final code rather than separately justified.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several byproducts that the downstream decoder does not use: wheel and whisker tertile thresholds in metadata, source trial indices, skipped-session logs, and rich `session_info`. It also filters on `feedbackType` and `feedback_times` even though neither variable is used in the exported inputs or outputs.

ii. ```python
needed = ['choice','probabilityLeft','feedbackType','feedback_times',
          'stimOn_times','firstMovement_times']
```

```python
wheel_c, wheel_q = tertiles(wheel_b)
whisk_c, whisk_q = tertiles(whisk_b)
...
session_info=[dict(eid=s['eid'],source_trial_indices=s['trials'].tolist(),
                   wheel_tertiles=s['wheel_q'],whisker_tertiles=s['whisk_q']) for s in kept],
skipped_sessions=failures
```

iii. The trajectory frames most of this as documentation and robustness rather than decoder necessity; it wanted to preserve thresholds and failure reasons for inspection while keeping the main exported tensors categorical.
