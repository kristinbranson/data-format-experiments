# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globbed every NWB file one directory below `data/sub-*`, sorted the paths, opened each with `NWBHDF5IO`, and retained sessions having at least two accepted trials. `--sample` limits this to the first two files; otherwise all 152 files are processed.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
return files[:2] if sample else files
...
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
...
if len(sess['neural']) >= 2:
    sessions.append(sess)
```

iii. The notes justify this from the observed layout: 152 NWB session files in 11 subject directories, with one behavioral/ophys NWB per session. The full-run counts were treated as evidence that all released data were loaded.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from `nwb.subject.subject_id` (falling back to the parent directory name), deduplicated and sorted. Each session gets an index into that list.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64)
```

iii. The agent observed 11 `sub-*` directories/subjects and confirmed that the converted result retained 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and produces one entry in the outer `neural`, `input`, and `output` lists, provided it has at least two trials.

ii.
```python
for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```

iii. The notes say each `*_behavior+ophys.nwb` contains both modalities for a single session; all 152 files survived conversion.

## 1-d. How are the data split into trials?

i. Every positive `trial_start` sample begins a trial. Its end is the next start, while the last trial extends to the end of the recording. Teleport events are loaded but not used for boundaries.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
for i, s in enumerate(starts):
    e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
    if e > s:
        bounds.append((s, e))
```

iii. The agent reasoned that the representative file had matching counts of trial-start and teleport impulses, but chose successive `trial_start` events as a simple, trial-aligned delimiter.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if shorter than two samples, if all stored trial numbers are negative, or if all positions are at/below the invalid sentinel threshold. Sessions with fewer than two surviving trials are dropped. No reference minimum-duration filter is used.

ii.
```python
if e - s < 2:
    continue
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue
```

iii. The notes describe these as baseline/invalid-period exclusions and cite the target format’s two-trial session requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is taken directly from the NWB `processing['ophys']['Deconvolved'].roi_response_series['plane0']` dataset.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The agent said the methods use deconvolved activity and therefore treated the stored NWB deconvolution as the primary neural signal.

## 2-b. How is the `neural` data processed?

i. No fluorescence preprocessing or new deconvolution is done. Time slices of the stored matrix are transposed from time-by-ROI to ROI-by-time and cast to `float32`.

ii.
```python
trial_neural = neural[s:e, :].T.astype(np.float32)
sess_neural.append(trial_neural)
```

iii. The notes characterize this as consistent with the paper’s use of deconvolved activity and emphasize that it is fast and preserves the raw NWB values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered. Every column in the stored `Deconvolved/plane0` matrix is retained; `iscell` curation and speed-correlated putative-interneuron removal are absent.

ii.
```python
n_neurons = neural.shape[1]
region_idx = np.zeros(n_neurons, dtype=np.int64)
```

iii. The agent reported matching the 260,091 raw ROI columns as a successful consistency check. Its notes mention paper curation rules but do not justify omitting the applicable cell filters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials start at the same array index as the positive `trial_start` sample, so sample zero is the alignment event.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The agent states that the native behavior and neural grids are synchronized and that indexing both with the same trial slice aligns them to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Metadata uses `1000 / rate` ms from the first accepted session (about 64.5 ms at 15.5078125 Hz), while trials retain native samples.

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
...
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. The notes say behavior and stored deconvolved traces share the native ~15.5 Hz grid, so no resampling was thought necessary.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to `Deconvolved/plane0`; if absent, timestamps are synthesized from its rate.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
```

iii. The agent assumed neural and behavior streams share this time grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial’s first sample is subtracted from every timestamp in that trial and the result is cast to `float32`.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. This directly realizes “time from start of trial,” with every trial beginning at zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[s:e]` indices are used for timestamps and neural rows, with no explicit timestamp equality check or interpolation.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The agent’s raw-versus-converted sanity check confirmed exact matching for one representative session under its shared-grid assumption.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` series.

ii.
```python
env = np.asarray(behavior['environment'][0]).ravel()
env_valid = env[s:e][env[s:e] >= 0]
```

iii. The agent observed task codes 0/1 and baseline code -1 and interpreted 0/1 as ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded, the median of remaining samples is rounded to an integer (default 0 if none), then broadcast across the trial.

ii.
```python
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
np.full(e - s, env_label, dtype=np.float32)
```

iii. The agent expected environment to be constant within a trial and used a robust per-trial summary.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavioral `trial number` time series, not the loop counter except as a fallback.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
```

iii. The notes planned to use the NWB within-session trial label and exclude its negative baseline values.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median nonnegative stored value is used, or `ti` if none exists, and is broadcast over time.

ii.
```python
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
np.full(e - s, tr_label, dtype=np.float32)
```

iii. The agent treated it as a constant per-trial continuous label.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from timestamps of the behavioral `Reward` series and the deconvolved timestamp interval for each trial.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel()
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. The agent notes that reward is an event stream rather than a dense framewise label, so event presence within a trial defines its outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are appended as trials are processed. The preceding appended outcome is used, with 0 for the first accepted trial, and broadcast across time.

ii.
```python
reward_outcomes.append(rew)
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
np.full(e - s, prev_rew, dtype=np.float32)
```

iii. This implements omitted=0/rewarded=1 and the documented first-trial default. Because filtering occurs during this loop, it means the previous *retained* trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral `position` and `reward_zone`; reward timestamps and position provide a fallback. A trial’s zone is inferred from the median physical position of samples carrying its modal positive raw zone code, collapsed to A/B/C.

ii.
```python
zone_centers = reward_zone_centers(position, reward_zone)
code = int(np.bincount(rz_nz.astype(int)).argmax())
center = zone_centers.get(code, np.nan)
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The agent observed raw zone codes 1–6 but three physical locations, so it chose physical position instead of raw code. It acknowledged that the sample’s missing exact-zero distance class suggested this needed refinement.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The inferred A/B/C label selects a hard-coded point (90, 205, or 325 cm), and signed distance is `position - point`. This is distance to a center-like location, not zero distance throughout the reward-zone interval.

ii.
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
```

iii. The agent described this as reward-relative distance, based on approximate low/mid/high physical zone locations.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. A scalar loop assigns seven categories at -50, -10, 0, 10, and 50 cm. Exact zero is class 3; positive 0–10, including 10, is class 4.

ii.
```python
if d < -50: return 0
if d < -10: return 1
if d < 0: return 2
if d == 0: return 3
if d <= 10: return 4
if d <= 50: return 5
return 6
```

iii. The thresholds were intended to reproduce the requested category definitions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with identical `[s:e]` indices.

ii.
```python
d = position[s:e] - zc
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The agent assumes all dense streams share a synchronized native grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` series.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
out_pos = pos_bins(position[s:e])
```

iii. The notes identify position as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped into `[0, 450)` and digitized using internal edges 90, 180, 270, and 360 cm.

ii.
```python
edges = np.linspace(lo, hi, 6)
out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
```

iii. The agent followed the specified five equal bins over a 450 cm track; clipping keeps out-of-range samples in an endpoint class.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Categories are 0 for clipped values below 90, 1 for 90–<180, 2 for 180–<270, 3 for 270–<360, and 4 for 360–450 cm.

ii.
```python
edges = np.linspace(0.0, 450.0, 6)
np.digitize(np.clip(pos, 0.0, 450.0 - 1e-6), edges[1:-1], right=False)
```

iii. These are the requested equal-width 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial indices slice position and neural activity.

ii.
```python
out_pos = pos_bins(position[s:e])
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. No additional alignment was considered necessary because the streams were assumed synchronized.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavioral `lick` series.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
```

iii. The agent identifies this series as per-sample lick counts/events.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every value greater than zero becomes 1; all others become 0.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. This converts possible counts above one into the requested binary no/yes output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural activity use the same `[s:e]` slice.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The agent’s representative raw-data check found the converted lick vector identical to this raw slice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived primarily from behavioral `reward_zone` plus `position`; an in-trial reward position is the fallback when no positive zone samples exist, and median trial position is the final fallback.

ii.
```python
rz_nz = reward_zone[s:e][reward_zone[s:e] > 0]
...
center = float(position[ridx])
...
center = np.nanmedian(position[s:e])
```

iii. The agent reasoned that physical locations are more meaningful than the six raw codes and map naturally to three requested A/B/C classes.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The inferred representative position is classified A below 150 cm, B below 260 cm, otherwise C, then broadcast across every sample in the trial.

ii.
```python
if pos < 150: return 0
if pos < 260: return 1
return 2
...
out_rz = np.full(e - s, zone_label, dtype=np.int64)
```

iii. The cutoffs were chosen to separate the observed low/middle/high physical clusters. The notes cite approximate centers around 80–100, 200, and 320–330 cm.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the timestamps of behavioral `Reward` events.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel()
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. The agent followed the paper’s rewarded/omission distinction and treated any in-trial reward event as rewarded.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Event timestamps are compared directly with the first and last deconvolved timestamps in the trial. The resulting binary scalar is broadcast over the trial.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. The notes report a representative ten-trial check against raw reward timestamps and say all outcomes matched.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several fallbacks are used: absent timestamps are synthesized from rate; unknown region becomes `unknown`; malformed region lookup is caught broadly; missing valid environment defaults to 0; missing trial number defaults to loop index; missing zone evidence falls back to reward position, median trial position, then a 225 cm default. Very short/sentinel trials are skipped. There is no general neural/behavior length reconciliation, NaN policy, or reward-alignment assertion.

ii.
```python
except Exception:
    return 'unknown'
...
env_label = ... if len(env_valid) else 0
tr_label = ... if len(tr_valid) else float(ti)
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The agent describes these as defensive handling of baseline codes, invalid position periods, and absent reward-zone metadata; successful format validation was used as the main evidence.

## 13-a. What are the most time-consuming steps of the code?

i. The agent identified repeated full NWB reads as the primary cost; the produced 24 GB pickle also makes serialization substantial. Its measured conversion was about 0.7 seconds/session, while decoder training was a separate expensive downstream step.

ii.
```python
for i, f in enumerate(files, 1):
    sess = process_file(f)
...
with open(args.outpickle, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes explicitly say large NWB reads are slow and estimate a two-to-three-minute full conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The scalar distance and speed binning comprehensions could be replaced with `np.digitize`/vectorized comparisons. Trial iteration is natural for ragged output, though some per-session labels and outcomes could be precomputed. The file loop could also be parallelized, subject to memory/I/O limits.

ii.
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. The agent claimed “vectorized slicing and single-pass per-session processing,” but did not document these two avoidable per-timepoint Python loops.

## 13-c. What processing does the code repeat multiple times?

i. It loads all listed behavior streams even though `teleport`, `autoreward`, and `scanning` are never consumed. Within every trial, it repeatedly slices the same arrays and allocates full-length broadcasts. Reward timestamps are rescanned for every trial, and zone centers scan full session arrays once per raw code.

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. The notes only flag repeated full NWB reads as a possible concern and otherwise describe the implementation as single-pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `teleport`, `autoreward`, and `scanning` data and timestamps are loaded but unused; `trial_zone_labels` is populated but never returned; `session_id` is returned but omitted from final metadata; the initial all-zero per-neuron `region_idx` is replaced later; and `--show-processing` is a no-op placeholder.

ii.
```python
trial_zone_labels = []
trial_zone_labels.append(zone_label)
...
'session_id': sess_id,
...
region_idx = np.zeros(n_neurons, dtype=np.int64)
```

iii. The notes acknowledge that `--show-processing` creates no plots, but do not discuss the other discarded work.
