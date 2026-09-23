# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively discovers every subject NWB session with a sorted glob, then loads each session through `pynwb.NWBHDF5IO`. Full conversion parallelizes independent sessions with a process pool; sample mode selects two representative files.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
    for res in ex.map(_worker, remaining):
        results.append(res)
```

iii. The agent documented that all 152 public NWB files (11 mice) are included, that `pynwb` is used as required, and that multiprocessing reduces wall time while sessions remain independent.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `nwb.subject.subject_id`; unique IDs are naturally sorted numerically, and each session receives an index into that list.

ii.
```python
d['subject'] = nwb.subject.subject_id
subjects = sorted({inf['subject'] for inf in infos}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(inf['subject']) for inf in infos])
```

iii. The notes report the expected 11 mouse IDs and prefer NWB metadata over parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and becomes one top-level entry, unless it has fewer than two usable trials.

ii.
```python
for (n, i, o, info) in results:
    if len(n) < 2:
        continue
    neural.append(n); inputs.append(i); outputs.append(o); infos.append(info)
```

iii. The agent says all 152 available sessions are used because decoder variables exist on stay and switch days; the paper’s switch-day restriction served a different analysis.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and stops are positive `teleport` samples. The reference’s one-based convention is reproduced as the half-open slice `[start-1:stop-1)`.

ii.
```python
d['trial_starts'] = np.where(np.asarray(beh['trial_start'].data[:]) > 0)[0]
d['teleports'] = np.where(np.asarray(beh['teleport'].data[:]) > 0)[0]
sl = slice(s - 1, e - 1)
```

iii. The notes cite `glmUtils.get_timeseries_data`, exclude the teleport/ITI, and validate equal start/stop counts and ordering.

## 1-e. How are trials filtered based on quality controls?

i. Trials with lick-sensor failure are removed; additionally, trials shorter than two samples or containing nonfinite neural events are skipped. Sessions with fewer than two retained trials are excluded.

ii.
```python
lick_error[i] = lk.size > 0 and (np.sum(lk > 2) / lk.size) > .30
if lick_error[i]: continue
if T < 2: continue
if not np.all(np.isfinite(ev_trial)): continue
```

iii. The paper’s >30% cumulative-lick criterion flags exactly 81 trials. Since categorical lick outputs cannot carry NaN, the agent removes whole affected trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is recomputed from NWB `Fluorescence` (F) and `Neuropil` (Fneu), pooled across planes, rather than using NWB `Deconvolved`.

ii.
```python
Fs.append(np.asarray(oph['Fluorescence'].roi_response_series[pl].data[:]))
Fneus.append(np.asarray(oph['Neuropil'].roi_response_series[pl].data[:]))
```

iii. The agent determined that stored `Deconvolved` is Suite2p output on raw fluorescence, whereas the paper analyzes OASIS events derived from its own dF/F pipeline.

## 2-b. How is the `neural` data processed?

i. It applies 0.7 neuropil subtraction with trial-mean addback, Gaussian/maximin baseline (sigma 15, 300-sample min then max), absolute-baseline dF/F, sigma-2 smoothing, and OASIS (`tau=.7`) per trial. Teleport-spanning baseline segments are used on the experiment days specified by paper metadata.

ii.
```python
f_ -= NEU_COEF * fneu_
seg = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH])
seg = minimum_filter1d(seg, BASELINE_WINDOW, axis=-1)
seg = maximum_filter1d(seg, BASELINE_WINDOW, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl]), 2000, TAU, fs)
```

iii. These operations and parameters are explicitly ported from `preprocessing.dff`, the Methods, Suite2p settings, and teleport metadata.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only manually curated `iscell==1` ROIs are used, then cells with dF/F–speed Pearson correlation above 0.5 are removed as putative interneurons.

ii.
```python
F = raw['F'][raw['iscell']]
speed_corr = (dm_c @ sp_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
events = events[~is_int]
```

iii. Both filters are attributed to the paper’s cell-curation procedure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials start at the trial-start sample and are sliced through the sample before teleport; time zero is therefore trial start.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
```

iii. The notes identify trial start/track entry as the alignment event and specify `off_start=0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native per-plane sampling is retained without rebinning: `scan_rate/n_planes`, approximately 15.5078 Hz or 64.48 ms.

ii.
```python
d['fs'] = d['scan_rate'] / len(planes)
'time_bin_size': float(1000.0 / infos[0]['fs'])
```

iii. Neural and behavioral streams already share this clock; preserving it matches the paper and keeps contiguous decoder samples.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps of the raw `position` behavioral series and the trial-start indices.

ii.
```python
d['time'] = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
t_rel = time_v[sl] - time_v[s - 1]
```

iii. The notes state that all behavior and neural streams are on the same sampling clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial.

ii.
```python
t_rel = time_v[sl] - time_v[s - 1]
inp[0] = t_rel
```

iii. This makes the aligned sample exactly zero and expresses later samples in seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamp vector and neural events use the identical half-open trial slice. A one-frame neural/behavior mismatch is truncated to the common length before conversion.

ii.
```python
n = min(nB, nF)
d['time'] = d['time'][:n]; d['F'] = d['F'][:, :n]
ev_trial = events[:, sl]
t_rel = time_v[sl] - time_v[s - 1]
```

iii. The agent identifies this mismatch as a known scan-stopping correction and reports that no trial boundary is lost.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` time series.

ii.
```python
d['env'] = np.asarray(beh['environment'].data[:], dtype=np.float64)
```

iii. The notes map the raw binary values to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative samples are discarded; the rounded within-trial median is calculated and broadcast across the trial (default zero only if no valid values exist).

ii.
```python
ev = env[sl]; ev = ev[ev >= 0]
env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0
inp[1] = env_trial[i]
```

iii. The agent treats environment as a per-trial constant and uses a robust summary rather than trusting one frame.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the ordered trial-start/teleport pairs, using their zero-based loop index rather than the loaded raw `trial number` series.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    inp[2] = i
```

iii. The notes define trial number as the within-session zero-based index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond broadcasting the index across all samples of the trial.

ii.
```python
inp[2] = i
```

iii. This supplies the required continuous per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It ultimately comes from `Reward` timestamps and the `reward_zone` behavioral series used to determine each preceding trial’s rewarded flag.

ii.
```python
d['reward_times'] = np.asarray(beh['Reward'].timestamps[:])
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
isreward[i] = int(bool(has_rew) and bool(in_zone))
```

iii. The agent says this ports `behavior.get_trial_types`, where reward delivery and reward-zone entry jointly define a rewarded trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are mapped to behavioral frame indices. Trial `i` receives `isreward[i-1]`, broadcast in time; the first trial receives zero. Filtering does not renumber this relationship.

ii.
```python
rew_frames = np.searchsorted(time_v, raw['reward_times'])
inp[3] = isreward[i - 1] if i > 0 else 0
```

iii. Zero is documented as the conservative encoding when the previous, un-imaged warm-up trial is unknown.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone bounds inferred from the NWB scene identifier and the trial index (including the switch after trial 30).

ii.
```python
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
z0, z1 = ZONE_DICT[zlab]
rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. The agent cites `behavior.get_reward_zones` and reports that inferred labels matched observed `reward_zone>0` positions on every trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the zone, position minus its start before it, and position minus its end after it; that continuous value is then categorized.

ii.
```python
d[before] = pos[before] - zone_start
d[after] = pos[after] - zone_end
```

iii. This is distance to the nearest point in the zone and follows the task’s sign convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks produce seven classes: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
cat[d < -50] = 0
cat[(d >= -50) & (d < -10)] = 1
cat[(d >= -10) & (d < 0)] = 2
cat[d == 0] = 3
cat[(d > 0) & (d <= 10)] = 4
cat[(d > 10) & (d <= 50)] = 5
cat[d > 50] = 6
```

iii. The explicit masks implement the requested boundary semantics, particularly the separate exact-zero class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and events use the same trial slice, so the category vector has the same `T` as neural data.

ii.
```python
ev_trial = events[:, sl]
p = pos[sl]
out[0] = rz_cat
```

iii. No interpolation is needed because the NWB streams are sample-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` series.

ii.
```python
d['pos'] = np.asarray(beh['position'].data[:], dtype=np.float64)
p = pos[sl]
```

iii. Position is already measured in centimeters along the 450 cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position samples are digitized directly; no smoothing or resampling is applied.

ii.
```python
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
out[1] = pos_cat
```

iii. Native values preserve the paper’s spatial coordinate.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges 90, 180, 270, and 360 cm creates five 90 cm classes, with under/overflow absorbed by the end classes.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
pos_cat = np.digitize(p, POS_EDGES)
```

iii. These are five equal divisions of the specified 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial slice is used for position and neural events.

ii.
```python
ev_trial = events[:, sl]
p = pos[sl]
```

iii. The shared NWB sampling clock makes index alignment sufficient.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavioral `lick` series.

ii.
```python
d['lick'] = np.asarray(beh['lick'].data[:], dtype=np.float64)
lk = lick[sl]
```

iii. The raw series contains per-frame cumulative lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive value is binarized to one. Trials whose sensor signal is corrupt by the paper’s rule are removed first.

ii.
```python
lick_error[i] = (np.sum(lk > LICK_ERROR_COUNT) / lk.size) > LICK_ERROR_FRAC
lick_cat = (lk > 0).astype(np.int64)
```

iii. The task requires a binary output, and the paper states remaining lick counts were converted to a binary vector.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data are extracted with the identical half-open trial slice.

ii.
```python
ev_trial = events[:, sl]
lk = lick[sl]
out[3] = lick_cat
```

iii. No separate temporal conversion is required.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene string in `nwb.identifier` plus trial order, not directly from framewise reward-zone values.

ii.
```python
d['scene'] = nwb.identifier.split('/')[-1]
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
```

iii. The agent ports the paper repository’s scene-name logic and checks it against raw reward-zone observations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed-location scenes map entirely to A/B/C; switch scenes use their first zone for trials 0–29 and second afterward. A/B/C map to 0/1/2 and are broadcast across the trial.

ii.
```python
return [first] * min(30, ntrials) + [second] * (ntrials - min(30, ntrials))
zone_code = {'A': 0, 'B': 1, 'C': 2}
out[4] = zone_code[zlab]
```

iii. This matches the experimental design and `behavior.get_reward_zones`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It derives from `Reward` event timestamps and the framewise `reward_zone` series.

ii.
```python
rew_frames = np.searchsorted(time_v, raw['reward_times'])
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
```

iii. The conjunction is attributed to the paper code’s `get_trial_types` definition.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is one only if a mapped reward event occurs within its slice and the reward-zone signal becomes positive; otherwise zero. The result is broadcast across time.

ii.
```python
isreward[i] = int(bool(has_rew) and bool(in_zone))
out[5] = isreward[i]
```

iii. This produces the expected approximately 15% omission rate and guards against spurious reward timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior arrays differing by one sample are truncated jointly; out-of-range reward frames are discarded; boundaries are clipped/reconciled; invalid environment samples are ignored; nonfinite or tiny trials are skipped; and sessions need two trials.

ii.
```python
n = min(nB, nF)
rew_frames = rew_frames[rew_frames < len(time_v)]
d['trial_starts'] = d['trial_starts'][:len(d['teleports'])]
if T < 2 or not np.all(np.isfinite(ev_trial)): continue
```

iii. The one-frame mismatch is documented as known acquisition behavior. Defensive checks preserve sample alignment and prevent malformed decoder entries.

## 13-a. What are the most time-consuming steps of the code?

i. Reading full F/Fneu arrays and per-trial dF/F/OASIS processing dominate; serialization is secondary. Sessions are parallelized.

ii.
```python
dff, events = compute_events(F, Fneu, starts, stops, raw['fs'], ...)
with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
```

iii. Notes measured roughly 0.3–1.5 s loading and up to 12 s neural processing per large session, motivating 12 workers.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops for reward/lick/environment summaries, neural preprocessing, and assembly remain; plane loading and plotting also loop. Some could be masked/vectorized, though variable boundaries and OASIS make trial loops natural. Interneuron correlations were already vectorized.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
speed_corr = (dm_c @ sp_c) / denom
```

iii. The notes specifically say the original per-cell correlation loop was replaced by matrix-vector computation; independent sessions were parallelized instead.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trial boundaries repeatedly: once to build baseline/event segments, once to calculate trial-level behavior/QC, and once to assemble outputs. Plot mode traverses selected trials again.

ii.
```python
for s, e in zip(starts, stops):  # compute_events baseline/deconvolution
for i, (s, e) in enumerate(zip(starts, stops)):  # QC summaries
for i, (s, e) in enumerate(zip(starts, stops)):  # assembly
```

iii. This separation mirrors distinct processing stages and avoids retaining larger intermediate structures, though common trial slices could be cached.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/stores unused raw fields (`trialnum`, `scanning`, `planeIdx`) and computes some metadata/timing; `discretize_reward_distance` returns a continuous distance that conversion discards. Plot generation, when requested, recomputes continuous traces solely for diagnostics.

ii.
```python
d['trialnum'] = np.asarray(beh['trial number'].data[:])
d['scanning'] = np.asarray(beh['scanning'].data[:])
d['planeIdx'] = np.asarray(ps['planeIdx'].data[:])[roi_idx]
rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. These fields support inspection/provenance or optional plots but are not consumed by decoder training; the agent otherwise emphasizes float32 and limited intermediates.
