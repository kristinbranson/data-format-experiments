# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively enumerates every `sub-*/*.nwb` file, sorts the 152 paths, and loads each session with `pynwb.NWBHDF5IO`; full conversion is the default and multiprocessing distributes sessions.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
with ctx.Pool(min(args.nproc, len(files))) as pool:
    for k, res in enumerate(pool.imap(_worker, tasks)):
```

iii. The notes report 11 subject directories and 152 NWB files and say `pynwb` was used as required. Sample mode deliberately selects two files; the produced full dataset used all 152.

## 1-b. How are the data split into subjects?

i. Subject identity comes from each NWB file's `nwb.subject.subject_id`; unique IDs are naturally sorted numerically, and each session receives an index into that list.

ii.
```python
subject = nwb.subject.subject_id
subjects = sorted({r['info']['subject'] for r in results}, key=lambda s: int(s[1:]))
'subject_idx': np.array([subjects.index(r['info']['subject']) for r in results])
```

iii. The agent preferred authoritative NWB metadata and checked that it yielded the expected 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session, identified using `nwb.session_id`; each conversion result becomes one element of the session-level lists.

ii.
```python
session_id = nwb.session_id
'neural': [r['neural'] for r in results]
```

iii. This follows the dataset organization and preserves one independently recorded population per file.

## 1-d. How are the data split into trials?

i. Trial starts are samples where `trial_start == 1`, ends are samples where `teleport == 1`, but the stored window is shifted one frame earlier at both ends: `[start-1, teleport-1)`. Invalid boundary pairs are rejected.

ii.
```python
si = np.where(raw['trial_start'] == 1)[0]
ti = np.where(raw['teleport'] == 1)[0]
keep = (si >= 1) & (ti <= len(ts))
sl = slice(s - 1, e - 1)
```

iii. The notes say this is the precise window used by the paper's `preprocessing.dff` and `glmUtils.get_timeseries_data`, excluding the teleport jump.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped for the paper's lick-sensor error criterion, any non-imaging `scanning` sample, fewer than two neural samples, or nonfinite neural values. Out-of-range trial pairs are also excluded. Quality variables are computed before filtering so previous outcome remains tied to the actual preceding trial.

ii.
```python
lick_error[i] = (np.sum(L > 2) / len(L)) > 0.3
scan_bad[i] = np.any(scanning[s - 1:e - 1] != 1)
if lick_error[i] or scan_bad[i]: continue
if act.shape[1] < 2 or not np.all(np.isfinite(act)): continue
```

iii. The paper described the lick-error rule; dropping whole trials was justified because lick is a required non-NaN decoder output. The notes report exactly 81 such trials and no scan/NaN drops.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is derived from each plane's raw `Fluorescence` and `Neuropil` ROI response series, restricted at load time to manually curated `iscell` ROIs; the stored NWB `Deconvolved` series is not used.

ii.
```python
rrs = oph['Fluorescence'].roi_response_series[pn]
sel = iscell[rid]
f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
nrs = oph['Neuropil'].roi_response_series[pn]
fneu_list.append(np.asarray(nrs.data[:, :], dtype=np.float32)[:, sel].T)
```

iii. The agent determined that NWB `Deconvolved` was Suite2p raw-F deconvolution, whereas the paper recomputed dF/F and events from F and Fneu.

## 2-b. How is the `neural` data processed?

i. F is corrected by `0.7*Fneu`, the per-trial neuropil mean is added back, a maximin baseline (Gaussian sigma 15, 300-frame minimum then maximum) is computed, dF/F is formed and Gaussian-smoothed at sigma 2. OASIS events are also computed, but the full delivered conversion defaults to and stores dF/F, not events. Planes are pooled.

ii.
```python
f_ -= neu_coef * f_neu_
base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
spks[:, sl] = dcnv.oasis(..., tau, frame_rate)
act = (events if signal == 'events' else dff)[:, sl]
```

iii. The algorithm and parameters were taken from the authors' preprocessing. Although the notes initially call events the preferred signal, the agent ultimately chose dF/F because it decoded substantially better for the supplied per-timepoint linear decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `iscell==1` ROIs are loaded. Putative interneurons are removed when Pearson correlation between whole-session in-trial dF/F and speed exceeds 0.5. Nonfinite trials are dropped.

ii.
```python
sel = iscell[rid]
r_speed = (d_c @ sp_c) / denom
keep_cells = r_speed <= INTERNEURON_SPEED_CORR_THR
dff = dff[keep_cells]
events = events[keep_cells]
```

iii. Both cell filters reproduce the Methods: manual Suite2p curation followed by the stated speed-correlation criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural and behavior already share the imaging-frame grid. Each neural trial uses `[trial_start-1, teleport-1)`, and time zero is the `trial_start` sample, so the first stored neural column is one frame before time zero.

ii.
```python
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
inp[0] = ts[sl] - ts[s]
```

iii. This was justified as exact reference-code alignment; diagnostic overlays were inspected for temporal shifts.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native, already aligned imaging frames are retained without rebinning. The median interval is about 64.4836 ms (about 15.5 Hz) and is recorded in metadata.

ii.
```python
dt = float(np.median(np.diff(ts)))
'time_bin_size': float(np.median(dts) * 1000.0)
```

iii. The agent verified all sessions share this interval and stated that the NWB behavior was already interpolated to imaging frames.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses the timestamps attached to the raw `position` BehavioralTimeSeries and the raw `trial_start` indices.

ii.
```python
ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
si = np.where(raw['trial_start'] == 1)[0]
```

iii. Position timestamps define the common 2P-aligned behavioral grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the flagged start sample is subtracted from all samples in the shifted trial window; consequently the first value is about -0.0645 s and the next is zero.

ii.
```python
inp[0] = ts[sl] - ts[s]
```

iii. The notes explicitly document the negative first sample and `off_start` because of the reference window.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is indexed by the identical slice and has the same T as the neural matrix.

ii.
```python
T = act.shape[1]
inp = np.empty((4, T), dtype=np.float32)
inp[0] = ts[sl] - ts[s]
```

iii. No interpolation is required because both streams share the frame grid; extra trailing frames are trimmed first.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw `environment` BehavioralTimeSeries (called `morph` internally).

ii.
```python
morph = raw['environment']
mvals = np.unique(morph[s:e])
```

iii. The agent mapped the observed 0/1 values to ENV1/ENV2 and cross-checked them against session scenes.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. If a trial has one value, that integer is used; otherwise the rounded median is used. It is broadcast across every trial timepoint.

ii.
```python
trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
inp[1] = trial_morph[i]
```

iii. Environment is intended to be a binary per-trial context and was observed to be constant in normal trials.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes from the zero-based enumeration of detected `trial_start`/`teleport` pairs, not the NWB `trial number` values (though that raw series is loaded).

ii.
```python
for i, (s, e) in enumerate(zip(si, ti)):
    inp[2] = i
```

iii. The notes identify this with the reference code's `trial_ids` and preserve original trial numbers even when a trial is dropped.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform is applied beyond using the zero-based session trial index and broadcasting it over time.

ii.
```python
inp[2] = i
```

iii. This directly supplies the requested continuous per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Sparse `Reward` timestamps are mapped onto the position timestamp grid; current reward additionally requires a positive raw `reward_zone` flag. Previous outcome is the preceding detected trial's resulting binary value.

ii.
```python
reward_idx = np.searchsorted(ts, raw['reward_times'])
has_reward = np.any((reward_idx >= s) & (reward_idx < e))
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
prev_reward[1:] = isreward[:-1]
```

iii. This ports the paper's trial-type logic; computing it before QC prevents a removed trial from changing what “previous” means.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcome is shifted by one original trial; the first trial is assigned 0, then the scalar is broadcast across time.

ii.
```python
prev_reward = np.zeros(ntrials_all, dtype=np.int64)
prev_reward[1:] = isreward[:-1]
inp[3] = prev_reward[i]
```

iii. The first value is undefined because warm-up trials were not imaged, so the agent chose 0 and documented its small scope.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone identity inferred from the NWB scene/identifier; fixed scenes end in `LocationA/B/C`, while switch scenes use their encoded before/after zones with a switch after trial 30.

ii.
```python
scene = nwb.identifier.split('/')[-1]
zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. The scene rule ports `behavior.get_reward_zones`; the agent checked it against measured reward-zone entry positions on all 10,394 trials with a flag.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Zone ranges are A=80–130, B=200–250, C=320–370 cm. Distance is negative before the near edge, zero anywhere inside, and positive after the far edge, then categorical conversion is applied.

ii.
```python
d[before] = pos[before] - zone_start
d[after] = pos[after] - zone_end
```

iii. This implements signed distance to the nearest point in the active reward zone, matching the requested semantic.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit comparisons implement seven classes with exact zero isolated: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii.
```python
out = np.full(d.shape, 3, dtype=np.int64)
out[d < 0] = 2; out[d < -10] = 1; out[d < -50] = 0
out[d > 0] = 4; out[d > 10] = 5; out[d > 50] = 6
```

iii. The comparisons were chosen to reproduce the task's boundary inclusivity exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the exact same shifted trial slice as neural activity.

ii.
```python
p = pos[sl]
out[0] = discretize_distance(signed_distance_to_zone(p, ...))
```

iii. The common already-aligned sample grid makes further resampling unnecessary.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the raw `position` BehavioralTimeSeries.

ii.
```python
pos = raw['position']
p = pos[sl]
```

iii. The variable is already expressed in centimeters on the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Apart from common-length trimming and per-trial slicing, the raw position is only discretized.

ii.
```python
out[1] = np.digitize(p, POS_EDGES)
```

iii. No normalization is needed because the requested thresholds are in raw centimeters.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges 90, 180, 270, and 360 creates five 90 cm classes; values at an edge enter the higher class.

ii.
```python
POS_EDGES = (90.0, 180.0, 270.0, 360.0)
out[1] = np.digitize(p, POS_EDGES)
```

iii. These are five equal divisions of the specified 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity use the identical `sl` and T.

ii.
```python
act = ...[:, sl]
p = pos[sl]
```

iii. The NWB behavior is already on the imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` BehavioralTimeSeries.

ii.
```python
lick = raw['lick']
lk = lick[sl]
```

iii. The raw stream contains per-frame lick counts/cumulative sensor values.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive raw value becomes 1 and zero/nonpositive becomes 0. Separately, sensor-error trials are removed before output creation.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
```

iii. This follows the reference lick binarization and requested no/yes coding.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick values use the same per-trial slice and thus the same columns as neural activity.

ii.
```python
lk = lick[sl]
out[3] = (lk > 0).astype(np.int64)
```

iii. No interpolation is needed on the shared grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene name embedded in `nwb.identifier` and the original within-session trial index; raw reward-zone flags were used as a validation, not the primary source.

ii.
```python
scene = nwb.identifier.split('/')[-1]
zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
```

iii. The agent chose the authors' scene-parsing rule because omission trials lack the flag and reported perfect agreement where flags existed.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed scenes produce one label; switch scenes use the source zone for trials 0–29 and destination thereafter. A/B/C become 0/1/2 and are broadcast across time.

ii.
```python
n0 = min(change_trial, ntrials)
return np.array([first] * n0 + [last] * (ntrials - n0))
zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab])
out[4] = zone_code[i]
```

iii. This is described as a direct port of `behavior.get_reward_zones` and the experiment's after-30-trials switch.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse raw `Reward` timestamps and the raw `reward_zone` flag within each detected trial.

ii.
```python
reward_idx = np.searchsorted(ts, raw['reward_times'])
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
```

iii. This mirrors the authors' `get_trial_types`; the agent observed that the reward-zone condition was redundant in these NWBs but retained it.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if a reward timestamp maps into `[start, teleport)` and that interval contains a positive reward-zone flag, otherwise 0; the scalar is broadcast.

ii.
```python
has_reward = np.any((reward_idx >= s) & (reward_idx < e))
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
out[5] = isreward[i]
```

iii. The observed omission rate was checked against the paper's approximately 15% figure.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavioral streams are trimmed to their shorter common length (ten two-plane sessions had one extra imaging frame). Out-of-range boundaries, too-short trials, nonfinite neural trials, scan failures, and lick-sensor failures are excluded. Ambiguous within-trial environment falls back to rounded median; zero-variance correlations become zero.

ii.
```python
n = min(len(ts), f.shape[1])
ts = ts[:n]; f, f_neu = f[:, :n], f_neu[:, :n]
r_speed = np.nan_to_num(r_speed, nan=0.0)
if act.shape[1] < 2 or not np.all(np.isfinite(act)): continue
```

iii. The notes identify and quantify each edge case, emphasizing that trimming affects only an extra trailing frame and that no ordinary trial was lost to scan/NaN checks.

## 13-a. What are the most time-consuming steps of the code?

i. Per-session dF/F filtering/deconvolution dominates (about 10–16 seconds for roughly 1,000 cells), followed by loading Fluorescence/Neuropil (about 1–3.5 seconds) and writing the 9.6 GB pickle.

ii.
```python
t1 = time.time()
dff, events = dff_and_events(...)
t_dff = time.time() - t1
```

iii. The notes base this on recorded per-session timing and explain the use of float32 and session-level multiprocessing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over trial slices in `dff_and_events` and `convert_session`, plus repeated zone-label construction, could be partly vectorized. Variable-length trials and independent per-trial maximin filters limit full vectorization. Cell-speed correlations are already vectorized.

ii.
```python
for sl in slices:
    base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
for i, (s, e) in enumerate(zip(si, ti)):
    ...
r_speed = (d_c @ sp_c) / denom
```

iii. The agent focused optimization on higher-payoff choices: early ROI filtering, float32, vectorized cell correlations, and parallel sessions.

## 13-c. What processing does the code repeat multiple times?

i. Trial ranges are traversed once to mask/calculate dF/F, again to smooth/deconvolve, once to compute trial QC/task scalars, and again to assemble arrays. Reward-zone labels are translated into starts, ends, and codes separately. With diagnostic plotting, several trial variables and distances are recalculated.

ii.
```python
for sl in slices: f_[:, sl] = f[:, sl]
for sl in slices: ... flow[:, sl] = base
for sl in slices: ... spks[:, sl] = dcnv.oasis(...)
for i, (s, e) in enumerate(zip(si, ti)): ...
for i, (s, e) in enumerate(zip(si, ti)): ...
```

iii. The notes do not explicitly inventory repetition; they justify per-trial processing as necessary for reference-equivalent baselines and use optional plotting only for selected sessions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script always computes both dF/F and OASIS events even though only one signal is stored; in the delivered default dF/F run, all event computation is discarded. It also retains plane indices only transiently, computes extensive summary statistics, and conditionally generates diagnostics that are not decoder inputs.

ii.
```python
dff, events = dff_and_events(...)
act = (events if signal == 'events' else dff)[:, sl]
plane_of_cell = raw['plane_of_cell'][keep_cells]
events_frac_zero=float(np.mean(events[:, valid] == 0))
```

iii. The notes explain that both signals were compared to choose the decoder signal and that diagnostic calculations supported sanity checking; they do not claim these are required downstream.
