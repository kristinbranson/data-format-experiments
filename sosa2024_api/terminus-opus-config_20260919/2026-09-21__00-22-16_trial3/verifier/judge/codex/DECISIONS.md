# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all data by globbing every `.nwb` file under `/app/data/sub-*`, then processing each NWB as one session with `pynwb.NWBHDF5IO`. Within each session it reads both behavior and ophys interfaces from the NWB and constructs trials from the behavior streams.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))
...
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    oph = nwb.processing['ophys'].data_interfaces
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent says the dataset is 152 NWB files in 11 subject folders and that each NWB already contains the aligned behavior and ophys streams needed for conversion. Step 10 Check 3 says the NWB stores the same aligned streams the reference `sess` object would have contained, so direct NWB loading is sufficient.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the `subject_id` stored inside each NWB file, then uniqued and sorted numerically when assembling the final dataset.

ii. 
```python
subject = nwb.subject.subject_id
...
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. In Step 2 and Step 9, the notes state that the dandiset contains 11 mice (`m3` ... `m19`) and that `nwb.subject.subject_id` matches those released mouse identifiers, so using the NWB subject field is the authoritative split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is taken from the file being processed and recorded from `nwb.session_id`.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))
...
session_id = nwb.session_id
```

iii. The notes repeatedly describe the dataset as 152 NWB files corresponding to 152 sessions, and Step 9 reports successful conversion of 152 sessions, so the agent’s session split is one file per session.

## 1-d. How are the data split into trials?

i. Trials are defined as on-track laps from the frame where `trial_start` is positive to the frame where `teleport` is positive, using slices `[start:stop)`. Inter-trial/teleport periods are excluded.

ii. 
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ev = events[:, s:e]
    p = pos[s:e]
```

iii. Step 5 decision 3 in the notes says “Trial = [trial_start frame, teleport frame)” and justifies this by checking that `pos[start]` is at track entry and `pos[teleport]` is already ITI/jitter. Step 10 Check 3(c) says this is the correct NWB-space equivalent of the reference trial window.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops the first trial of every session, lick-sensor-error trials, pathologically short trials, trials with `scanning != 1`, and any trial containing non-finite neural or behavioral values. It also requires at least two kept trials per session.

ii. 
```python
if i == 0:
    n_drop_first += 1
    continue
if lick_error[i]:
    n_drop_lick += 1
    continue
if (e - s) < MIN_TRIAL_FRAMES:
    n_drop_short += 1
    continue
if np.any(scanning[s:e] != 1):
    n_drop_scan += 1
    continue
if (np.any(~np.isfinite(ev)) or np.any(~np.isfinite(p)) or
        np.any(~np.isfinite(sp)) or np.any(~np.isfinite(lk))):
    n_drop_nan += 1
    continue
...
results = [r for r in results if len(r['neural']) >= 2]
```

iii. Step 5 decision 5 says the first trial is dropped because `prev_trial_outcome` is undefined, lick-error trials are dropped because lick is a decoder output and NaNs are disallowed, and all sessions still retain at least two trials. Step 10 adds that no trials were dropped for scanning/non-finite issues in the full dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from raw suite2p fluorescence and neuropil traces (`Fluorescence`, `Neuropil`), plus the ROI curation fields `iscell` and `planeIdx`. It is not taken from the NWB’s stored `Deconvolved` series.

ii. 
```python
ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)
...
Fp = np.asarray(oph['Fluorescence'].roi_response_series[name].data[:]).T
Np = np.asarray(oph['Neuropil'].roi_response_series[name].data[:]).T
```

iii. Step 4 and Step 5 decision 1 say the stored `Deconvolved` array is suite2p output from raw `F`, not the paper’s analysis signal, so the agent recomputes the paper-style dF/F and deconvolution from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The agent recomputes the reference dF/F pipeline: restrict to within-trial samples, subtract `0.7 * Fneu`, add back per-trial neuropil mean, compute a per-trial maximin baseline, form dF/F, smooth with a 2-frame Gaussian, and also run OASIS deconvolution. However, the final script defaults to using the dF/F trace itself as `neural` (`signal='dff'`) rather than the paper-style deconvolved events.

ii. 
```python
f_ -= NEU_COEF * fneu_
...
f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
seg = gaussian_filter1d(f_[:, s:e], BASELINE_SIGMA, axis=-1)
seg = minimum_filter1d(seg, BASELINE_WIN, axis=-1)
seg = maximum_filter1d(seg, BASELINE_WIN, axis=-1)
...
dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
...
dff[:, s:e] = gaussian_filter1d(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=-1)
events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, FRAME_RATE)
...
if signal == 'dff':
    events = dff
```

iii. Step 6 says `compute_dff_and_events` is intended as a reimplementation of `reward_relative.preprocessing.dff(..., deconvolve=True)`. But Step 12 explicitly says the agent changed the final conversion to use dF/F by default because sample and full decoder accuracies were better for every output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent first keeps only `iscell` ROIs, pools them across planes, then removes putative interneurons defined by Pearson correlation of dF/F with running speed greater than 0.5. It also drops zero-variance/invalid ROIs by requiring finite correlation values.

ii. 
```python
keep = iscell[plane_idx == plane]
F_list.append(Fp[keep])
Fneu_list.append(Np[keep])
...
with np.errstate(invalid='ignore', divide='ignore'):
    r_speed = (dff_c @ sp_c) / denom
keep_cells = ~(r_speed > SPEED_CORR_THR)
keep_cells &= np.isfinite(r_speed)
```

iii. Step 5 decision 4 cites the paper’s neuron curation rules: suite2p/manual `iscell` plus putative interneuron exclusion at `r > 0.5`. Step 10 Check 5 adds that zero-variance ROIs are excluded to avoid NaN correlations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from `trial_start` to `teleport`; no further temporal shift is applied. The alignment event is the start of the trial.

ii. 
```python
starts = np.where(g('trial_start') > 0)[0]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ev = events[:, s:e]
    tt = t[s:e] - t[s]
```

iii. The notes say the NWB behavior streams are already on the imaging frame clock, so temporal alignment is inherent. Step 10 Check 3(c) specifically argues that `[start:teleport)` is the correct on-track trial window in the NWB export.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native imaging-frame resolution of about 64.48 ms per sample (`15.5078125 Hz`). No temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. Step 3 and Step 5 decision 2 say the paper’s aligned analyses operate at the imaging frame rate and that the NWB already stores behavior on that same frame clock, so no extra rebinning is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the shared behavior timestamps, specifically `beh['position'].timestamps[:]`, which the agent uses as the session time base.

ii. 
```python
t = np.asarray(beh['position'].timestamps[:])
...
tt = t[s:e] - t[s]
inp[0] = tt
```

iii. Step 2 says all behavior streams share one timestamp vector on the imaging clock, so any of those timestamps can serve as the common time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each kept trial, the agent subtracts the timestamp at trial start from the per-frame timestamps in that trial.

ii. 
```python
tt = t[s:e] - t[s]
inp[0] = tt
```

iii. Step 5’s variable mapping says `time_from_trial_start` is simply `timestamps[s:e] - timestamps[s]`, because the data are already aligned at frame resolution.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the exact same frame indices `[s:e]` as the neural slice for each trial. If ophys and behavior lengths differ by one frame, both are truncated to the common length before trial construction.

ii. 
```python
n_frames = min(F.shape[1], len(t))
...
ev = events[:, s:e]
tt = t[s:e] - t[s]
```

iii. Step 9 says ten two-plane sessions had one extra imaging frame and were truncated to the shared frame count. Step 10 Check 2 says an independent sanity script verified `time_from_trial_start` exactly matched `timestamps[s:e] - timestamps[s]`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii. 
```python
env = g('environment')
...
env_vals = np.unique(env[s:e])
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
...
inp[1] = env_trial[i]
```

iii. Step 5 maps `behavior['environment']` directly to the decoder input and notes that the session/trial values correspond to Env1/Env2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent reduces the within-trial `environment` stream to one per-trial value, taking the unique value when the trial is constant and otherwise falling back to the rounded median, then broadcasts that constant across the trial.

ii. 
```python
env_vals = np.unique(env[s:e])
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
...
inp[1] = env_trial[i]
```

iii. Step 5 describes `environment` as a per-trial decoder input. The notes say the stream is effectively constant within trials and only switches across trials/session blocks.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the session trial index `i` produced by looping over the detected trial boundaries.

ii. 
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = i
```

iii. Step 5 says `trial_number` is the trial index within session. The agent uses the reconstructed trial order from `trial_start`/`teleport` rather than any separate behavioral field.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond assigning the current trial index and broadcasting it across the trial timepoints.

ii. 
```python
inp[2] = i
```

iii. The notes treat `trial_number` as a simple per-trial contextual variable, so a constant value across each trial is sufficient.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward delivery timestamps (`Reward.timestamps`) plus the `reward_zone` occupancy stream. First, the agent computes a per-trial `rewarded` label from those raw variables; then it uses the previous trial’s label.

ii. 
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
rz_series = g('reward_zone')
...
reward_idx = np.searchsorted(t, reward_times)
...
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. Step 5 maps `Reward` plus `reward_zone` to the reference code’s `isreward` concept from `behavior.get_trial_types`, and Step 10 Check 2 says the independent sanity script recomputed `prev_trial_outcome` from raw NWB data and matched the pickle exactly.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes `rewarded[i]` for every raw trial, then sets `inp[3] = rewarded[i-1]` for each kept trial. It drops trial 0 in every session rather than encoding its previous outcome as 0.

ii. 
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        n_drop_first += 1
        continue
    ...
    inp[3] = rewarded[i - 1]
```

iii. Step 5 decision 5 says the first trial is dropped because `prev_trial_outcome` is undefined there. Step 12 Check 3 later defends this as one of the deliberate deviations from the reference.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavior `position` stream plus the trial’s reward-zone identity. Reward-zone identity is inferred from the NWB scene name (`nwb.identifier`), not from the noisy per-frame `reward_zone` occupancy stream.

ii. 
```python
identifier = nwb.identifier
scene = identifier.split('/')[-1]
...
zone_labels = scene_reward_zones(scene, n_trials_raw)
...
p = pos[s:e]
zlab = zone_labels[i]
z0, z1 = REWARD_ZONES[zlab]
d = signed_distance_to_zone(p, z0, z1)
```

iii. Step 4 resolves a discrepancy: the recorded `reward_zone` stream marks occupancy only on rewarded trials, so the agent adopts the paper/reference-code rule of deriving reward-zone identity from the scene name and validates it against rewarded trials with zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the agent computes signed distance to the nearest point of the active reward zone: negative before the zone, 0 inside it, positive after it.

ii. 
```python
def signed_distance_to_zone(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d
```

iii. Step 5 decision 8 says this is a deliberate task-driven choice: the decoder specification asks for distance to any location in the reward zone, so in-zone frames are set to 0 instead of using the paper’s reward-zone-start coordinate.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is thresholded into seven categories with cutoffs at `-50`, `-10`, `0`, `10`, and `50` cm, with category 3 reserved for exactly-in-zone (`0`) frames.

ii. 
```python
def discretize_distance(d):
    b = np.full(d.shape, 3, dtype=np.int64)
    b[(d < 0) & (d >= -10)] = 2
    b[(d < -10) & (d >= -50)] = 1
    b[d < -50] = 0
    b[(d > 0) & (d <= 10)] = 4
    b[(d > 10) & (d <= 50)] = 5
    b[d > 50] = 6
    return b
```

iii. The notes’ mapping table states these bins were chosen to match the decoder-task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned frame-for-frame with the neural data by computing it from the same per-trial `[s:e]` position slice used for the neural slice.

ii. 
```python
ev = events[:, s:e]
p = pos[s:e]
d = signed_distance_to_zone(p, z0, z1)
out[0] = discretize_distance(d)
```

iii. Step 10 Check 2 says an independent script recomputed the distance bins directly from raw trial slices and matched the saved output exactly, confirming alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii. 
```python
pos = g('position')
...
p = pos[s:e]
```

iii. The notes’ variable map lists behavior `position` as the source for absolute-position decoding.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent takes the per-trial position slice and discretizes it into five bins over the 450 cm track.

ii. 
```python
out[1] = np.digitize(p, POS_EDGES)
```

iii. Step 5 says the track is 450 cm and the task calls for five equal-sized bins, so the agent uses 90 cm boundaries.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded with edges at 90, 180, 270, and 360 cm, producing five categories.

ii. 
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. The notes explicitly say these are the 5 equal-sized bins required by the task over a 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking `position[s:e]` from the same frame indices as the neural slice.

ii. 
```python
ev = events[:, s:e]
p = pos[s:e]
out[1] = np.digitize(p, POS_EDGES)
```

iii. Step 10 Check 2 reports that an independent raw-NWB recomputation of position bins matched the saved outputs exactly.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii. 
```python
lick = g('lick')
...
lk = lick[s:e]
```

iii. The notes’ variable map lists `behavior['lick']` as the direct source for the lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick stream is binarized per frame: any positive cumulative lick count becomes 1, otherwise 0.

ii. 
```python
out[3] = (lk > 0).astype(np.int64)
```

iii. Step 3 and Step 5 say the NWB lick field is cumulative lick count per frame, so it must be thresholded to match the task’s binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing `lick[s:e]` on the same frame indices as the neural trial slice.

ii. 
```python
ev = events[:, s:e]
lk = lick[s:e]
out[3] = (lk > 0).astype(np.int64)
```

iii. Step 10 Check 2 says the independent sanity script recomputed the lick output from the same raw frames and matched the pickle.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the scene encoded in `nwb.identifier`, which specifies the active zone(s) for the session and whether there is a switch.

ii. 
```python
identifier = nwb.identifier
scene = identifier.split('/')[-1]
zone_labels = scene_reward_zones(scene, n_trials_raw)
```

iii. Step 4 says the agent deliberately chose the scene-name route because it matches the reference `get_reward_zones` logic and covers omission trials, whereas the per-frame `reward_zone` stream is absent on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses the scene name. Non-switch scenes get one zone for all trials; switch scenes change from one zone label to another after trial index 30. The resulting A/B/C label is then mapped to integers 0/1/2.

ii. 
```python
def scene_reward_zones(scene, n_trials, change_trial=CHANGE_TRIAL):
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return np.array([m.group(1)] * n_trials, dtype='<U1')
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        labels = np.array([m.group(1)] * n_trials, dtype='<U1')
        labels[change_trial:] = m.group(2)
        return labels
...
out[4] = ZONE_TO_IDX[zlab]
```

iii. Step 4’s consistency table says switch trial 30 was confirmed from the data and that scene-derived labels had zero mismatches against rewarded-trial occupancy, so this processing was treated as both reference-faithful and empirically validated.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward delivery timestamps (`Reward.timestamps`) together with the `reward_zone` occupancy stream for the same trial.

ii. 
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
rz_series = g('reward_zone')
...
reward_idx = np.searchsorted(t, reward_times)
...
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. Step 5 maps this directly to the paper’s `behavior.get_trial_types` notion of `isreward`: reward delivered during the trial and inside the reward zone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The agent maps reward timestamps onto frame indices with `searchsorted`, computes a per-trial binary `rewarded` flag using both reward delivery and zone occupancy, and then broadcasts that flag across all frames in the trial.

ii. 
```python
reward_idx = np.searchsorted(t, reward_times)
...
rewarded[i] = int(got_reward and in_zone)
...
out[5] = rewarded[i]
```

iii. Step 10 Check 2 says an independent script recomputed `reward_outcome` from raw NWB reward timestamps plus `reward_zone` and matched the converted data exactly.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases: it truncates behavior and ophys to a common frame count when they differ, drops trials with lick-sensor errors, very short duration, non-finite values, or `scanning != 1`, and drops zero-variance ROIs via the finite-correlation check. It also sanity-checks that scene-derived zone labels agree with rewarded-trial occupancy.

ii. 
```python
n_frames = min(F.shape[1], len(t))
...
if F.shape[1] != n_frames:
    F = F[:, :n_frames]
    Fneu = Fneu[:, :n_frames]
if len(t) != n_frames:
    t = t[:n_frames]; pos = pos[:n_frames]; speed = speed[:n_frames]
    lick = lick[:n_frames]; env = env[:n_frames]
    rz_series = rz_series[:n_frames]; scanning = scanning[:n_frames]
...
lick_error[i] = (lick[s:e] > 2).mean() > LICK_ERROR_FRAC
...
keep_cells &= np.isfinite(r_speed)
```

iii. Step 9 documents the discovered one-frame mismatch in m17/m18 and says truncation was verified not to remove trial data. Step 10’s edge-case list explains the other trial and ROI drops as defensive handling for invalid data.

## 13-a. What are the most time-consuming steps of the code?

i. The agent identifies two dominant costs: loading the ophys arrays from NWB and computing dF/F plus OASIS. Session-level processing is then parallelized across workers.

ii. 
```python
timing['load_ophys'] = time.time() - t_load0
...
timing['dff_oasis'] = time.time() - t0
...
with ProcessPoolExecutor(max_workers=args.nworkers, mp_context=ctx) as pool:
```

iii. Step 6 and the Step 7/9 runtime tables say `load_ophys` is about 0.4-1.0 s per session and dF/F+OASIS about 1.5-2.5 s per session, making them the clear runtime bottlenecks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent says the per-trial baseline/deconvolution loops are the main obvious loops, but treats them as largely unavoidable because the reference method computes baselines per trial. The trial-construction loop also remains in Python because trials have variable length.

ii. 
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
...
for s, e in zip(starts, stops):
    f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
    seg = gaussian_filter1d(f_[:, s:e], BASELINE_SIGMA, axis=-1)
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural.append(np.ascontiguousarray(ev, dtype=np.float32))
```

iii. Step 6 says the per-trial loops are part of matching the reference maximin-baseline method and are already vectorized across cells, so only limited further vectorization is practical without changing the processing definition.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats some work by computing both dF/F and OASIS events even when the final output uses dF/F, and by reusing trial loops both for per-trial labels and for trial assembly. It does not perform the separate full-dataset survey pass used by the human reference solution.

ii. 
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
...
if signal == 'dff':
    events = dff
...
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_idx >= s) & (reward_idx < e))
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural.append(np.ascontiguousarray(ev, dtype=np.float32))
```

iii. Step 6 calls out that dF/F and OASIS are both computed because the script supports comparing `events` versus `dff`. Step 12 says the default was later switched to dF/F, leaving the event computation as overhead for the default path.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded work is computing OASIS events even when the default `signal='dff'` causes the final output to use dF/F instead. The script also computes plotting/sanity-check helpers (`zone_mismatch`, `dff_keep`, `plane_of_cell`, timing fields) that do not enter the saved decoder dataset.

ii. 
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
...
if signal == 'dff':
    events = dff
...
zone_mismatch = 0
...
dff_keep = dff[keep_cells]
plane_of_cell = np.concatenate(keep_list)[keep_cells]
...
timing=timing, n_frames_trunc=n_frames_trunc,
```

iii. Step 12 explicitly says the final conversion uses dF/F by default, so the OASIS path is unnecessary work on the default run. The notes also describe several QC-only values that are used for review and logging rather than downstream decoding.
