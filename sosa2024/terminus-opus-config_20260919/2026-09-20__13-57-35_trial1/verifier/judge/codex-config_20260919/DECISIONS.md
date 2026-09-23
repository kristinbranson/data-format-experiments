# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively gathers every matching NWB session under `/app/data/sub-*`, sorts the paths, and processes all 152 files by default. Each session is read directly with `h5py`; fluorescence, neuropil, ROI metadata, behavioral streams, timestamps, and reward timestamps are loaded. Full conversion uses a spawned `ProcessPoolExecutor` across sessions.

ii.
```python
def session_files():
    import glob
    return sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')))

with h5py.File(path, 'r') as f:
    ...
    F = np.concatenate(Fl, axis=0)
    Fneu = np.concatenate(Fnl, axis=0)
```

iii. The agent justified this by scanning the release and finding 11 subject directories, 152 NWB files, and 12,216 trial starts. It chose `h5py` over `pynwb` for faster partial reads and documented that all released switch-task sessions were retained.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file's `general/subject/subject_id`. After processing, unique IDs are numerically sorted and each session receives an index into that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({i['subject'] for i in infos}, key=lambda x: int(x[1:]))
subject_idx = np.array([subjects.index(i['subject']) for i in infos], dtype=np.int64)
```

iii. The notes report that the 11 IDs in the files are exactly the 11 switch-task mice described in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The sorted file list determines session order; each worker returns one session-level list of trials.

ii.
```python
for i, r in enumerate(ex.map(_worker,
        [(f, False, '/app', args.signal) for f in files])):
    results.append(r)
neural = [r[0] for r in results]
```

iii. The agent used the one-file-per-session organization and checked that the release contains 14 days per mouse except m11, whose imaging begins on day 3.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples in `trial_start`; trial ends are positive samples in `teleport`. Counts are truncated to the smaller count, and emitted trial windows are `[start-1, stop-1)`, matching the reference code's offset convention.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
ntr = min(len(starts), len(stops))
starts, stops = starts[:ntr], stops[:ntr]
...
sl = slice(s - 1, e - 1)
```

iii. The agent inspected boundary positions and the upstream paper code, concluding that the teleport sample is an interpolated reset sample and must be excluded.

## 1-e. How are trials filtered based on quality controls?

i. A complete trial is dropped if more than 30% of its cumulative-lick samples exceed 2, if it has fewer than two emitted samples, if neural values contain NaNs, or if position indicates pre-synchronization (`< -100`). This removed 81 lick-error trials and retained 12,135 trials.

ii.
```python
lick_error[i] = (np.sum(lick_tr > 2) / max(1, len(lick_tr))) > LICK_ERR_FRAC
if lick_error[i]: continue
if ev.shape[1] < 2: continue
if np.any(np.isnan(ev)): continue
if np.any(pos < -100): continue
```

iii. The agent tied the lick rule to the paper's reported 81 corrupted trials and dropped them because categorical decoder outputs cannot contain NaNs. The other checks were defensive safeguards.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from suite2p `Fluorescence` (F) and `Neuropil` (Fneu), restricted by the ROI `iscell` flag and pooled across planes. The stored NWB `Deconvolved` series is not used.

ii.
```python
iscell = seg['iscell'][:, 0] > 0
plane_idx = seg['planeIdx'][:]
Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
Fnl.append(f[f'processing/ophys/Neuropil/{p}/data'][:, :].T[sel])
```

iii. The agent correctly observed that the paper recomputed dF/F and events from raw suite2p F/Fneu, rather than analyzing the NWB's suite2p-default deconvolution.

## 2-b. How is the `neural` data processed?

i. The code subtracts `0.7*Fneu`, adds back the per-trial neuropil mean, estimates a per-trial maximin baseline (Gaussian sigma 15, 300-sample minimum then maximum filters), computes dF/F, and smooths it with sigma 2. It also computes OASIS events (`tau=0.7`), but the command-line default and delivered pickle select smoothed dF/F, not events. Unlike the human implementation, baselines never span teleports on designated imaging days.

ii.
```python
f_ -= NEU_COEF * fneu_
flow[:, sl] = nansmooth(f_[:, sl], BASELINE_SIGMA)
flow[:, sl] = ndimage.minimum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
flow[:, sl] = ndimage.maximum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
events[:, sl] = dcnv.oasis(..., TAU, frame_rate)
...
ap.add_argument('--signal', choices=['events', 'dff'], default='dff')
```

iii. Initially the agent intended to use paper-style events. It later compared decoder accuracy for dF/F and events, found dF/F decoded every target better, and changed the delivered default to dF/F. It argued this was decoder-motivated and that dF/F remains a paper-processed signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only manually curated `iscell` ROIs are loaded. Cells whose dF/F has Pearson correlation greater than 0.5 with speed are classified as putative interneurons and removed.

ii.
```python
iscell = seg['iscell'][:, 0] > 0
speed_corr = (Dz @ spz) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
events = events[~is_int]
dff_kept = dff[~is_int]
```

iii. Both filters are attributed to the paper: suite2p manual curation and the Methods' `r > 0.5` interneuron criterion. The observed exclusion rate was checked against the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start simply by slicing the already sample-aligned neural stream from `start-1`; no resampling or fixed-duration window is used.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
tt = S['time'][sl] - S['time'][s - 1]
```

iii. The notes state that VR behavior had already been interpolated onto imaging frames before NWB export, making neural and behavior sample-aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging/behavior frames are retained with no rebinning. The metadata uses the median session frame rate, approximately 15.5078 Hz or 64.484 ms per bin.

ii.
```python
frame_rate = 1.0 / np.median(np.diff(time))
bin_ms = float(1000.0 / np.median(frame_rates))
```

iii. The agent verified that the per-plane and behavioral sampling rates are constant across sessions and match the paper's approximately 15.5 Hz rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position/timestamps` behavioral timestamp vector.

ii.
```python
time = b['position/timestamps'][:]
tt = S['time'][sl] - S['time'][s - 1]
```

iii. The agent found all behavior streams share the imaging-frame timestamps, so position timestamps were a valid common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the first emitted trial sample is subtracted from every timestamp in that trial.

ii.
```python
tt = S['time'][sl] - S['time'][s - 1]
```

iii. This makes every variable-length trial begin at zero while preserving native sampling intervals.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same slice is applied to timestamps and neural data. All streams are first truncated to a common session length.

ii.
```python
n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
...
ev = events[:, sl]
tt = S['time'][sl] - S['time'][s - 1]
```

iii. The agent documented a known one-frame mismatch in ten two-plane sessions and used common-minimum truncation to preserve one-to-one alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` time series within each trial.

ii.
```python
beh = dict(..., env=g('environment'), ...)
u = np.unique(beh['env'][sl])
morph[i] = int(u[0])
```

iii. Inspection showed trial values are constant and encode ENV1/ENV2 as 0/1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code takes the first unique in-trial environment value and broadcasts it over the trial.

ii.
```python
morph[i] = int(np.unique(beh['env'][sl])[0])
np.full(T, float(morph[i]))
```

iii. Broadcasting yields the required two-dimensional input representation; constancy was checked empirically.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index of the trial boundaries, not the stored `trial number` stream (although that stream is loaded).

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    np.full(T, float(i))
```

iii. The agent treated the boundary order as the unambiguous sequential trial number within a session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond broadcasting the zero-based index to every time bin.

ii.
```python
np.full(T, float(i))
```

iii. This matches the requested continuous per-trial covariate and homogeneous input shape.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Sparse `Reward/timestamps` are mapped to the behavioral clock. Current trial reward is defined as the presence of a mapped reward event and reward-zone entry; previous outcome uses that prior trial value.

ii.
```python
reward_t = b['Reward/timestamps'][:]
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. The agent explicitly followed the paper's `get_trial_types` definition, including the reward-zone-entry condition.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The prior trial's `isreward` value is broadcast across the current trial. The first trial is assigned 1, unlike the human solution's 0.

ii.
```python
np.full(T, float(isreward[i - 1]) if i > 0 else 1.0)
```

iii. The agent argued that mice had approximately 30 rewarded warm-up laps immediately before imaging, so 1 is a meaningful prior for the first recorded trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from behavioral position and nominal zone bounds. The zone label comes from the NWB scene identifier, with switch scenes changing zone at zero-based trial 30.

ii.
```python
zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
z0, z1 = REWARD_ZONES[zone]
dist = signed_distance_to_zone(pos, z0, z1)
```

iii. The agent followed `behavior.get_reward_zones` and verified nominal starts against observed reward-zone-entry positions across all sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the zone, zero inside its inclusive bounds, and positive after it, measured to the nearest zone edge.

ii.
```python
d = np.zeros_like(pos)
d[pos < zone_start] = pos[pos < zone_start] - zone_start
d[pos > zone_end] = pos[pos > zone_end] - zone_end
```

iii. This directly implements the task's signed-distance definition and the paper's reward-relative coordinate concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks create seven bins: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[d == 0] = 3
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The masks were chosen to reproduce the decoder specification exactly, especially the distinct zero class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[start-1, stop-1)` slice, so the derived distance has exactly one value per neural frame.

ii.
```python
ev = events[:, sl]
pos = beh['pos'][sl]
dist = signed_distance_to_zone(pos, z0, z1)
```

iii. Independent raw-data spot checks reportedly found exact agreement in trial length and distance bins.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
beh = dict(pos=g('position'), ...)
pos = beh['pos'][sl]
```

iii. This stream is already VR position aligned to imaging frames.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm using floor division and clipped to class indices 0 through 4.

ii.
```python
return np.clip((pos // 90).astype(np.int64), 0, 4)
```

iii. Five equal bins span the 450 cm track; clipping absorbs small measurement excursions beyond the nominal endpoints.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes are `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`, subject to clipping into 0–4.

ii.
```python
discretize_position(pos)  # np.clip((pos // 90).astype(np.int64), 0, 4)
```

iii. The agent used the instructed five equal 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The identical trial slice is applied to position and neural arrays.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
pos = beh['pos'][sl]
```

iii. The NWB streams are already sample-aligned; spot checks against raw NWB values passed exactly.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavioral `lick` stream.

ii.
```python
beh = dict(..., lick=g('lick'), ...)
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. The agent identified this as the cumulative per-frame lick signal used by the paper.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive value becomes 1 and zero becomes 0. Trials failing the sensor-error rule are removed entirely.

ii.
```python
lick = (beh['lick'][sl] > 0).astype(np.int64)
if lick_error[i]:
    continue
```

iii. Binarization matches the requested output. Whole-trial removal was justified as the only clean categorical treatment of corrupted lick samples.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same trial slice and native frames.

ii.
```python
ev = events[:, sl]
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. No interpolation is needed because behavioral data were aligned to imaging frames upstream.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The location is inferred from the scene name stored in the NWB identifier and the trial index for switch sessions, rather than from the raw `reward_zone` samples.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
```

iii. The agent chose the paper's deterministic `get_reward_zones` mapping and validated it against raw zone-entry positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene strings are parsed by regex. Fixed scenes retain one A/B/C label; switch scenes change from the first to second label at trial 30. A/B/C are mapped to 0/1/2 and broadcast.

ii.
```python
n0 = min(change_trial, ntrials)
return np.array([z0] * n0 + [z1] * (ntrials - n0))
...
np.full(T, ZONE_TO_IDX[zone], dtype=np.int64)
```

iii. This is justified as a direct reproduction of the reference repository and the experimental design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse `Reward/timestamps`, the common behavior clock, and the in-trial `reward_zone` stream.

ii.
```python
reward_t = b['Reward/timestamps'][:]
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. The combined condition was taken from the paper's `get_trial_types`; omission frequency was checked against the reported approximately 15%.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are placed at insertion indices on the frame clock. A trial is 1 only if it contains both a reward event and reward-zone entry; otherwise it is 0, then broadcast over time.

ii.
```python
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
np.full(T, isreward[i], dtype=np.int64)
```

iii. The agent cited the paper code and verified the resulting omission rate. Unlike the human code, it does not assert sub-bin reward timestamp error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All neural and behavioral streams are truncated to their common minimum length to handle known one-frame mismatches. Start/end lists are similarly truncated. Trials are dropped for NaN neural data, fewer than two bins, pre-sync position, or corrupted licks. Reward indices are clipped to valid bounds, and undefined speed correlations are converted to zero.

ii.
```python
n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
starts, stops = starts[:ntr], stops[:ntr]
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
```

iii. The one-frame correction was discovered during the full run and documented as known upstream behavior. The remaining checks were intended to keep all emitted decoder arrays finite and aligned.

## 13-a. What are the most time-consuming steps of the code?

i. The agent identified F/Fneu loading, per-session maximin dF/F plus OASIS, and serializing the approximately 9.6 GB pickle as dominant. It timed loading and neural processing per session.

ii.
```python
t_load = time.time() - t0
...
dff, events = compute_events(...)
t_dff = time.time() - t0
...
pickle.dump(data, fh, protocol=4)
```

iii. Benchmarks in the notes estimated roughly 13 minutes serial and about 3 minutes with eight workers, plus large-file pickling.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized the per-cell speed correlation into a matrix product. Trial assembly, per-trial baseline/OASIS work, per-plane loading, and plotting rasters remain loops; some session-wide behavioral transformations could be computed before splitting, but variable trial lengths limit full vectorization.

ii.
```python
speed_corr = (Dz @ spz) / denom
for start, stop in zip(starts, stops):
    ...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes say correlation vectorization is small relative to neural preprocessing and use process-level parallelism as the main speedup.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, trial boundaries are traversed during dF/F masking, baseline estimation, smoothing/deconvolution, trial-label calculation, and final assembly. The delivered dF/F mode nevertheless computes OASIS events first and then replaces them. Optional plotting also re-derives continuous distance and constructs padded output rasters.

ii.
```python
for start, stop in zip(starts, stops): ...  # repeated in compute_events
...
dff, events = compute_events(...)
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```

iii. The agent's documentation emphasizes parallelism and partial reads, but does not explicitly acknowledge that default dF/F conversion still performs the discarded OASIS computation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the delivered default `--signal dff` path, OASIS events are computed for every cell and trial and then discarded. The raw `scanning`, `autoreward`, and stored `trialnum` streams are loaded but not used in emitted data. Diagnostic plots, when requested, recompute quantities solely for visualization.

ii.
```python
beh = dict(..., autoreward=g('autoreward'), scanning=g('scanning'),
           trialnum=g('trial number'))
...
dff, events = compute_events(...)
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```

iii. The agent retained both neural representations to support its control experiment and an `--events` option, but for the actual delivered default this makes deconvolution unnecessary overhead.
