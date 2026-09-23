# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all NWB files by globbing `/app/data/sub-*/sub-*.nwb`, then opening each file directly with `h5py`. It reads the needed ophys arrays (`Fluorescence`, `Neuropil`, segmentation metadata) and behavior arrays from each NWB file, rather than using `pynwb`.

ii.
```python
def session_files():
    import glob
    return sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')))

def load_session(path):
    with h5py.File(path, 'r') as f:
        ...
        fl = f['processing/ophys/Fluorescence']
        ...
        b = f['processing/behavior/BehavioralTimeSeries']
```

iii. In `CONVERSION_NOTES.md`, the agent says the NWB release already contains the aligned streams needed from the paper pipeline and that `h5py` avoids `pynwb` overhead while reading only the needed datasets.

## 1-b. How are the data split into subjects?

i. Subjects are split using the NWB field `general/subject/subject_id`, and the final `subjects` list is the sorted unique set of those IDs across sessions.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({i['subject'] for i in infos}, key=lambda x: int(x[1:]))
subject_idx = np.array([subjects.index(i['subject']) for i in infos], dtype=np.int64)
```

iii. The notes say the DANDI release has 11 mouse directories and that all 152 files belong to those 11 switch-task mice, so the agent keeps them all.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session ordering comes from the sorted list of file paths. The session day is read from `general/session_id`.

ii.
```python
def session_files():
    import glob
    return sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')))

day = f['general/session_id'][()].decode()
```

iii. The notes state that the DANDI dataset is one file per subject-session and that this matches the paper’s session structure.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` and `teleport`. The agent finds all indices where `trial_start > 0` and all indices where `teleport > 0`, pairs them in order, and later emits per-trial slices using `[start-1 : stop-1)`.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
...
sl = slice(s - 1, e - 1)          # reference trial window
ev = events[:, sl]
```

iii. The notes justify this as matching the paper’s “reference window” and explicitly describe trials as laps ending before the teleport sample.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials with lick-sensor errors, trials shorter than 2 bins, trials whose neural window still contains NaNs, and trials containing pre-sync samples (`position < -100`). It does not use the human reference’s `<50`-timepoint rule.

ii.
```python
lick_error[i] = (np.sum(lick_tr > 2) / max(1, len(lick_tr))) > LICK_ERR_FRAC
...
if lick_error[i]:
    drop['lick_error'] += 1
    continue
if ev.shape[1] < 2:
    drop['too_short'] += 1
    continue
if np.any(np.isnan(ev)):
    drop['nan_events'] += 1
    continue
if np.any(pos < -100):
    drop['pre_sync'] += 1
    continue
```

iii. The notes justify dropping lick-error trials because lick is a decoder output and NaNs are not allowed; they also describe the pre-sync and NaN checks as defensive guards.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data is derived from raw suite2p `Fluorescence` and `Neuropil` traces, restricted to ROIs with `iscell == 1`.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:, 0] > 0
...
Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
Fnl.append(f[f'processing/ophys/Neuropil/{p}/data'][:, :].T[sel])
```

iii. The notes explicitly reject the NWB `Deconvolved` series as the paper’s analyzed signal and say the paper recomputes its own dF/F and events from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The agent reimplements the paper’s dF/F pipeline: neuropil subtraction (`0.7`), add back per-trial neuropil mean, maximin baseline, dF/F normalization, 2-frame smoothing, and OASIS deconvolution. However, the delivered dataset defaults to storing `dff` rather than deconvolved `events`.

ii.
```python
f_ -= NEU_COEF * fneu_
...
f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
flow[:, sl] = nansmooth(f_[:, sl], BASELINE_SIGMA)
flow[:, sl] = ndimage.minimum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
flow[:, sl] = ndimage.maximum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
...
dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl], dtype=np.float32),
                           2000, TAU, frame_rate)
...
if signal == 'dff':
    events = dff_kept.astype(np.float32)
...
ap.add_argument('--signal', choices=['events', 'dff'], default='dff',
```

iii. The notes first describe the paper-faithful events pipeline, then later justify switching the final dataset to dF/F because controlled decoder experiments gave better accuracy than events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent first keeps only manually curated suite2p cells (`iscell == 1`), then removes putative interneurons whose dF/F is too correlated with speed (`r > 0.5`).

ii.
```python
iscell = seg['iscell'][:, 0] > 0
...
speed_corr = (Dz @ spz) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
dff_kept = dff[keep_cells]
```

iii. The notes say this matches the paper’s neuron curation: suite2p manual curation plus exclusion of putative interneurons by dF/F-speed correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural trials using the slice `[trial_start-1 : teleport-1)`, so the first emitted bin is one frame before the `trial_start` index rather than at `trial_start` itself.

ii.
```python
sl = slice(s - 1, e - 1)          # reference trial window
ev = events[:, sl]
neural.append(ev.astype(np.float32))
```

iii. The notes justify this as the paper’s “reference trial window,” and explicitly describe the trial as ending before the teleport sample.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data stays at the native frame rate, with no temporal rebinning. The bin size is computed from the median sampling interval and ends up at about 64.48 ms.

ii.
```python
frame_rate = 1.0 / np.median(np.diff(time))
...
frame_rates = np.array([i['frame_rate'] for i in infos])
bin_ms = float(1000.0 / np.median(frame_rates))
```

iii. The notes say the behavior and imaging streams are already frame-aligned in the NWB files and that native imaging-frame bins should be kept.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps, specifically `position/timestamps`, which are used as the common session time base.

ii.
```python
time = b['position/timestamps'][:]
...
tt = S['time'][sl] - S['time'][s - 1]
```

iii. The notes say the behavior streams are already aligned to imaging frames and use a shared timestamp grid, so a single behavior timestamp series suffices.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent subtracts the first timestamp of the emitted trial window from every timestamp in that window.

ii.
```python
tt = S['time'][sl] - S['time'][s - 1]
...
inp = np.stack([tt, ...]).astype(np.float32)
```

iii. The notes describe this as straightforward construction of time-from-trial-start once the trial window is chosen.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same `[start-1 : stop-1)` slice as the neural data, so time and neural arrays are synchronized to each other inside the converted dataset.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
tt = S['time'][sl] - S['time'][s - 1]
```

iii. The notes say no interpolation is needed because the NWB behavior and neural samples are already 1:1 aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii.
```python
beh = dict(..., env=g('environment'), ...)
...
u = np.unique(beh['env'][sl])
morph[i] = int(u[0])
```

iii. The notes identify `environment` as the NWB version of the paper’s `morph`/environment variable and state that it is constant within trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent takes the unique environment value within each trial, stores it as `morph[i]`, and tiles that scalar across all time bins in the trial.

ii.
```python
u = np.unique(beh['env'][sl])
morph[i] = int(u[0])
...
np.full(T, float(morph[i]))
```

iii. The notes justify this by saying environment is constant within trial and corresponds to ENV1 vs ENV2.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the within-session trial loop index `i`, not from the stored `trial number` behavior variable.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    np.full(T, float(i))
```

iii. The notes describe it as the 0-indexed lap number within a session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is repeated across all time bins of the trial.

ii.
```python
inp = np.stack([tt,
                np.full(T, float(morph[i])),
                np.full(T, float(i)),
                ...]).astype(np.float32)
```

iii. The notes justify this as the decoder-required per-trial covariate representation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery timestamps and the `reward_zone` time series, via a per-trial `isreward` label that requires both a reward event and reward-zone occupancy.

ii.
```python
reward_t = b['Reward/timestamps'][:]
...
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
beh['reward'] = reward
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. The notes say this reproduces `behavior.get_trial_types`, where reward outcome is defined as reward delivered and reward zone entered.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i>0`, the agent uses `isreward[i-1]`. For the first trial, it sets the previous outcome to `1.0` rather than `0`.

ii.
```python
np.full(T, float(isreward[i - 1]) if i > 0 else 1.0)
```

iii. The notes justify the first-trial value by assuming each imaging session followed rewarded warm-up trials on the same zone.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial position samples plus a per-trial reward-zone identity inferred from the scene name, not from the `reward_zone` behavior time series.

ii.
```python
zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
...
pos = beh['pos'][sl]
zone = zone_labels[i]
z0, z1 = REWARD_ZONES[zone]
dist = signed_distance_to_zone(pos, z0, z1)
```

iii. The notes say this matches the reference `behavior.get_reward_zones` logic from the paper code, including the switch at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest edge of the reward zone: negative before the zone, zero inside it, positive after it.

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

iii. The notes justify this as the task-required linearized reward-relative position variable.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into seven categories using explicit inequality rules matching the requested bins.

ii.
```python
def discretize_distance(d):
    out = np.full(d.shape, 3, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
```

iii. The notes say these bins follow the decoder task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance from the same per-trial slice `sl` that is used for neural data.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
pos = beh['pos'][sl]
dist = signed_distance_to_zone(pos, z0, z1)
```

iii. The notes rely on the NWB’s shared frame grid and the chosen reference trial window.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii.
```python
beh = dict(pos=g('position'), ...)
...
pos = beh['pos'][sl]
```

iii. The notes identify `position` as the animal’s VR position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent takes the per-trial position samples and bins them by 90 cm chunks using floor division and clipping.

ii.
```python
def discretize_position(pos):
    return np.clip((pos // 90).astype(np.int64), 0, 4)
```

iii. The notes justify this as five equal bins over the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into five bins: `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`, with clipping to the end bins.

ii.
```python
def discretize_position(pos):
    return np.clip((pos // 90).astype(np.int64), 0, 4)
```

iii. The notes say this matches the decoder task’s five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking position from the same trial slice as neural data.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[:, sl]
pos = beh['pos'][sl]
```

iii. The notes again rely on 1:1 behavior-neural alignment in the NWB files.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii.
```python
beh = dict(..., lick=g('lick'), ...)
```

iii. The notes describe this as the cumulative lick-count-per-frame stream stored in the NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick trace is binarized as `lick > 0`.

ii.
```python
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. The notes say the decoder task requires a binary lick output and treat any positive cumulative lick count as a lick.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the lick array with the same per-trial slice as neural data.

ii.
```python
sl = slice(s - 1, e - 1)
lick = (beh['lick'][sl] > 0).astype(np.int64)
```

iii. The notes justify this through the shared frame grid and the reference trial window.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the scene string in the NWB `identifier`, plus the trial index relative to the fixed switch point at trial 30.

ii.
```python
ident = f['identifier'][()].decode()
scene = ident.split('/')[-1]
...
zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
```

iii. The notes say this reproduces the paper’s `behavior.get_reward_zones` logic directly from scene naming conventions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses scene names with regexes, applies a switch at `CHANGE_TRIAL = 30`, converts A/B/C to integer class IDs, and tiles the per-trial value across time.

ii.
```python
CHANGE_TRIAL = 30
...
def scene_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    ...
    n0 = min(change_trial, ntrials)
    return np.array([z0] * n0 + [z1] * (ntrials - n0))
...
np.full(T, ZONE_TO_IDX[zone], dtype=np.int64)
```

iii. The notes say the switch-at-30 rule was verified across the dataset and matches the paper.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward timestamps and the `reward_zone` series, through the same per-trial `isreward` variable used for previous-trial outcome.

ii.
```python
reward_t = b['Reward/timestamps'][:]
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
```

iii. The notes say this follows `behavior.get_trial_types`, where rewarded means reward delivered and reward zone entered.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped onto the frame grid with `searchsorted`; then each trial gets a binary label `isreward[i]`, which is tiled across time in the output array.

ii.
```python
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
reward[idx] = 1
...
isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
...
np.full(T, isreward[i], dtype=np.int64)
```

iii. The notes justify this as the paper’s trial-type definition rather than a simple “any reward timestamp in trial” rule.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases: it truncates one-frame neural/behavior mismatches to a common length; clips reward timestamp insertion indices; truncates unmatched `starts`/`stops` to the shared count; drops trials with lick-sensor errors, NaN neural data, too few bins, or pre-sync position samples. It does not implement the human reference’s Viterbi fallback for missing reward-zone evidence, because it derives zone from scene names instead.

ii.
```python
n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
...
F = F[:, :n]; Fneu = Fneu[:, :n]; time = time[:n]
beh = {k: v[:n] for k, v in beh.items()}
starts = starts[starts < n]
stops = stops[stops <= n]
ntr = min(len(starts), len(stops))
starts, stops = starts[:ntr], stops[:ntr]
...
idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
...
if lick_error[i]: ...
if ev.shape[1] < 2: ...
if np.any(np.isnan(ev)): ...
if np.any(pos < -100): ...
```

iii. The notes explicitly discuss the one-frame correction for some two-plane sessions and the choice to drop lick-error trials because the decoder format cannot carry NaNs in the lick output.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large NWB arrays, computing dF/F plus OASIS deconvolution in `compute_events`, and serializing the large pickle. Parallel per-session processing is used to reduce wall-clock time.

ii.
```python
S = load_session(path)
...
dff, events = compute_events(S['F'], S['Fneu'], starts, stops, S['frame_rate'])
...
with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
    for i, r in enumerate(ex.map(_worker, [(f, False, '/app', args.signal) for f in files])):
        ...
with open(args.outfile, 'wb') as fh:
    pickle.dump(data, fh, protocol=4)
```

iii. The notes explicitly give timing breakdowns for loading, dF/F/OASIS, and total runtime, and describe these as the main bottlenecks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are the repeated per-trial loops in `compute_events` and the per-trial assembly loop in `process_session`. Plotting also loops over trials and outputs. The interneuron correlation step was already vectorized.

ii.
```python
for start, stop in zip(starts, stops):
    f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
    ...
for start, stop in zip(starts, stops):
    sl = slice(start - 1, stop - 1)
    ...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes emphasize vectorization where convenient, but the code still keeps trial-wise loops because windows are variable-length and trial-local.

## 13-c. What processing does the code repeat multiple times?

i. The code computes both `dff` and `events` for every session even when the final dataset uses `dff`. It also traverses trial windows multiple times inside `compute_events` for masking, baseline estimation, smoothing, and deconvolution.

ii.
```python
dff, events = compute_events(S['F'], S['Fneu'], starts, stops, S['frame_rate'])
...
if signal == 'dff':
    events = dff_kept.astype(np.float32)
```

iii. The notes explain that both signals were kept available because the agent compared events versus dF/F in later control analyses.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary work is that OASIS deconvolution is always run even though the default `--signal` is `dff`, so the computed `events` are discarded in the delivered dataset. The plotting path also keeps both dF/F and events for visualization only.

ii.
```python
events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl], dtype=np.float32),
                           2000, TAU, frame_rate)
...
if signal == 'dff':
    events = dff_kept.astype(np.float32)
...
ap.add_argument('--signal', choices=['events', 'dff'], default='dff',
```

iii. In the notes, the agent says it switched the delivered dataset to dF/F after decoder comparisons, but left `--signal events` available for control analyses.
