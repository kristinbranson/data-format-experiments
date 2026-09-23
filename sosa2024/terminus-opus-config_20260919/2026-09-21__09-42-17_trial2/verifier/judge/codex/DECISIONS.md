# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file matching `/app/data/sub-*/*.nwb`, treats each file as one session, and converts sessions independently before assembling them into the final dataset. It uses `h5py` rather than `pynwb`.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
with h5py.File(fname, 'r') as f:
    scene = f['identifier'][()].decode().split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    ...
    beh = load_behavior(f)
    F, Fneu, plane_idx = load_fluorescence(f)
```

iii. In `CONVERSION_NOTES.md`, the AI says it verified that the DANDI release contains 11 subject folders and 152 NWB files, so globbing `sub-*/*.nwb` captures the full released dataset. The trajectory shows it switched from exploratory `pynwb` inspection to `h5py` for the final converter because the relevant NWB arrays were straightforward to read directly.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB metadata field `general/subject/subject_id`, then uniqued and sorted after session conversion.

ii. 
```python
subject = f['general/subject/subject_id'][()].decode()
...
results.sort(key=lambda r: (r['subject'], r['session_id']))
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
...
'subjects': subjects,
'subject_idx': np.array([subjects.index(r['subject']) for r in results],
                        dtype=np.int64),
```

iii. The notes say the release contains the expected 11 mice (`m3`, `m4`, `m7`, `m11`-`m15`, `m17`-`m19`). The AI justified reading the subject ID from the file itself rather than inferring it only from the directory name.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. After conversion, sessions are ordered by `(subject, session_id)`.

ii. 
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
session_id = f['general/session_id'][()].decode()
...
results.sort(key=lambda r: (r['subject'], r['session_id']))
```

iii. In the notes, the AI says the DANDI structure is one file per imaging day/session and that `general/session_id` corresponds to the experiment day discussed in the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by the interval `[trial_start, teleport)`: the first frame with `trial_start > 0` marks trial start, and the first frame with `teleport > 0` marks the excluded teleport frame at trial end.

ii. 
```python
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    T = e - s
    pos = beh['position'][s:e].astype(np.float64)
```

iii. The notes explicitly justify excluding the teleport frame because its position is already interpolated between the end of the track and the ITI. The trajectory shows the AI checked example traces and concluded `[trial_start, teleport)` best matched both the paper and the reference `dff` windows.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops three kinds of unusable trial/session data: trials flagged as lick-sensor-error trials, trials whose teleport falls beyond the truncated common behavior/ophys frame count, and whole sessions with fewer than two usable trials after filtering.

ii. 
```python
valid = stops < nframes
if not np.all(valid):
    ...
    starts, stops = starts[valid], stops[valid]
...
lick_bad = np.array([(beh['lick'][s:e] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for s, e in zip(starts, stops)])
...
for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_bad[i]:
        continue
...
if len(neural) < 2:
    print(f'  !! {os.path.basename(fname)}: only {len(neural)} usable trials, skipping')
    return None
```

iii. The notes justify the lick filter from the paper's reported rule: trials with `>30%` of frames having cumulative lick count `>2`. The AI says this reproduces the paper's 81 problematic trials exactly. The truncation rule was justified in the notes and trajectory as a needed fix for occasional one-frame behavior/ophys mismatches.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural data come from raw fluorescence and neuropil traces, specifically `processing/ophys/Fluorescence/plane*/data` and `processing/ophys/Neuropil/plane*/data`, after subsetting to manually curated `iscell` ROIs.

ii. 
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:, 0].astype(bool)
plane_idx = seg['planeIdx'][:]
...
Fs.append(f[f'processing/ophys/Fluorescence/plane{p}/data'][:].T)
Fns.append(f[f'processing/ophys/Neuropil/plane{p}/data'][:].T)
...
return F[iscell], Fneu[iscell], plane_idx[iscell]
```

iii. The notes state that the NWB `Deconvolved` array is suite2p's deconvolution of raw fluorescence and is not the paper's analyzed signal, so the AI chose to recompute the paper's processed signal from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's dF/F pipeline: copy trial samples into NaN-masked arrays, subtract `0.7 * Fneu`, add back each trial's mean neuropil, compute a per-trial maximin baseline with Gaussian smoothing and min/max filters, form dF/F, smooth it with a 2-sample Gaussian, compute OASIS deconvolution, and then save either deconvolved `events` or dF/F. In the shipped converter, `dff` is the default saved neural signal.

ii. 
```python
def compute_dff(F, Fneu, starts, stops):
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    f_ -= NEU_COEF * fneu_
    ...
    for s, e in zip(starts, stops):
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=-1)
        seg = ndimage.minimum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
        seg = ndimage.maximum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
        flow[:, s:e] = seg
    ...
    dff[:, mask] = (f_[:, mask] - flow[:, mask]) / np.abs(flow[:, mask])
    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
    return dff

def deconvolve(dff, starts, stops):
    from suite2p.extraction import dcnv
    ...
    spks[:, s:e] = dcnv.oasis(
        np.ascontiguousarray(dff[:, s:e], dtype=np.float32),
        2000, TAU, FRAME_RATE)
...
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. The notes say this is a line-by-line reimplementation of `reward_relative.preprocessing.dff`. The trajectory shows the AI initially planned to save deconvolved events, then deliberately switched the default output to dF/F after running side-by-side decoder comparisons and observing better performance on most decoder targets.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only ROIs marked `iscell`, then removes putative interneurons whose dF/F has Pearson correlation `> 0.5` with running speed across in-trial samples.

ii. 
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
d = dff[:, in_trial]
sp = beh['speed'][in_trial].astype(np.float32)
...
r_speed = (dz @ sz) / denom
r_speed = np.nan_to_num(r_speed, nan=0.0)
keep_cells = r_speed <= INTERNEURON_R_THRESH
...
dff_kept = dff[keep_cells]
```

iii. The notes cite the Methods' manual curation plus interneuron exclusion rule and say the resulting counts and exclusion rate are consistent with the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start simply by slicing each trial on the same `[trial_start, teleport)` indices used to define the trial.

ii. 
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
    neural.append(act)
```

iii. The notes say no extra offset or interpolation is needed because the NWB behavior streams are already on imaging-frame time and the decoder asks for alignment to start of trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging-frame resolution, `1 / 15.5078125 Hz = 64.48 ms`, and does not temporally rebin the data.

ii. 
```python
FRAME_RATE = 15.5078125
DT = 1.0 / FRAME_RATE
...
'time_bin_size': 1000.0 / FRAME_RATE,
```

iii. In the notes, the AI says the paper analyzes neural and behavior streams at the imaging frame rate and that the NWB behavior time series are already interpolated onto those frames, so no further rebinning was needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps stored on the `position` time series and loaded into `beh['t']`.

ii. 
```python
beh['t'] = b['position']['timestamps'][:]
...
tt = beh['t'][s:e] - beh['t'][s]
...
inp[0] = tt
```

iii. The AI notes that all behavior streams share the same timestamps after interpolation to imaging frames, so using `position.timestamps` was sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the trial's first timestamp so time starts at zero.

ii. 
```python
tt = beh['t'][s:e] - beh['t'][s]
inp[0] = tt
```

iii. The justification in the notes is simply that decoder input should represent elapsed time from trial onset while preserving the native frame spacing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the exact same `[s:e]` trial slice as the neural data after first truncating behavior and ophys to a common frame count if needed.

ii. 
```python
nframes = min(n_beh, n_ophys)
if n_beh != n_ophys:
    ...
    for k in list(beh.keys()):
        beh[k] = beh[k][:nframes]
    F = F[:, :nframes]
...
tt = beh['t'][s:e] - beh['t'][s]
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. The notes and trajectory justify this with the release's one-frame mismatch issue: because behavior is already on imaging frames, alignment is preserved by cropping both streams to the same frame count before slicing trials.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved decoder input is derived from the behavior `environment` stream. The converter also parses scene names into environment labels, but that parsed value is kept for bookkeeping rather than used directly in `input`.

ii. 
```python
beh = {k: b[k]['data'][:] for k in
       ['position', 'environment', 'lick', 'reward_zone', 'speed',
        'teleport', 'trial_start', 'trial number', 'autoreward', 'scanning']}
...
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
...
inp[1] = env_stream[i]
```

iii. The notes say the `environment` stream matches scene-derived environment labels on all trials, so the AI used the recorded per-frame stream and only kept scene parsing as a validation check.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI reduces the per-frame `environment` stream to one scalar per trial by taking the trial median, then broadcasts that scalar across all time points of the trial.

ii. 
```python
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
...
inp[1] = env_stream[i]
```

iii. The notes justify this by saying environment is a per-trial variable for the decoder and was empirically constant within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial loop index after trials have been segmented by `trial_start` and `teleport`.

ii. 
```python
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = i
```

iii. The notes say the decoder needs within-session trial number, so the AI used the sequential trial index rather than the raw `'trial number'` stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing beyond assigning the 0-based trial index and broadcasting it across the trial.

ii. 
```python
inp[2] = i
```

iii. The AI's notes justify this as the simplest session-local trial counter compatible with the decoder format.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from a per-trial reward-outcome vector `isreward`, which the AI computes from both the sparse `Reward` timestamps and the per-frame `reward_zone` stream.

ii. 
```python
rew_t = b['Reward']['timestamps'][:]
rew_bin = np.zeros_like(beh['position'])
if len(rew_t):
    idx = np.searchsorted(beh['t'], rew_t)
    idx = np.clip(idx, 0, len(rew_bin) - 1)
    rew_bin[idx] = 1
beh['reward'] = rew_bin
...
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
```

iii. The notes justify this by citing the paper's `behavior.get_trial_types` logic: trial reward is defined as reward delivered and reward-zone entry. The AI also notes that `autoreward` is all zeros in the NWB release, so it cannot be used.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i > 0`, the AI uses `isreward[i-1]`. For the first trial of a session it sets previous outcome to `1`, based on the assumption that warm-up trials immediately preceded the imaging session.

ii. 
```python
inp[3] = isreward[i - 1] if i > 0 else 1   # see notes: warm-up trials precede
```

iii. The justification in the notes is explicit: sessions were preceded by warm-up trials with the same reward zone, so the AI chose `1` for the first imaged trial rather than treating it as missing or setting it to `0`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-frame `position` and a per-trial reward-zone identity inferred from the session `scene` string in the NWB identifier. The AI does not infer reward zone from the noisy `reward_zone` behavior stream.

ii. 
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
...
pos = beh['position'][s:e].astype(np.float64)
zone = REWARD_ZONES[zones[i]]
dist = signed_distance_to_zone(pos, zone)
```

iii. The notes say this mirrors `behavior.get_reward_zones` from the paper's code, where reward zone is determined by scene name plus a switch after trial 30. The trajectory shows the AI validated that reward delivery positions and recorded environments matched this scene-based rule across the dataset.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest point of the current trial's reward zone: negative before the zone, `0` inside the zone, positive after the zone.

ii. 
```python
def signed_distance_to_zone(pos, zone):
    lo, hi = zone
    d = np.zeros_like(pos)
    before = pos < lo
    after = pos > hi
    d[before] = pos[before] - lo
    d[after] = pos[after] - hi
    return d
...
dist = signed_distance_to_zone(pos, zone)
```

iii. The notes justify this as a decoder-spec refinement of the paper's reward-relative position: the task asked for distance to "any location in the reward zone", so the AI used nearest-point distance rather than distance only from the zone start.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into the seven instructed bins using explicit comparisons.

ii. 
```python
def bin_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
...
out[0] = bin_distance(dist)
```

iii. The notes say these boundaries were chosen to match the decoder specification exactly, especially treating `0` as its own class.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance from the same per-trial `[s:e]` position slice used for the neural trial.

ii. 
```python
pos = beh['position'][s:e].astype(np.float64)
...
dist = signed_distance_to_zone(pos, zone)
...
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. The notes say that because behavior is already interpolated to imaging frames, using the same slice indexes is sufficient for alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` stream.

ii. 
```python
pos = beh['position'][s:e].astype(np.float64)
...
out[1] = bin_position(pos)
```

iii. The notes state that `position` is already in centimeters on the 450 cm track, so no additional reconstruction is needed.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices per-trial position and discretizes it into five equal-width 90 cm bins via `floor(position / 90)` clipped to `[0, 4]`.

ii. 
```python
def bin_position(pos):
    return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
...
out[1] = bin_position(pos)
```

iii. The notes justify this as the direct implementation of the decoder task's five equal bins spanning a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Category `0` is `< 90 cm`, `1` is `90-180`, `2` is `180-270`, `3` is `270-360`, and `4` is `> 360`, implemented through the `floor/clip` rule.

ii. 
```python
def bin_position(pos):
    return np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
```

iii. The notes say clipping the extremes is acceptable because the converter already excludes the teleport interval and any slight out-of-range values should fall into the end bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking `position[s:e]` on the same trial slice as the neural activity.

ii. 
```python
pos = beh['position'][s:e].astype(np.float64)
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. The AI's notes say the NWB behavior traces were already sampled on imaging frames, so indexing by the same trial window is the intended alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived directly from the per-frame behavior `lick` stream.

ii. 
```python
licks = beh['lick'][s:e].astype(np.float64)
...
out[3] = (licks > 0).astype(np.int64)
```

iii. The notes say the raw lick stream is a cumulative count per frame, so it can be thresholded to produce a binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes licks frame-by-frame with `lick > 0`.

ii. 
```python
out[3] = (licks > 0).astype(np.int64)
```

iii. The notes justify this from the decoder requirement that lick be a binary categorical output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same `[s:e]` slice as the neural trial.

ii. 
```python
licks = beh['lick'][s:e].astype(np.float64)
act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. The notes say no further alignment is needed because licks are already on imaging-frame time.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB session `scene` identifier, not from the per-frame `reward_zone` values.

ii. 
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
...
out[4] = ZONE_TO_IDX[zones[i]]
```

iii. The notes justify this by explicitly pointing to the paper's `behavior.get_reward_zones`, which derives reward zone from scene name and the known switch schedule rather than inferring it from sensor-like streams.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses scene names with regexes, handles both within-environment and between-environment switches, applies the switch after `CHANGE_TRIAL = 30`, and maps `A/B/C` to `0/1/2`.

ii. 
```python
def parse_scene(scene):
    m = re.match(r'^Env(\d)_Location([ABC])$', scene)
    ...
    m = re.match(r'^Env(\d)_Location([ABC])_to_([ABC])$', scene)
    ...
    m = re.match(r'^Env(\d)_([ABC])_to_Env(\d)_([ABC])$', scene)
    ...

def zone_per_trial(scene, ntrials):
    z1, z2, e1, e2 = parse_scene(scene)
    if z2 is None:
        return [z1] * ntrials, [e1] * ntrials
    n1 = min(CHANGE_TRIAL, ntrials)
    return ([z1] * n1 + [z2] * (ntrials - n1),
            [e1] * n1 + [e2] * (ntrials - n1))
```

iii. The notes say this was validated against reward positions and the recorded environment stream across all sessions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from both the sparse `Reward` timestamps and the per-frame `reward_zone` stream through the per-trial `isreward` definition.

ii. 
```python
rew_t = b['Reward']['timestamps'][:]
...
beh['reward'] = rew_bin
...
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
...
out[5] = isreward[i]
```

iii. The notes justify this as the paper's own trial-outcome definition from `behavior.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are snapped to the nearest behavior/imaging frame with `searchsorted`; then each trial is labeled rewarded if it contains at least one reward frame and at least one reward-zone frame. That binary label is broadcast across all time points of the trial.

ii. 
```python
idx = np.searchsorted(beh['t'], rew_t)
idx = np.clip(idx, 0, len(rew_bin) - 1)
rew_bin[idx] = 1
...
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
...
out[5] = isreward[i]
```

iii. The notes say this was chosen because the NWB `autoreward` channel is not populated, while the paper's own behavioral code uses reward-delivered plus reward-zone-entry.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles minor data issues by truncating behavior and ophys streams to a common frame count when they differ slightly, dropping any trial whose teleport lies past that common length, replacing NaN correlations with `0` for interneuron screening, converting saved dF/F NaNs to finite values with `np.nan_to_num`, and skipping sessions with too few usable trials.

ii. 
```python
nframes = min(n_beh, n_ophys)
if n_beh != n_ophys:
    assert abs(n_beh - n_ophys) <= 5, (n_beh, n_ophys)
    ...
    beh[k] = beh[k][:nframes]
    F = F[:, :nframes]
    Fneu = Fneu[:, :nframes]
...
valid = stops < nframes
...
r_speed = np.nan_to_num(r_speed, nan=0.0)
...
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
...
if len(neural) < 2:
    ...
    return None
```

iii. The trajectory shows the one-frame mismatch fix was added after failed full runs on a subset of sessions. The notes justify these choices as robustness measures for release-specific quirks rather than scientific processing changes.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large NWB arrays, computing dF/F over all curated cells and frames, and OASIS deconvolution. Optional plotting is also expensive when enabled.

ii. 
```python
timings = {}
...
F, Fneu, plane_idx = load_fluorescence(f)
timings['load'] = time.time() - t0
...
dff = compute_dff(F, Fneu, starts, stops)
timings['dff'] = time.time() - t1
...
events = deconvolve(dff_kept, starts, stops)
timings['deconv'] = time.time() - t1
...
print(f'... (load {timings["load"]:.1f}, dff {timings["dff"]:.1f}, '
      f'deconv {timings["deconv"]:.1f})', flush=True)
```

iii. The notes explicitly track per-session timing and describe deconvolution as dominant after I/O and dF/F. The trajectory also mentions runtime measurements on large sessions before committing to the full conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining vectorization opportunities are the repeated per-trial loops in `compute_dff`, `deconvolve`, the in-trial mask construction, and the final per-trial assembly loop. The speed-correlation step was already vectorized.

ii. 
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, stops):
    dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
...
for s, e in zip(starts, stops):
    spks[:, s:e] = dcnv.oasis(...)
...
for s, e in zip(starts, stops):
    in_trial[s:e] = True
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. In the notes, the AI says it already vectorized the interneuron correlation with a matrix expression and that the remaining trial loops were kept because trials have variable lengths and because the reference dF/F implementation is inherently trial-wise.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly traverses the same trial windows: once to build NaN-masked fluorescence arrays, again to compute per-trial baselines, again to smooth dF/F, again to deconvolve, again to mark in-trial samples for interneuron filtering, and again to assemble neural/input/output trial objects.

ii. 
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
    fneu_[:, s:e] = Fneu[:, s:e]
...
for s, e in zip(starts, stops):
    ...
    flow[:, s:e] = seg
...
for s, e in zip(starts, stops):
    dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
...
for s, e in zip(starts, stops):
    spks[:, s:e] = dcnv.oasis(...)
...
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes frame this as an accepted tradeoff for staying close to the reference code while still keeping memory use manageable at session scope.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded work is that the converter always computes deconvolved `events` even when the chosen saved neural stream is dF/F, so the expensive deconvolution output is often thrown away. It also computes `envs_scene` even though the final inputs use `env_stream`.

ii. 
```python
events = deconvolve(dff_kept, starts, stops)
timings['deconv'] = time.time() - t1
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
...
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
```

iii. The trajectory makes this explicit: the AI added the `--neural-signal` comparison late in development, switched the default to dF/F, but kept deconvolution in the main path both for comparison and because it had originally planned to save events.
