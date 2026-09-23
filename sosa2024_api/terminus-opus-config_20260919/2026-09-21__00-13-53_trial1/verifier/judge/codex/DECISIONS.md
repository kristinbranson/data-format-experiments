# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data/sub-*/sub-*.nwb`, then reads each file with `pynwb.NWBHDF5IO`. Within each session it pulls the behavior time series, fluorescence, neuropil, ROI metadata, and reward timestamps into NumPy arrays before converting trials.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))

with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    oph = nwb.processing['ophys'].data_interfaces
```

iii. The notes say the released dataset is one directory per subject and one NWB per session, so globbing all `sub-*/*.nwb` was intended to include the full public dataset. The trajectory and notes also emphasize that all I/O must go through `pynwb`, not `h5py`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from each NWB file’s `subject.subject_id`, and the final `subjects` list is rebuilt from the converted session metadata.

ii.
```python
d['subject'] = nwb.subject.subject_id

subjects = sorted({inf['subject'] for inf in infos}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(inf['subject']) for inf in infos], dtype=np.int64)
```

iii. The notes state that the public dataset contains 11 subject directories and that NWB metadata expose the mouse IDs directly, so the AI treated NWB subject IDs as the authoritative subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session-level metadata such as scene, date, and subject are extracted from the file, and each converted session becomes one element in `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))

d['scene'] = nwb.identifier.split('/')[-1]
d['date'] = nwb.identifier.split('/')[-2]

for (n, i, o, info) in results:
    if len(n) < 2:
        continue
    neural.append(n)
    inputs.append(i)
    outputs.append(o)
    infos.append(info)
```

iii. In the notes, the AI explicitly describes the DANDI layout as one NWB per session and reports 152 sessions total, so it uses file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the behavior streams: `trial_start` gives starts and `teleport` gives ends. The code uses the reference paper’s indexing convention and slices each trial as frames `[trial_start-1, teleport-1)`.

ii.
```python
d['trial_starts'] = np.where(np.asarray(beh['trial_start'].data[:]) > 0)[0]
d['teleports'] = np.where(np.asarray(beh['teleport'].data[:]) > 0)[0]

starts = raw['trial_starts']
stops = raw['teleports']
starts = np.maximum(starts, 1)

for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
```

iii. The file docstring and notes both say the AI intentionally followed `reward_relative.glmUtils.get_timeseries_data`, which uses the `[trial_start-1, teleport-1)` convention. The notes also mention that teleport/ITI samples are excluded to match the reference analysis windows.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with lick-sensor errors, trials shorter than 2 samples, and any trial whose neural matrix still contains non-finite values after preprocessing. It does not keep the human reference’s `<50`-timepoint threshold.

ii.
```python
lick_error[i] = lk.size > 0 and (np.sum(lk > LICK_ERROR_COUNT) / lk.size) > LICK_ERROR_FRAC

for i, (s, e) in enumerate(zip(starts, stops)):
    if lick_error[i]:
        continue
    sl = slice(s - 1, e - 1)
    T = (e - 1) - (s - 1)
    if T < 2:
        continue
    ev_trial = events[:, sl]
    if not np.all(np.isfinite(ev_trial)):
        continue
```

iii. The notes justify dropping lick-error trials by arguing that the paper flags these trials and that this decoder format cannot store NaNs for a categorical lick output. The notes also say sessions with fewer than 2 usable trials would be dropped, and the finite-value check is described as a guard against bad frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from raw `Fluorescence` and `Neuropil` ROI response series, plus the ROI curation field `iscell`. The AI explicitly does not use the NWB `Deconvolved` signal as the model input.

ii.
```python
rrs = oph['Fluorescence'].roi_response_series[pl]
Fs.append(np.asarray(rrs.data[:], dtype=np.float32))
Fneus.append(np.asarray(oph['Neuropil'].roi_response_series[pl].data[:], dtype=np.float32))

ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
iscell = np.asarray(ps['iscell'].data[:])[:, 0]
```

iii. The notes repeatedly state that NWB `Deconvolved` is suite2p’s own deconvolution and not the paper’s analyzed signal, so the AI recomputed the paper’s event signal from `Fluorescence` and `Neuropil`.

## 2-b. How is the `neural` data processed?

i. The AI ports the paper’s `dff()` logic: mask to trial segments, subtract `0.7 * Fneu`, add back the per-segment neuropil mean, compute a per-segment maximin baseline, form dF/F, smooth it, then deconvolve with suite2p OASIS. It also supports the paper’s `keep_teleports` handling for selected subject/day combinations.

ii.
```python
def compute_events(F, Fneu, starts, stops, fs, keep_teleports=False):
    starts, stops = baseline_segments(starts, stops, keep_teleports)
    ...
    f_ -= NEU_COEF * fneu_
    ...
    f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
    seg = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH])
    seg = minimum_filter1d(seg, BASELINE_WINDOW, axis=-1)
    seg = maximum_filter1d(seg, BASELINE_WINDOW, axis=-1)
    ...
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
    events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl]), 2000, TAU, fs)
```

iii. The notes say this was chosen to match the paper’s Methods and `preprocessing.dff()` exactly, including the teleport-specific baseline rule taken from `teleport_metadata.py`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only ROIs with `iscell == 1`, then removes putative interneurons whose dF/F is too correlated with running speed (`r > 0.5`).

ii.
```python
F = raw['F'][raw['iscell']]
Fneu = raw['Fneu'][raw['iscell']]

nanmask = ~np.isnan(dff[0, :])
sp = speed_all[nanmask]
dm = dff[:, nanmask]
...
speed_corr = (dm_c @ sp_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
keep_cells = ~is_int
events = events[keep_cells]
```

iii. The notes describe this as direct replication of the paper’s curation: suite2p/manual `iscell` plus putative-interneuron exclusion based on dF/F-speed correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by defining each trial window from trial start onward. No extra temporal shifting is applied after trial slicing.

ii.
```python
starts = raw['trial_starts']
stops = raw['teleports']
...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    ev_trial = events[:, sl]
```

iii. The notes explicitly say the alignment event is trial start and that the reference code already uses the same trial window, so simple trial slicing is the intended alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native per-plane imaging frame period, `1000 / fs` ms, with `fs = scan_rate / n_planes`. No temporal rebinning is applied.

ii.
```python
d['scan_rate'] = float(oph['Fluorescence'].roi_response_series[planes[0]].rate)
d['n_planes'] = len(planes)
d['fs'] = d['scan_rate'] / len(planes)

'time_bin_size': float(1000.0 / infos[0]['fs'])
```

iii. The notes say the NWB behavior is already on the imaging frame clock and that the paper sampled both neural and behavioral series at about 15.5 Hz, so the AI kept the native sampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps, specifically `beh['position'].timestamps`.

ii.
```python
d['time'] = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
...
time_v = raw['time']
...
t_rel = time_v[sl] - time_v[s - 1]
```

iii. The notes say the behavior series are already aligned on the imaging frame clock, so any behavior timestamp stream would work; the AI chose the position timestamps as the session timebase.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at the trial’s first included frame is subtracted from the timestamps of all frames in that trial.

ii.
```python
t_rel = time_v[sl] - time_v[s - 1]
inp[0] = t_rel
```

iii. The notes describe this as direct alignment to trial start, with no extra interpolation or smoothing because the behavior is already on the same clock as the neural data.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the exact same frame slice `sl = slice(s - 1, e - 1)` for both neural and behavioral arrays.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
t_rel = time_v[sl] - time_v[s - 1]
```

iii. The notes state that no extra realignment is needed because neural and behavior were already sampled on the same imaging frame clock in the NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
d['env'] = np.asarray(beh['environment'].data[:], dtype=np.float64)
...
env = raw['env']
...
ev = env[sl]
```

iii. The notes identify the NWB `environment` series as the per-frame encoding of ENV1 versus ENV2 and use it as the source for the decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI converts the per-frame environment series into a per-trial constant by taking the median of valid (`>= 0`) samples within the trial, rounding it to an integer, and then broadcasting that value across the trial’s time bins.

ii.
```python
ev = env[sl]
ev = ev[ev >= 0]
env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0
...
inp[1] = env_trial[i]
```

iii. The notes justify this by the decoder specification: environment type is a per-trial binary input. The edge-case notes also mention day-8 environment switches and say the within-trial median still labels those trials correctly.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the reconstructed trial order itself, not from the NWB `trial number` series. The loop index over trial windows becomes the within-session trial number.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = i
```

iii. The notes say the paper/reference logic is organized around trial windows from `trial_start` and `teleport`, and the AI uses that same ordering for a within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond assigning the current trial index and broadcasting it across all time bins in the trial.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
...
inp[2] = i
```

iii. The notes characterize this as a per-trial decoder input, so the same scalar is repeated across the full trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from reward timestamps and reward-zone occupancy. The AI first builds a per-trial `isreward` array from `Reward` timestamps and `reward_zone` activity, then uses the previous trial’s `isreward` entry as the input.

ii.
```python
rew_frames = np.searchsorted(time_v, raw['reward_times'])
...
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
isreward[i] = int(bool(has_rew) and bool(in_zone))
...
inp[3] = isreward[i - 1] if i > 0 else 0
```

iii. The notes say the AI chose to mirror the paper’s `behavior.get_trial_types` logic for rewarded versus omitted trials, rather than defining outcome from reward timestamps alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI computes a per-trial binary reward outcome, then shifts it back by one trial. The first trial in each session is assigned 0 because there is no preceding imaged trial.

ii.
```python
isreward = np.zeros(ntrials_all, dtype=np.int64)
...
isreward[i] = int(bool(has_rew) and bool(in_zone))
...
inp[3] = isreward[i - 1] if i > 0 else 0
```

iii. The notes explicitly justify the first-trial rule as a conservative encoding because warm-up trials before imaging are not present in the files.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal’s position and a per-trial reward-zone label inferred from the session scene name plus the trial-30 switch rule. The raw `reward_zone` series is used for reward-outcome logic, but not to infer zone identity.

ii.
```python
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
...
p = pos[sl]
zlab = labels[i]
z0, z1 = ZONE_DICT[zlab]
rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. The notes say this choice was made to match `reward_relative.behavior.get_reward_zones` from the reference code. The AI also reports a sanity check that scene-derived zone labels matched observed reward-zone positions on every trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest point of the active reward zone: negative before the zone, zero inside it, and positive after it. It then converts that continuous distance to categories.

ii.
```python
def discretize_reward_distance(pos, zone_start, zone_end):
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    ...
    return cat, d
```

iii. The notes say the paper’s reward-relative analyses motivated this transformation, but the task required a linear distance to the nearest reward-zone point rather than the paper’s circular reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into 7 categories with explicit comparisons: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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

iii. The notes say this explicit implementation was chosen to match the decoder task’s bin definitions exactly, especially the special category for exactly zero distance.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing it from `position[sl]` for the same trial slice used to extract the neural matrix.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
p = pos[sl]
rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. The notes say all time-varying outputs are built on the already aligned imaging-frame clock, so shared slicing is sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
d['pos'] = np.asarray(beh['position'].data[:], dtype=np.float64)
...
p = pos[sl]
```

iii. The notes identify NWB `position` as the mouse’s track position in centimeters and use it directly for the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices position to the trial window and discretizes it into five track bins; there is no further smoothing or interpolation.

ii.
```python
p = pos[sl]
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
out[1] = pos_cat
```

iii. The notes say the track is 450 cm long and the decoder task asked for five equal bins, so simple thresholding of the raw position values was sufficient.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It uses `np.digitize` with edges `[90, 180, 270, 360]`, yielding five 90 cm bins.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
```

iii. The notes justify this directly from the task specification: five equal-sized bins spanning a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial frame slice as the neural data.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
p = pos[sl]
```

iii. The notes say no extra alignment is necessary because behavior is already sampled at imaging-frame resolution.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii.
```python
d['lick'] = np.asarray(beh['lick'].data[:], dtype=np.float64)
...
lk = lick[sl]
```

iii. The notes identify the NWB `lick` series as cumulative lick counts per imaging frame and use that as the source signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick signal: any positive cumulative lick count becomes 1, otherwise 0.

ii.
```python
lick_cat = (lk > 0).astype(np.int64)
out[3] = lick_cat
```

iii. The notes say the decoder task requires a binary lick output, and they mention the paper’s own conversion of remaining lick counts to a binary vector.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by extracting the lick values from the same trial slice used for the neural data.

ii.
```python
sl = slice(s - 1, e - 1)
ev_trial = events[:, sl]
lk = lick[sl]
```

iii. The notes again rely on the shared imaging-frame clock and the common trial slice for alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session scene name (`identifier`) and the trial index, via the reference rule that some sessions switch reward zones after 30 trials.

ii.
```python
d['identifier'] = nwb.identifier
d['scene'] = nwb.identifier.split('/')[-1]
...
labels = get_reward_zone_labels(raw['scene'], ntrials_all)
zone_code = {'A': 0, 'B': 1, 'C': 2}
```

iii. The notes say the AI deliberately copied `behavior.get_reward_zones` from the reference code and validated the resulting zone labels against the observed reward-zone positions in the raw data.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string to decide whether the session is fixed-zone or switch-zone, applies the switch-at-trial-30 rule when needed, converts `A/B/C` to `0/1/2`, and broadcasts that per-trial label across time.

ii.
```python
def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    if scene.endswith('LocationA'):
        return ['A'] * ntrials
    ...
    for first in ('A', 'B', 'C'):
        if f'{first}_to' in scene:
            second = scene[-1]
            ...
            return [first] * n0 + [second] * (ntrials - n0)

out[4] = zone_code[zlab]
```

iii. The notes cite the paper’s methods and the reference behavior code as the justification for both the zone mapping and the switch timing.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` timestamps together with within-trial reward-zone occupancy (`reward_zone > 0`).

ii.
```python
d['reward_times'] = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
d['rzone'] = np.asarray(beh['reward_zone'].data[:], dtype=np.float64)
...
has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
in_zone = np.any(rzone[sl] > 0)
isreward[i] = int(bool(has_rew) and bool(in_zone))
```

iii. The notes say this follows the paper’s `behavior.get_trial_types` logic, where rewarded trials are those with reward delivery and reward-zone entry, not just any reward timestamp.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices with `np.searchsorted`, the code computes a per-trial binary `isreward`, and that value is broadcast across all time bins in the trial.

ii.
```python
rew_frames = np.searchsorted(time_v, raw['reward_times'])
rew_frames = rew_frames[rew_frames < len(time_v)]
...
isreward[i] = int(bool(has_rew) and bool(in_zone))
...
out[5] = isreward[i]
```

iii. The notes justify this as reference-faithful trial typing and report sanity checks that reward deliveries fall inside the correct reward zone on rewarded trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively: it truncates behavior and ophys arrays to a common length when they differ by one frame, crops trial indices after truncation, guards against `trial_start == 0`, filters trials with lick-sensor failures, and skips any trial whose neural segment is non-finite after preprocessing.

ii.
```python
nB = len(d['time'])
nF = d['F'].shape[1]
n = min(nB, nF)
...
if d['n_trunc'] > 0:
    for k in ('time', 'pos', 'speed', 'lick', 'rzone', 'env', 'trialnum', 'scanning'):
        d[k] = d[k][:n]
    d['F'] = d['F'][:, :n]
    d['Fneu'] = d['Fneu'][:, :n]
    d['trial_starts'] = d['trial_starts'][d['trial_starts'] < n]
    d['teleports'] = d['teleports'][d['teleports'] < n]

starts = np.maximum(starts, 1)

if lick_error[i]:
    continue
if T < 2:
    continue
if not np.all(np.isfinite(ev_trial)):
    continue
```

iii. The notes explain the one-frame mismatch as a known alignment artifact in 10 two-plane sessions and say truncation leaves all trial boundaries intact. The edge-case review in the notes also documents why the trial-start clamp and lick-error filtering were added.

## 13-a. What are the most time-consuming steps of the code?

i. The AI identifies full NWB reads and the dF/F + OASIS computation as the main bottlenecks. It also notes that these costs dominate wall-clock time and motivates multiprocessing across sessions.

ii.
```python
raw = read_nwb_session(path)
...
dff, events = compute_events(F, Fneu, starts, stops, raw['fs'],
                             keep_teleports=keep_teleports)
...
with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
    for res in ex.map(_worker, remaining):
        results.append(res)
```

iii. In Step 6 and Step 7 of the notes, the AI explicitly says loading `Fluorescence`/`Neuropil` is the main I/O cost and dF/F plus deconvolution is the main compute cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the interneuron-correlation computation, but the code still uses repeated trial-by-trial loops in `compute_events()` and `convert_session()` that could in principle be reduced or fused. Optional plotting also loops over kept trials and recomputes some per-trial quantities.

ii.
```python
for s, e in zip(starts, stops):
    f_[:, s - 1:e - 1] = F[:, s - 1:e - 1]
    ...

for s, e in zip(starts, stops):
    sl = slice(s - 1, e - 1)
    ...

for i, (s, e) in enumerate(zip(starts, stops)):
    ...
    rz_cat, _ = discretize_reward_distance(p, z0, z1)
```

iii. The notes say the biggest explicit vectorization win was replacing the original per-cell correlation loop with a matrix-vector computation. They otherwise leave the trial loops in place because the processing is trial-segment based and variable length.

## 13-c. What processing does the code repeat multiple times?

i. The code revisits the same trial segments multiple times during neural preprocessing: once to copy valid samples into masked arrays, again to compute baselines, and again to smooth/deconvolve. It then iterates over trials again to build decoder inputs/outputs, and optional processing plots recompute reward-zone distances for display.

ii.
```python
for s, e in zip(starts, stops):
    f_[:, s - 1:e - 1] = F[:, s - 1:e - 1]
    fneu_[:, s - 1:e - 1] = Fneu[:, s - 1:e - 1]

for s, e in zip(starts, stops):
    sl = slice(s - 1, e - 1)
    ...

for s, e in zip(starts, stops):
    sl = slice(s - 1, e - 1)
    dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
    events[:, sl] = dcnv.oasis(...)

for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes acknowledge that dF/F/deconvolution dominates run time and describe the implementation as a close port of the reference pipeline, which naturally performs several passes over each trial segment.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and retains dF/F only as an intermediate for interneuron filtering and deconvolution; the final dataset stores only the deconvolved events. It also reads some metadata arrays such as `trial number`, `scanning`, and `planeIdx` that are not used in the final exported tensors.

ii.
```python
dff, events = compute_events(F, Fneu, starts, stops, raw['fs'],
                             keep_teleports=keep_teleports)
...
nanmask = ~np.isnan(dff[0, :])
...
events = events[keep_cells]

d['trialnum'] = np.asarray(beh['trial number'].data[:], dtype=np.float64)
d['scanning'] = np.asarray(beh['scanning'].data[:], dtype=np.float64)
d['planeIdx'] = np.asarray(ps['planeIdx'].data[:])[roi_idx]
```

iii. The notes explicitly say the decoder should use deconvolved events, not dF/F, so dF/F is only an internal processing product. The extra fields are loaded during session parsing for checks and metadata, but they do not enter the final decoder arrays.
