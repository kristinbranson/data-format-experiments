# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed every NWB file under `/app/data/sub-*/*.nwb`, treated each file as one session, and processed sessions in parallel with `ProcessPoolExecutor`. Inside each file it loaded the required ophys and behavior arrays directly with `h5py`.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))

with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as ex:
    for fn, res in zip(files, ex.map(process_session, files)):
        results.append(res)
```

```python
with h5py.File(fn, 'r') as f:
    scene = f['identifier'][()].decode().split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    F = np.concatenate([f['processing/ophys/Fluorescence/' + p + '/data'][:].T
                        for p in planes], axis=0)
    pos = f[B + 'position/data'][:]
```

iii. In the trajectory, steps 24-27 established that the dataset contains 152 NWB sessions across 11 mice, and step 55 says the converter should process all of them. Step 59 justifies the `spawn` process pool as a workaround for an OpenMP `fork()` issue.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the NWB metadata field `general/subject/subject_id`, collected from each processed session and deduplicated into a sorted subject list. `subject_idx` is then built by indexing each session’s subject into that list.

ii. 
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r[3]['subject'] for r in results},
                  key=lambda s: int(s[1:]))
...
'subjects': subjects,
'subject_idx': np.array([subjects.index(r[3]['subject']) for r in results]),
```

iii. Steps 24-27 note that the NWB files already expose subject metadata and that the dataset spans 11 mice, so the AI used the file metadata rather than directory names as the authoritative subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The converted dataset keeps one top-level session entry per file in `data['neural']`, `data['input']`, and `data['output']`.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
session_id = f['general/session_id'][()].decode()
...
'neural': [r[0] for r in results],
'input': [r[1] for r in results],
'output': [r[2] for r in results],
```

iii. The trajectory consistently refers to “152 sessions” as the unit of processing and validation, especially in steps 24, 55, 68, and 78.

## 1-d. How are the data split into trials?

i. Trials are defined as laps from `trial_start` to `teleport`. The AI finds all frame indices where `trial_start > 0` and all frame indices where `teleport > 0`, pairs them, and then slices `[s:e]` for each trial.

ii. 
```python
tstart = np.where(f[B + 'trial_start/data'][:] > 0)[0]
teleport = np.where(f[B + 'teleport/data'][:] > 0)[0]
...
for s, e in zip(tstart, teleport):
    fseg = F[:, s:e] - NEU_COEF * Fneu[:, s:e]
```

```python
for i in range(1, ntrials):
    s, e = tstart[i], teleport[i]
    T = e - s
    ev = events[:, s:e]
```

iii. Steps 27, 42, 50, and 78 explicitly state that trials are “trial_start to teleport” laps and that the teleport/ITI period is excluded because that matches the on-track analyses in the paper.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops several classes of trials after trial splitting: the first trial of each session, trials with lick-sensor errors, trials whose per-trial environment is not binary 0/1, trials shorter than 2 frames, and trials whose event matrix contains non-finite values.

ii. 
```python
lick_error[i] = (np.sum(seg > 2) / len(seg)) > LICK_ERROR_FRAC
env_trial[i] = int(np.round(np.median(env[s:e])))
...
for i in range(1, ntrials):
    if lick_error[i]:
        continue
    if env_trial[i] not in (0, 1):
        continue
    ...
    if T < 2:
        continue
    ev = events[:, s:e]
    if not np.all(np.isfinite(ev)):
        continue
```

iii. Step 54 says the lick-error rule reproduces the paper’s 81 bad trials exactly, and steps 55 and 78 describe dropping lick-error trials and the first trial because previous-trial outcome is otherwise undefined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from the raw suite2p fluorescence and neuropil traces, `processing/ophys/Fluorescence/*/data` and `processing/ophys/Neuropil/*/data`, after first subsetting to curated `iscell` ROIs.

ii. 
```python
F = np.concatenate([f['processing/ophys/Fluorescence/' + p + '/data'][:].T
                    for p in planes], axis=0)
Fneu = np.concatenate([f['processing/ophys/Neuropil/' + p + '/data'][:].T
                       for p in planes], axis=0)
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0
...
F = F[iscell].astype(np.float32)
Fneu = Fneu[iscell].astype(np.float32)
```

iii. Steps 17, 33, 34, 50, and 78 say the AI intentionally avoided the NWB `Deconvolved` field and instead rebuilt the paper’s neural signal from raw fluorescence plus neuropil, because that is what the paper’s pipeline uses.

## 2-b. How is the `neural` data processed?

i. For each trial, the AI subtracts `0.7 * Fneu`, adds back the trial-mean neuropil, computes a maximin baseline with Gaussian smoothing and 300-frame min/max filters, forms dF/F, smooths dF/F with a 2-frame Gaussian, and then deconvolves with `suite2p.extraction.dcnv.oasis` using `tau = 0.7` and `fs = rate / nplanes`.

ii. 
```python
for s, e in zip(tstart, teleport):
    fseg = F[:, s:e] - NEU_COEF * Fneu[:, s:e]
    fseg = fseg + NEU_COEF * np.mean(Fneu[:, s:e], axis=1, keepdims=True)
    flow = ndi.gaussian_filter1d(fseg, BASELINE_SMOOTH, axis=1)
    flow = ndi.minimum_filter1d(flow, BASELINE_WIN, axis=-1)
    flow = ndi.maximum_filter1d(flow, BASELINE_WIN, axis=-1)
    d = (fseg - flow) / np.abs(flow)
    d = ndi.gaussian_filter1d(d, DFF_SMOOTH, axis=1).astype(np.float32)
    dff[:, s:e] = d
    events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
```

iii. Steps 32-36 and 50-55 describe this as the paper-matching neural preprocessing recipe: neuropil coefficient `0.7`, maximin baseline, 2-frame smoothing, and OASIS deconvolution with `tau=0.7`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered twice: first by retaining only ROIs marked `iscell`, and then by excluding putative interneurons whose dF/F is too correlated with running speed (`r > 0.5`).

ii. 
```python
F = F[iscell].astype(np.float32)
Fneu = Fneu[iscell].astype(np.float32)
...
on_track = ~np.isnan(dff[0, :])
sp = speed[on_track]
dsub = dff[:, on_track]
...
r = (dsub @ spc) / denom
is_int = np.nan_to_num(r, nan=0.0) > INT_R_THRESH
keep = ~is_int
events = events[keep]
```

iii. Steps 33, 34, 37, 38, 50, and 78 explicitly say that `iscell` curation and the `r > 0.5` interneuron filter were copied from the paper/repo.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by using the `trial_start`/`teleport` trial slices. There is no additional event-centered shifting or padding.

ii. 
```python
tstart = np.where(f[B + 'trial_start/data'][:] > 0)[0]
teleport = np.where(f[B + 'teleport/data'][:] > 0)[0]
...
s, e = tstart[i], teleport[i]
ev = events[:, s:e]
```

iii. Steps 42, 50, and 78 state that the temporal alignment event is trial start and that the emitted per-trial matrices are the raw `[trial_start:teleport]` laps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data stay at the original per-plane imaging rate `fs = rate / nplanes` with no rebinning or resampling. The metadata hard-code the corresponding time bin size as `1000 / 15.5078125` ms.

ii. 
```python
rate = float(f['processing/ophys/Fluorescence/' + planes[0] +
               '/starting_time'].attrs['rate'])
fs = rate / nplanes
...
'time_bin_size': 1000.0 / (15.5078125),
```

iii. Steps 24, 26, and 50 describe the data as being sampled at about 15.5 Hz per plane and repeatedly note that the AI kept that native sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the neural/behavior frame count within a trial plus the session sampling rate `fs`; the AI does not use behavior timestamps here.

ii. 
```python
rate = float(f['processing/ophys/Fluorescence/' + planes[0] +
               '/starting_time'].attrs['rate'])
fs = rate / nplanes
...
T = e - s
t_in_trial = np.arange(T, dtype=np.float32) / fs
```

iii. The trajectory does not give a separate argument for this beyond steps 24, 26, and 55 noting the stable ~15.5 Hz sampling and using that native frame rate throughout the conversion.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For a trial of length `T`, the AI constructs `0, 1/fs, 2/fs, ...` with `np.arange(T) / fs`. This is then used directly as the first input channel.

ii. 
```python
t_in_trial = np.arange(T, dtype=np.float32) / fs
inp = np.stack([
    t_in_trial,
    np.full(T, env_trial[i], dtype=np.float32),
    np.full(T, i, dtype=np.float32),
    np.full(T, rewarded[i - 1], dtype=np.float32),
], axis=0)
```

iii. The trajectory contains no separate justification beyond using the uniform frame rate to keep all trial-aligned streams on a common sample grid.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is aligned by construction: it has exactly the same length `T = e - s` as the neural event slice `events[:, s:e]` for that trial.

ii. 
```python
T = e - s
ev = events[:, s:e]
...
t_in_trial = np.arange(T, dtype=np.float32) / fs
```

iii. Steps 55 and 78 describe the converter as building all trial-level inputs and outputs from the same `[s:e]` slices, so no extra alignment step was justified separately.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the raw behavior array `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. 
```python
env = f[B + 'environment/data'][:]
...
env_trial[i] = int(np.round(np.median(env[s:e])))
```

iii. Steps 26, 27, 34, and 55 identify `environment` as the per-trial ENV1/ENV2 signal and say the AI confirmed the session/trial coding before writing the converter.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI takes the median environment value and rounds it to an integer, then broadcasts that constant value across all time bins of the trial.

ii. 
```python
env_trial[i] = int(np.round(np.median(env[s:e])))
...
np.full(T, env_trial[i], dtype=np.float32),
```

iii. Step 54 says the AI checked “env per trial consistency,” which is the reason it felt comfortable collapsing the raw per-frame environment stream to a single constant per trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `i` over the already-split trial list; it is not read from a stored `trial number` behavior variable.

ii. 
```python
for i in range(1, ntrials):
    ...
    inp = np.stack([
        t_in_trial,
        np.full(T, env_trial[i], dtype=np.float32),
        np.full(T, i, dtype=np.float32),
```

iii. The trajectory does not record a separate defense of the exact indexing convention, but steps 55 and 78 describe trial number as one of the broadcast per-trial variables constructed after trial splitting.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond broadcasting the integer loop index across all time bins in that trial.

ii. 
```python
np.full(T, i, dtype=np.float32),
```

iii. No separate justification was recorded beyond the general plan in steps 55 and 78 to emit trial-level decoder inputs as constant channels.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward timestamps (`Reward/timestamps`) plus the reward-zone-entry flag (`reward_zone/data`). The AI first builds a per-trial `rewarded` label for the current trial and then uses `rewarded[i - 1]` as the previous outcome input on the next trial.

ii. 
```python
reward_ts = f[B + 'Reward/timestamps'][:]
rzone_flag = f[B + 'reward_zone/data'][:]
...
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. Steps 27, 53, 54, 55, and 78 justify this by saying the scene-based reward-zone rule was validated against `reward_zone` flags and that the converter follows the paper’s rewarded-versus-omission notion of trial outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI drops the first trial entirely, then assigns `rewarded[i - 1]` as a constant input channel for every later trial.

ii. 
```python
# first trial of each session is dropped: the previous trial's outcome
# (a decoder input) is unknown for it
for i in range(1, ntrials):
    ...
    np.full(T, rewarded[i - 1], dtype=np.float32),
```

iii. Steps 55 and 78 explicitly justify dropping the first trial on the grounds that previous-trial outcome is undefined there.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from trial position (`position/data`) and a per-trial reward-zone label inferred from the session scene name (`identifier`). The AI does not infer the zone from per-trial reward-zone occupancy in the final converter.

ii. 
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
zone_lab = zone_labels_for_session(scene, ntrials)
...
p = pos[s:e]
zone = REWARD_ZONES[zone_lab[i]]
dist = signed_distance_to_zone(p, zone)
```

iii. Steps 27, 42, 53, 54, 55, and 78 say the AI chose scene-based zone labels because the scene string determines the active zone and because this rule was validated against the reward-zone flag with 0 mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest edge of the active reward zone: negative before the zone, zero inside it, and positive after it. It then discretizes that continuous distance into seven categories.

ii. 
```python
def signed_distance_to_zone(pos, zone):
    start, stop = zone
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
    return d
...
dist = signed_distance_to_zone(p, zone)
bin_reward_distance(dist)
```

iii. Steps 55 and 78 summarize this as one of the planned decoder outputs, using the paper’s known zone coordinates and the task’s requested distance bins.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into seven bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii. 
```python
def bin_reward_distance(d):
    out = np.zeros(d.shape, dtype=np.int16)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
```

iii. Step 55 says the AI wrote the converter to use the task’s specified reward-zone-distance bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from the same per-trial `[s:e]` position slice that defines the neural event slice, so it is frame-aligned sample by sample.

ii. 
```python
s, e = tstart[i], teleport[i]
ev = events[:, s:e]
...
p = pos[s:e]
dist = signed_distance_to_zone(p, zone)
```

iii. The trajectory’s repeated theme in steps 42, 55, and 78 is that all outputs are produced on the same trial frame grid as the neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. 
```python
pos = f[B + 'position/data'][:]
...
p = pos[s:e]
```

iii. Steps 26, 27, and 55 identify `position` as one of the validated behavior streams used directly in the final converter.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the raw trial position slice and discretizes it with `np.digitize` into five equal 90 cm track bins.

ii. 
```python
def bin_position(pos):
    edges = np.array([90.0, 180.0, 270.0, 360.0])
    return np.digitize(pos, edges).astype(np.int16)
...
bin_position(p)
```

iii. Step 55 says the converter will emit “position bins” as one of the outputs, following the task specification.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are produced with bin edges `[90, 180, 270, 360]`, yielding bins `<90`, `90-180`, `180-270`, `270-360`, and `>360` cm.

ii. 
```python
def bin_position(pos):
    edges = np.array([90.0, 180.0, 270.0, 360.0])
    return np.digitize(pos, edges).astype(np.int16)
```

iii. Step 55 explicitly lists “position bins” among the outputs the AI intended to write according to the decoder instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by taking the same `[s:e]` slice as the neural event matrix for each trial.

ii. 
```python
s, e = tstart[i], teleport[i]
ev = events[:, s:e]
p = pos[s:e]
```

iii. No extra justification was recorded beyond the general decision, repeated in steps 55 and 78, to keep all trial streams on the same sample indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. 
```python
lick = f[B + 'lick/data'][:]
...
lk = (lick[s:e] > 0).astype(np.int16)
```

iii. Steps 26, 27, 33, 54, and 55 mention `lick` as both a raw behavior stream and a source of lick-sensor quality control.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the raw lick count per sample: any positive value becomes `1`, and zero stays `0`.

ii. 
```python
lk = (lick[s:e] > 0).astype(np.int16)
...
outp = np.stack([
    bin_reward_distance(dist),
    bin_position(p),
    bin_speed(sp_t),
    lk,
```

iii. Step 55 names lick as one of the categorical outputs and uses a binary representation to match the decoder task.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the same trial frame interval `[s:e]` used for the neural events.

ii. 
```python
s, e = tstart[i], teleport[i]
ev = events[:, s:e]
lk = (lick[s:e] > 0).astype(np.int16)
```

iii. As with the other time-varying outputs, steps 55 and 78 imply alignment by shared per-trial frame indices rather than any extra interpolation step.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session scene string in `identifier`, which encodes the active zone and any switch (`..._to_...`) across trials.

ii. 
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
zone_lab = zone_labels_for_session(scene, ntrials)
...
np.full(T, ZONE_IDX[zone_lab[i]], dtype=np.int16),
```

iii. Steps 27, 42, 53, 54, 55, and 78 justify using scene names because they uniquely determine zone identity and were validated against the reward-zone flags with zero mismatches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For non-switch sessions the last character of the scene name is repeated for all trials. For switch sessions the pre-switch label is used for the first 30 trials and the post-switch label for later trials, then those labels are mapped to indices `A=0`, `B=1`, `C=2` and broadcast across time within each trial.

ii. 
```python
def zone_labels_for_session(scene, ntrials):
    if '_to_' in scene:
        pre, post = scene.split('_to_')
        z0, z1 = pre[-1], post[-1]
        labels = [z0] * min(SWITCH_TRIAL, ntrials) + [z1] * max(0, ntrials - SWITCH_TRIAL)
    else:
        labels = [scene[-1]] * ntrials
    return labels
```

iii. Step 42 says the AI empirically confirmed that the switch happens at trial 30, and steps 53-55 say the scene rule matched the reward-zone flag perfectly.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward timestamps (`Reward/timestamps`) together with trial boundaries and reward-zone occupancy. The per-trial `rewarded` label is then written as the reward-outcome output.

ii. 
```python
reward_ts = f[B + 'Reward/timestamps'][:]
rzone_flag = f[B + 'reward_zone/data'][:]
...
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
...
np.full(T, rewarded[i], dtype=np.int16),
```

iii. Steps 27, 54, 55, and 78 justify this as the paper-style rewarded-versus-omission labeling, using reward timing together with in-zone behavior.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls between the position timestamps at the trial start and end and whether the reward-zone flag was ever positive on that trial. The resulting binary label is then broadcast across the whole trial.

ii. 
```python
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
...
np.full(T, rewarded[i], dtype=np.int16),
```

iii. The trajectory does not record a separate micro-justification beyond steps 54, 55, and 78 describing this as the validated reward-outcome rule used in the final converter.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues defensively. It truncates imaging and behavior to their common frame count, truncates trial-start and teleport arrays to their common count, drops trials whose teleport would exceed the aligned frame count, rejects trials with lick-sensor errors, nonbinary environment values, very short length, or non-finite event values, and uses a `spawn` process pool to avoid multiprocessing crashes.

ii. 
```python
nframes = min(F.shape[1], len(pos))
F = F[:, :nframes]
Fneu = Fneu[:, :nframes]
pos = pos[:nframes]; speed = speed[:nframes]; lick = lick[:nframes]
env = env[:nframes]; rzone_flag = rzone_flag[:nframes]; ts = ts[:nframes]
npairs = min(len(tstart), len(teleport))
tstart, teleport = tstart[:npairs], teleport[:npairs]
keep_tr = teleport < nframes
tstart, teleport = tstart[keep_tr], teleport[keep_tr]
```

```python
if lick_error[i]:
    continue
if env_trial[i] not in (0, 1):
    continue
if T < 2:
    continue
if not np.all(np.isfinite(ev)):
    continue
```

iii. Steps 59 and 61-63 justify the process-pool and length-truncation fixes as responses to observed runtime failures and frame-count mismatches in dual-plane sessions. Steps 54, 55, and 78 justify the lick-error exclusion specifically.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are the per-session NWB reads, the per-trial dF/F plus OASIS deconvolution loop, and writing the final pickle. The code parallelizes session processing to reduce wall-clock time.

ii. 
```python
with h5py.File(fn, 'r') as f:
    F = np.concatenate([...], axis=0)
    Fneu = np.concatenate([...], axis=0)
```

```python
for s, e in zip(tstart, teleport):
    ...
    events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
...
with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as ex:
```

iii. Steps 51, 52, 55, 57, 67, and 68 discuss timing repeatedly: the AI benchmarked single-session processing, concluded the dF/F + OASIS pipeline was fast enough, and then ran the full conversion in parallel over all sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are the trial loop for dF/F/deconvolution, the trial loop that derives `rewarded`, `lick_error`, and `env_trial`, and the later trial loop that constructs `neural_trials`, `input_trials`, and `output_trials`. These could only be vectorized with padding or more complicated bookkeeping because trial lengths vary.

ii. 
```python
for s, e in zip(tstart, teleport):
    ...
```

```python
for i, (s, e) in enumerate(zip(tstart, teleport)):
    ...
for i in range(1, ntrials):
    ...
```

iii. The trajectory does not explicitly argue about vectorization, but steps 51-57 show that the AI chose parallel session-level execution instead of trying to remove the variable-length per-trial loops.

## 13-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the same trial boundaries. It loops once to compute dF/F and events, once to derive per-trial `rewarded`/`lick_error`/`env_trial`, and once more to assemble trial outputs. It also reads the full set of behavior arrays for every session even when only a subset of them is needed for a specific intermediate calculation.

ii. 
```python
for s, e in zip(tstart, teleport):
    ...
for i, (s, e) in enumerate(zip(tstart, teleport)):
    ...
for i in range(1, ntrials):
    ...
```

iii. The trajectory does not frame this as a problem, but steps 55-57 show that the AI accepted these repeated passes because the per-session runtime was already low after parallelization.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest discarded intermediate is the full-session `dff` array, which is kept only long enough to compute interneuron correlations and is never saved. The local `trials_kept` list is also populated but never used downstream.

ii. 
```python
dff = np.full(F.shape, np.nan, dtype=np.float32)
events = np.full(F.shape, np.nan, dtype=np.float32)
...
dff[:, s:e] = d
events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
...
dsub = dff[:, on_track]
...
trials_kept = []
...
trials_kept.append(i)
```

iii. The trajectory does not explicitly justify these discarded intermediates. They are visible from the final code rather than from a stated design rationale.
