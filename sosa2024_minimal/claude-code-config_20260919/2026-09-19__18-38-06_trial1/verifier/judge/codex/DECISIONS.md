# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file matching `/app/data/sub-*/sub-*.nwb`, sorts them, and processes each file as one session with `h5py`. Trials are then extracted inside each session by reading behavioral arrays and slicing them into per-trial windows.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')),
               key=lambda p: (int(os.path.basename(p).split('_')[0][5:]),
                              os.path.basename(p)))

with h5py.File(path, 'r') as f:
    beh = f['processing/behavior/BehavioralTimeSeries']
    trial_starts = np.nonzero(beh['trial_start/data'][:])[0]
    teleports = np.nonzero(beh['teleport/data'][:])[0]
```

iii. In the trajectory, the agent explicitly scanned the dataset layout and reported 152 NWB files across 11 mice, then described the output as “152 sessions, 11 mice” in the final summary. The justification was that every NWB file in the dataset should be included as a session.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the `subject_id` stored inside each NWB file, then uniqued and sorted numerically for the final `subjects` list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()

subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(s[1:]))
```

iii. In the trajectory, the agent inspected both the directory names and the NWB metadata and treated the NWB `subject_id` as authoritative.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session/day label is read from `general/session_id`.

ii.
```python
with h5py.File(path, 'r') as f:
    day = int(f['general/session_id'][()].decode())
    ...
return dict(..., info=info)
```

iii. The trajectory shows the agent surveying all NWB files and summarizing the result as 152 sessions, so the decision was one-file-per-session.

## 1-d. How are the data split into trials?

i. Trials are split by taking nonzero `trial_start` samples and nonzero `teleport` samples, then forming per-trial windows `[trial_start - 1, teleport - 1)`.

ii.
```python
trial_starts = np.nonzero(beh['trial_start/data'][:])[0]
teleports = np.nonzero(beh['teleport/data'][:])[0]

starts = trial_starts - 1
stops = teleports - 1
...
s, t = starts[i], stops[i]
```

iii. In the trajectory and module docstring, the agent justified this as matching `reward_relative.preprocessing.dff`, and said the teleport sample itself should be excluded because position is interpolated across the jump.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops two trial classes: the first trial of every session, and any trial where more than 30% of samples have `lick > 2`, which it interprets as a stuck lick sensor.

ii.
```python
LICK_ERR_FRAC = 0.3

lick_error[i] = np.mean(lick[s:t] > 2) > LICK_ERR_FRAC

for i in range(ntrials):
    if i == 0:
        continue
    if lick_error[i]:
        continue
```

iii. The trajectory says this was motivated by the paper’s lick-sensor QC and by the decoder input requirement that “previous trial outcome” is undefined on the first trial. The agent also noted that this flags exactly 81 trials, which it took as confirming the lick-sensor rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the raw `Fluorescence` and `Neuropil` signals, not from the NWB `Deconvolved` dataset.

ii.
```python
F_list.append(f['processing/ophys/Fluorescence'][plane]['data'][:, :].T[keep])
Fneu_list.append(f['processing/ophys/Neuropil'][plane]['data'][:, :].T[keep])
...
dff, events = compute_dff_and_events(F, Fneu, starts, stops, frame_rate)
```

iii. The trajectory repeatedly states that the stored NWB `Deconvolved` series is raw suite2p output and is not the paper’s analyzed signal; the agent justified recomputing the signal from `Fluorescence` and `Neuropil` for paper consistency.

## 2-b. How is the `neural` data processed?

i. The agent recomputes per-trial dF/F by subtracting `0.7 * Fneu`, adding back each trial’s mean neuropil, applying a maximin baseline (`gaussian -> minimum_filter1d -> maximum_filter1d`), converting to `(F - F0) / |F0|`, smoothing with a 2-sample Gaussian, and then deconvolving with `dcnv.oasis`.

ii.
```python
f = F[:, start:stop] - NEU_COEF * Fneu[:, start:stop]
f = f + NEU_COEF * np.nanmean(Fneu[:, start:stop], axis=1, keepdims=True)
flow = gaussian_filter1d(f, BASELINE_SIG, axis=-1)
flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
d = (f - flow) / np.abs(flow)
d = gaussian_filter1d(d, DFF_SIG, axis=-1)
events[:, start:stop] = dcnv.oasis(d, 2000, TAU, frame_rate)
```

iii. In the trajectory and final summary, the agent justified this as copying `reward_relative.preprocessing.dff` and the paper methods: same neuropil coefficient, same maximin baseline, same smoothing, and OASIS deconvolution with `tau = 0.7`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent first keeps only suite2p-curated ROIs (`iscell`), then removes putative interneurons whose dF/F is correlated with running speed above `0.5`.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
...
keep = iscell[rois]
...
speed_corr = (dff_c @ spd_c) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
events = events[keep_cells]
```

iii. The trajectory shows the agent reading the paper/code about manual ROI curation and `is_putative_interneuron`, then summarizing this as paper-consistent neuron curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data by slicing each trial on the same `[trial_start - 1, teleport - 1)` window; practically, the first included neural bin is the sample immediately before the stored `trial_start` index.

ii.
```python
starts = trial_starts - 1
stops = teleports - 1
...
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. The trajectory says the alignment event is trial start / lap start, but the agent justified using the `dff` window from the paper code even though that window is shifted by one sample.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame interval, about `64.484 ms` (`~15.5 Hz` per plane). No temporal rebinning is applied.

ii.
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
frame_rate = float(imaging_rate) / len(planes)
...
dt = float(np.median(np.diff(tstamps)))
...
'time_bin_size': float(round(list(dts)[0] * 1000, 4)),
```

iii. The trajectory notes that all behavioral and neural samples are already aligned at about `15.5 Hz`, and the final summary explicitly says the output uses the native imaging resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the median interval of the behavior timestamps (`position/timestamps`) plus the number of within-trial samples, rather than from each raw timestamp value directly.

ii.
```python
tstamps = beh['position/timestamps'][:]
...
dt = float(np.median(np.diff(tstamps)))
...
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. The trajectory emphasized that the dataset is sampled at a constant frame rate, so the agent treated a fixed `dt` grid as sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent creates a regularly spaced vector `0, dt, 2*dt, ...` with length equal to the number of trial bins.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
inp[0] = np.arange(T, dtype=np.float32) * dt
```

iii. The trajectory justification was implicit: because the behavior timestamps are uniformly sampled, subtracting the first timestamp is equivalent to constructing a uniform grid from `dt`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by assigning one time value per neural sample on the same per-trial slice `[s:t]`; time zero is the first included sample of that slice.

ii.
```python
T = t - s
inp[0] = np.arange(T, dtype=np.float32) * dt
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. The trajectory treated all per-trial variables as sharing the same imaging-frame grid, so the time vector was built directly to match the sliced neural matrix.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii.
```python
env = beh['environment/data'][:]
...
env_vals = np.unique(env[s:t])
envs[i] = int(env_vals[0])
```

iii. In the trajectory, the agent described this variable as the VR “environment” / morph value and noted that trials do not span multiple environment values.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent checks that each trial has exactly one unique environment value, stores that scalar, and then broadcasts it across all time bins in the trial input matrix.

ii.
```python
env_vals = np.unique(env[s:t])
assert len(env_vals) == 1, f'{path}: trial {i} spans environments {env_vals}'
envs[i] = int(env_vals[0])
...
inp[1] = envs[i]
```

iii. The trajectory says the environment variable is per-trial context, so the agent kept it constant within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the within-session trial loop index after trial boundaries have been determined from `trial_start` and `teleport`.

ii.
```python
for i in range(ntrials):
    ...
    inp[2] = i
```

iii. The trajectory framed this as the protocol trial index within each session rather than as a separate raw NWB variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No further processing is applied; the trial index `i` is written as a constant value across the full trial.

ii.
```python
inp[2] = i
```

iii. The trajectory justification was simply that trial number is a per-trial contextual variable, so it should be constant over the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the agent’s per-trial `rewarded` array, which itself is computed from `Reward/timestamps` and whether the `reward_zone` flag is active on that trial.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
...
in_zone = rzone[s:t] > 0
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
...
inp[3] = rewarded[i - 1]
```

iii. The trajectory says “Rewarded” means reward delivered in the zone on that trial, following `behavior.get_trial_types`, and the previous-trial input is then taken from that per-trial label.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The current trial receives the previous trial’s binary `rewarded` label. Instead of assigning a default value for trial 0, the agent drops the first trial of every session.

ii.
```python
for i in range(ntrials):
    if i == 0:
        continue
    ...
    inp[3] = rewarded[i - 1]
```

iii. In the trajectory, the agent explicitly justified dropping the first trial because “previous trial outcome” is undefined there.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal’s `position` and the inferred reward-zone identity for that trial. The trial’s zone is inferred from `reward_zone`-flagged positions, with missing trials filled by a single-switch heuristic.

ii.
```python
pos = beh['position/data'][:]
rzone = beh['reward_zone/data'][:]
...
if in_zone.any():
    observed_zone[i] = zone_label(pos[s:t][in_zone].min())
...
z0, z1 = REWARD_ZONES[zone_of_trial[i]]
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
```

iii. The trajectory says the reward zone should be read from the VR reward-zone flag and snapped to the protocol switch at trial 30 after checking all sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside the zone, positive after the zone.

ii.
```python
z0, z1 = REWARD_ZONES[zone_of_trial[i]]
dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))
out[0] = discretize_distance(dist)
```

iii. The trajectory justification was that this matches the decoder specification and the paper’s reward-relative framing.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is thresholded into 7 bins by explicit comparisons: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, `> 50`.

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

iii. The trajectory and docstring say these bins were chosen to match the decoder instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing `dist` from the same per-trial position slice `p = pos[s:t]` that is paired with `events[:, s:t]`.

ii.
```python
p = pos[s:t]
...
out[0] = discretize_distance(dist)
...
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. The trajectory consistently treated behavioral variables and neural data as sharing the same imaging-frame indices inside each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavioral `position` time series.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[s:t]
out[1] = np.digitize(p, POS_EDGES)
```

iii. The trajectory inspected `position` directly and treated it as the absolute VR track coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent takes the within-trial position slice and discretizes it with `np.digitize`.

ii.
```python
p = pos[s:t]
out[1] = np.digitize(p, POS_EDGES)
```

iii. The trajectory justification was that the decoder requires categorical outputs, so continuous position was binned.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 bins with cut points at `90`, `180`, `270`, and `360` cm.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POS_EDGES)
```

iii. The trajectory says this matches the instruction to divide the 450 cm track into five equal 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position` and `events` with the same `[s:t]` trial indices.

ii.
```python
p = pos[s:t]
...
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. The trajectory treated position as already sampled on the same aligned frame grid as the neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = beh['lick/data'][:]
...
lk = (lick[s:t] > 0).astype(np.int64)
```

iii. The trajectory inspected the raw lick values and treated them as cumulative/event-like lick measurements that needed binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent binarizes licking at each time bin by thresholding `lick > 0`.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
...
out[3] = lk
```

iii. The trajectory justification was that the decoder output is binary lick/no-lick, so positive raw lick values should be converted to `1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking `lick[s:t]` from the same trial slice used for `events[:, s:t]`.

ii.
```python
lk = (lick[s:t] > 0).astype(np.int64)
...
neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
```

iii. The trajectory assumes shared per-frame alignment among behavior and neural arrays after session loading.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`: the agent looks for samples with `reward_zone > 0`, converts the minimum in-zone position to one of three zone labels, and fills missing labels by assuming at most one switch.

ii.
```python
in_zone = rzone[s:t] > 0
if in_zone.any():
    observed_zone[i] = zone_label(pos[s:t][in_zone].min())
...
out[4] = zone_of_trial[i]
```

iii. The trajectory says the reward zone should come from the VR reward-zone flag and paper zone definitions A/B/C, with the switch anchored at trial 30 after whole-dataset checks.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The processing is: detect observed zone labels where reward-zone samples exist, assert at most one switch, use the protocol switch trial (`30`) if the observed change brackets it, otherwise fall back to the first observed post-switch trial, then assign one categorical zone label per trial.

ii.
```python
known = np.nonzero(~np.isnan(observed_zone))[0]
labels = observed_zone[known]
...
changes = np.nonzero(np.diff(labels))[0]
...
switch_trial = (SWITCH_TRIAL if last_before < SWITCH_TRIAL <= first_after
                else first_after)
zone_of_trial[:switch_trial] = int(labels[0])
zone_of_trial[switch_trial:] = int(labels[-1])
```

iii. The trajectory justifies this by saying the protocol moves the zone after 30 trials and that the heuristic was verified against all 152 sessions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the reward timestamps plus the presence of a reward-zone period on that trial, through the per-trial `rewarded` array.

ii.
```python
reward_t = beh['Reward/timestamps'][:]
...
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
...
out[5] = rewarded[i]
```

iii. The trajectory says reward outcome should mean rewarded vs omitted, with “rewarded” defined as reward delivered inside the zone on that trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the agent checks whether any reward timestamp falls within that trial’s timestamp interval and whether the trial contains reward-zone samples, then writes the resulting binary label across all time bins of the trial output.

ii.
```python
got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
rewarded[i] = int(in_zone.any() and got_reward)
...
out[5] = rewarded[i]
```

iii. The trajectory explicitly links this to `behavior.get_trial_types`, where rewarded trials are those with reward delivery in the reward zone.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles a few dataset issues explicitly: it truncates a one-frame neural/behavior mismatch in some multi-plane sessions, fills missing reward-zone labels by a single-switch heuristic, asserts that each trial has a single environment value, and drops trials with lick-sensor errors. It does not add a generic missing-data pathway beyond those rules.

ii.
```python
assert 0 <= F.shape[1] - nframes <= 1, (path, F.shape, nframes)
F = F[:, :nframes]
Fneu = Fneu[:, :nframes]
...
assert len(env_vals) == 1, f'{path}: trial {i} spans environments {env_vals}'
...
switch_trial = (SWITCH_TRIAL if last_before < SWITCH_TRIAL <= first_after
                else first_after)
...
if lick_error[i]:
    continue
```

iii. The trajectory shows the agent discovering the one-frame mismatch, the missing reward-zone trials, and the 81 lick-sensor-error trials during dataset-wide scans, then encoding targeted fixes rather than a general repair layer.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading each NWB file, recomputing dF/F and OASIS events for all cells, and writing the large pickle. The session conversion dominates, which is why the code parallelizes over sessions.

ii.
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops, frame_rate)
...
with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(load_session, files)):
        results.append(res)
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the trajectory, the agent benchmarked deconvolution work and then used a process pool with `spawn`, explicitly noting that conversion took minutes while the final pickle was about 9.5 GB.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are the per-trial loop in `compute_dff_and_events`, the per-trial loop that infers reward-zone / reward / lick-error summaries, and the per-trial loop that builds output arrays. Some pieces inside them are vectorized already, but the trial loop itself is still explicit.

ii.
```python
for start, stop in zip(starts, stops):
    ...

for i, (s, t) in enumerate(zip(starts, stops)):
    ...

for i in range(ntrials):
    ...
```

iii. The trajectory does not give a long explicit efficiency critique, but it does show the agent preferring session-level parallelism over trying to remove every per-trial loop.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code computes both `dff` and `events`, then uses `dff` again for interneuron filtering before discarding it. It also walks the trial boundaries multiple times: once for reward/environment/lick-error summaries and once again to assemble trial outputs.

ii.
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops, frame_rate)
...
speed_corr = (dff_c @ spd_c) / denom
...
for i, (s, t) in enumerate(zip(starts, stops)):
    ...
for i in range(ntrials):
    ...
```

iii. The trajectory shows the agent intentionally avoiding a separate survey pass, but the final code still repeats some work inside each session because dF/F is needed both for event extraction and for interneuron screening.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest extra work is building rich per-session metadata (`scene`, `date`, `trial_ids`, `rewarded_fraction`, per-plane counts) that the decoder itself does not use. The transient `dff` matrix is also discarded after interneuron filtering and deconvolution, although it is still needed as an intermediate.

ii.
```python
info = {
    'scene': scene,
    'date': date,
    'trial_indices': trial_ids,
    'rewarded_fraction': float(np.mean(rewarded)),
    ...
}
...
'session_info': [r['info'] for r in results],
```

iii. The trajectory indicates the agent wanted strong validation and inspection metadata, so it kept more session annotations than the downstream decoder strictly requires.
