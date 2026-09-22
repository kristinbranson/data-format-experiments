# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively discovers every NWB file one directory below `/app/data`, sorts them by numeric mouse and session identifiers, and loads each file directly with `h5py`. Sessions are converted in parallel with spawned worker processes.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
               key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))
with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
    for i, res in enumerate(ex.map(convert_session, files)):
        results.append(res)
```

iii. The trajectory says this covered all 152 NWB sessions and 11 mice. Direct HDF5 access was chosen for the large dataset, and spawn-based parallelism was used because suite2p's OASIS/Numba processing was found not to be fork-safe.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, then unique subject IDs are numerically sorted and each session receives an index into that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
'subject_idx': np.array([subjects.index(r['subject']) for r in results])
```

iii. The AI treated NWB subject metadata as authoritative and reported 11 mice, consistent with the paper and directory layout.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The session/day is read from `general/session_id`; one result entry becomes one element of each session-level output list.

ii.
```python
day = int(f['general/session_id'][()].decode())
return dict(neural=neural, input=inputs, output=outputs,
            subject=subject, region=region, info=info)
```

iii. The filename and NWB session metadata both encode a recording day, and the trajectory reports 152 converted sessions.

## 1-d. How are the data split into trials?

i. Trials begin at every sample where `trial_start == 1` and end at the corresponding sample where `teleport == 1`. Slices are start-inclusive and teleport-exclusive.

ii.
```python
starts = np.where(tstart == 1)[0]
stops = np.where(teleport == 1)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
...
for t, (s, e) in enumerate(zip(starts, stops)):
    neural.append(events[:, s:e])
```

iii. The AI identified a trial with one virtual-track lap and intentionally excluded the gray teleport/ITI, matching its reading of the paper.

## 1-e. How are trials filtered based on quality controls?

i. A trial is discarded when more than 30% of its samples have raw lick values greater than 2, interpreted as a stuck lick sensor. There is no minimum-duration filter.

ii.
```python
if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
    n_lick_error += 1
    continue
```

iii. The AI linked this criterion to the paper's licking-quality control and reported that it removed exactly 81 trials, the number reported by the paper. It dropped rather than masked them because lick is a required decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is recomputed from suite2p ROI fluorescence (`Fluorescence`) and neuropil fluorescence (`Neuropil`) for all imaging planes, using ROI references to apply `iscell`. The stored NWB `Deconvolved` signal is not used.

ii.
```python
Fs.append(f[f'processing/ophys/Fluorescence/{p}/data'][:].T)
Fneus.append(f[f'processing/ophys/Neuropil/{p}/data'][:].T)
roi_ids.append(f[f'processing/ophys/Fluorescence/{p}/rois'][:])
```

iii. The AI reasoned that the paper computes `events` through its own dF/F and OASIS pipeline, whereas NWB `Deconvolved` is suite2p's different deconvolution of raw fluorescence.

## 2-b. How is the `neural` data processed?

i. Per trial, it subtracts `0.7*Fneu`, adds back the trial mean neuropil, computes a maximin baseline after Gaussian smoothing (sigma 15; 300-sample minimum then maximum filters), forms dF/F, smooths it with sigma 2, and deconvolves it with OASIS (`tau=0.7`, per-plane sampling rate). Planes are concatenated. Unlike the reference, teleports are always excluded from baseline estimation.

ii.
```python
trial = f[:, s:e] + NEU_COEF * np.nanmean(fneu[:, s:e], axis=1, keepdims=True)
base = gaussian_filter1d(trial, BASELINE_SMOOTH, axis=1)
base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
d = gaussian_filter1d((trial - base) / np.abs(base), DFF_SMOOTH, axis=1)
events[:, s:e] = dcnv.oasis(np.ascontiguousarray(d, dtype=np.float32),
                            2000, TAU, fs)
```

iii. The parameters were taken from the paper and repository. The trajectory explicitly says per-trial baselines were used so blanked-laser teleport periods could not contaminate them; it did not account for the session-specific exceptions where imaging continued through teleport.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps only manually curated suite2p `iscell` ROIs and removes cells whose dF/F has Pearson correlation greater than 0.5 with running speed.

ii.
```python
keep = iscell[roi_ids]
F, Fneu = F[keep], Fneu[keep]
...
r = (d @ spc) / denom
good = ~(r > INTERNEURON_R)
events = events[good]
```

iii. Both filters were attributed to the Methods and paper code. The AI reported excluding 402 putative interneurons (0.29%), close to the paper's reported scale.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is implicit: every trial matrix starts at the `trial_start` sample (time zero) and ends before teleport; no resampling or shifting is done.

ii.
```python
for t, (s, e) in enumerate(zip(starts, stops)):
    neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The AI concluded that neural and behavior streams already share the imaging-frame clock, so slicing at trial start supplies the requested alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain at the native per-plane imaging rate, 15.5078125 Hz, or about 64.48 ms per bin. No rebinning or resampling is applied.

ii.
```python
fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
           .attrs['rate']) / len(planes)
...
'time_bin_size': 1000.0 / 15.5078125
```

iii. The trajectory states that dividing scanner rate by plane count yields the same neural sampling rate across recordings and preserves the paper's native resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the trial slice length and fluorescence sampling rate, rather than directly from raw timestamps.

ii.
```python
T = e - s
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The AI relied on the constant shared imaging clock; raw position timestamps are read for reward-event matching but not for this input.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based sample index is divided by the per-plane sampling frequency.

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. This makes the first trial sample exactly zero and advances in native 64.48 ms increments.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly `e-s` entries and is placed alongside `events[:, s:e]`, so columns correspond one-to-one.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
...
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
inputs.append(inp)
```

iii. The AI considered all streams pre-aligned on the same imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment is parsed from the session scene name stored in the NWB identifier, including the day-8 environment switch encoded in that name.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
m = re.fullmatch(r'Env(\d)_([ABC])_to_Env(\d)_([ABC])', scene)
```

iii. The AI followed the repository's scene-name task schedule and said it verified scene-derived task labels against reward-zone occupancy.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. One is subtracted from the scene's environment number to produce binary ENV1=0 and ENV2=1. For switch scenes, the first 30 trials use the first environment and later trials the second. The scalar is broadcast across every trial timepoint.

ii.
```python
envs = ([int(m.group(1)) - 1] * change_trial
        + [int(m.group(3)) - 1] * (ntrials - change_trial))
...
inp[1] = envs[t]
```

iii. This mirrors the known switch design and required binary encoding.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number comes from the zero-based loop index over start/teleport pairs, not the NWB `trial number` stream.

ii.
```python
for t, (s, e) in enumerate(zip(starts, stops)):
    inp[2] = t
```

iii. The agent treated the ordered detected laps as the reliable within-session trial sequence.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is performed beyond broadcasting the integer loop index over all samples of that trial. Removed trials retain their original indices, leaving gaps.

ii.
```python
inp[2] = t
kept_trials.append(t)
```

iii. The AI intended the value to describe the experimental trial number, rather than renumbering only retained trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the AI's per-trial `rewarded` array, itself based on raw `reward_zone` occupancy, raw `Reward/timestamps`, and position timestamps.

ii.
```python
in_zone = np.any(rzone[s:e] > 0)
got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
rewarded[t] = int(in_zone and got)
```

iii. The AI followed what it understood as the paper's rewarded-trial rule: delivery must occur while the active zone is represented.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Trials after the first receive the immediately preceding original trial's binary reward outcome. The first recorded trial is imputed as rewarded (1). The value is broadcast across time.

ii.
```python
prev = 1 if t == 0 else int(rewarded[t - 1])
inp[3] = prev
```

iii. The trajectory calls this a judgment: the unrecorded predecessor was among warm-up trials and about 85% of trials were rewarded, so 1 was viewed as the most probable imputation.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and the active reward-zone identity parsed from the NWB scene name; numerical zone bounds are constants from the paper/repository.

ii.
```python
REWARD_ZONES = {'A': (80., 130.), 'B': (200., 250.), 'C': (320., 370.)}
out[0] = bin_reward_distance(p, zones[t])
```

iii. The AI used the experimental schedule because it reported zero disagreements with observed reward-zone occupancy in all 10,394 trials having occupancy evidence.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the zone is measured relative to its start, position after it relative to its end, and every position inside the interval is assigned zero.

ii.
```python
d = np.zeros_like(pos)
d[pos < start] = pos[pos < start] - start
d[pos > stop] = pos[pos > stop] - stop
```

iii. This implements signed distance to the nearest point in the reward zone, the reward-relative quantity described in the task.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit categories implement `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50` cm.

ii.
```python
out = np.full(pos.shape, 3, dtype=np.int64)
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The boundaries were directly transcribed from the decoder instructions; initializing to class 3 preserves the full zero-distance zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural events are sliced with the same `[s:e]` indices; distance therefore has one value per neural column.

ii.
```python
p = pos[s:e]
out[0] = bin_reward_distance(p, zones[t])
neural.append(events[:, s:e])
```

iii. The streams were considered already aligned to the imaging clock.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[s:e]
```

iii. This is the NWB's virtual-corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position is divided by 90 cm, cast to integer, and clipped to classes 0–4.

ii.
```python
return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. Five equal bins over the 450 cm track are each 90 cm wide; clipping absorbs marginal out-of-range samples.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes are `<90`, `90–<180`, `180–<270`, `270–<360`, and `>=360` cm after clipping.

ii.
```python
TRACK_LENGTH = 450.
def bin_position(pos):
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The threshold spacing follows the requested five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and events use the identical trial slice.

ii.
```python
p = pos[s:e]
out[1] = bin_position(p)
neural.append(events[:, s:e])
```

iii. No further alignment is needed because both are on the imaging-frame clock.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the raw behavioral `lick/data` stream.

ii.
```python
lick = beh['lick/data'][:]
licks = lick[s:e]
```

iii. The agent interpreted positive values as detected licking and large sustained values as sensor errors.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Each retained sample is binarized as 1 when its lick value is positive and 0 otherwise; sensor-error trials are removed first.

ii.
```python
if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
    continue
out[3] = (licks > 0).astype(np.int64)
```

iii. Binarization is required by the task, and the quality filter was attributed to the paper.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural activity are sliced using the same start and end samples.

ii.
```python
licks = lick[s:e]
out[3] = (licks > 0).astype(np.int64)
neural.append(events[:, s:e])
```

iii. The AI relied on the behavior streams' existing interpolation to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB identifier's scene name, not directly from the raw reward-zone stream. The raw stream is used only to determine reward outcome and was used during validation.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
zones, envs = parse_scene(scene, ntrials)
```

iii. The scene name encodes the intended zone(s) and switch; the AI reported validating that schedule against occupancy with no mismatches where occupancy existed.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene patterns determine a fixed zone or a switch after trial 30. Labels A/B/C are mapped to 0/1/2 and broadcast across time.

ii.
```python
zones = [m.group(2)] * change_trial + [m.group(3)] * (ntrials - change_trial)
...
out[4] = ZONE_ORDER.index(zones[t])
```

iii. This mirrors `behavior.get_reward_zones` and the experiment's stated change trial.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses raw `Reward/timestamps`, position timestamps defining each trial's time interval, and the raw `reward_zone/data` occupancy signal.

ii.
```python
reward_times = beh['Reward/timestamps'][:]
stamps = beh['position/timestamps'][:]
rzone = beh['reward_zone/data'][:]
```

iii. The AI interpreted a valid rewarded trial as one with both a delivery event and active reward-zone occupancy.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, it checks for any positive zone-occupancy sample and any reward timestamp in `[stamps[s], stamps[e])`; their conjunction becomes a binary scalar broadcast through the trial.

ii.
```python
in_zone = np.any(rzone[s:e] > 0)
got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
rewarded[t] = int(in_zone and got)
...
out[5] = rewarded[t]
```

iii. The trajectory says this follows the repository's trial-type rule and produces 84.2% rewarded trials, consistent with the approximately 15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and position lengths are cropped to their overlap; trial stops are capped at that overlap; lick-sensor-error trials are dropped; assertions enforce paired, ordered trial boundaries and recognizable scene names. There is no interpolation, timestamp-consistency assertion, short-trial filter, or fallback for missing scene metadata.

ii.
```python
nframes = min(F.shape[1], len(pos))
F, Fneu = F[:, :nframes], Fneu[:, :nframes]
...
assert len(starts) == len(stops) and np.all(stops > starts)
stops = np.minimum(stops, nframes)
...
raise ValueError(f'unrecognized scene name: {scene}')
```

iii. The code comments identify occasional one-frame neural/behavior mismatches and choose the common overlap. The trajectory emphasizes empirical validation of the resulting full dataset but does not describe broader missing-data recovery.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large NWB fluorescence arrays, per-trial filtering/baseline computation, OASIS deconvolution, and writing the 9.6 GB pickle dominate. The AI parallelizes whole sessions.

ii.
```python
with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
    for i, res in enumerate(ex.map(convert_session, files)):
        results.append(res)
```

iii. The trajectory specifically notes the large dataset and OASIS/Numba behavior, motivating spawned multiprocessing; it reports the completed output size as 9.6 GB.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop copying trial windows, per-trial dF/F/OASIS loop, reward-outcome loop, and output construction loop could partly be vectorized or consolidated. The latter two traverse the same trial boundaries separately, although variable lengths and per-trial baselines make full vectorization awkward.

ii.
```python
for s, e in zip(starts, stops):
    f[:, s:e] = F[:, s:e]
...
for t, (s, e) in enumerate(zip(starts, stops)):
    rewarded[t] = int(in_zone and got)
...
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The AI did not explicitly discuss vectorization in its final rationale; its main efficiency choice was session-level multiprocessing. Per-trial OASIS and variable-length outputs naturally constrain vectorization.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries are traversed to copy valid samples, compute dF/F/events, build the inside-trial mask, compute rewards, and assemble outputs. Trial data are also sliced separately for each behavioral variable.

ii.
```python
for s, e in zip(starts, stops): ...
for s, e in zip(starts, stops): ...
for s, e in zip(starts, stops): inside[s:e] = True
for t, (s, e) in enumerate(zip(starts, stops)): ...
for t, (s, e) in enumerate(zip(starts, stops)): ...
```

iii. No explicit justification was given for these repeated passes; they keep each processing stage simple and operate on only tens of trials per session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full-session dF/F is retained temporarily only to calculate speed correlations, then discarded; neural processing is also performed for trials later rejected by the lick-sensor filter. `reward_amounts` are read nowhere, but several metadata/statistics values are computed solely for reporting.

ii.
```python
dff, events = compute_dff_events(F, Fneu, starts, stops, fs)
...
events = events[good]
...
if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
    continue
```

iii. The dF/F intermediate is necessary for the paper's interneuron filter even though only OASIS events are saved. The trajectory does not identify discarded processing; this assessment follows the data flow in the code.
