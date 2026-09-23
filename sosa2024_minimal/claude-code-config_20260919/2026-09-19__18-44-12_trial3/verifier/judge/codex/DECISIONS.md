# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `.nwb` file under `/app/data/sub-*/*.nwb`, sorts them by mouse and session number, and converts each file as one session. Within each file it reads the needed behavior and ophys datasets directly from the NWB/HDF5 structure with `h5py`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
               key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))

with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
    for i, res in enumerate(ex.map(convert_session, files)):
        results.append(res)
```

```python
def convert_session(path):
    with h5py.File(path, 'r') as f:
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        ...
        beh = f['processing/behavior/BehavioralTimeSeries']
        ...
        planes = sorted(f['processing/ophys/Fluorescence'].keys())
```

iii. In the trajectory, the agent first listed `/app/data`, counted NWB files, inspected file structure, then surveyed all NWBs (`steps 3, 9, 11, 44`). In its final summary it explicitly said it converted “all 152 NWB sessions.”

## 1-b. How are the data split into subjects?

i. Subjects are identified per session from the NWB `subject_id` field, then deduplicated into a sorted `subjects` list. `subject_idx` maps each converted session back to that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
...
'subjects': subjects,
'subject_idx': np.array([subjects.index(r['subject']) for r in results],
                        dtype=np.int64),
```

iii. The trajectory shows the agent inspected the released file layout and the NWB metadata fields before settling on this organization. The final summary also reports the dataset as 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session ordering is the sorted file order, and the session/day metadata are read from the NWB `session_id`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
               key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))
```

```python
day = int(f['general/session_id'][()].decode())
...
return dict(neural=neural, input=inputs, output=outputs,
            subject=subject, region=region, info=info)
```

iii. The trajectory includes explicit file-counting and survey steps over the NWB files and a final statement that all 152 sessions were converted.

## 1-d. How are the data split into trials?

i. Trials are defined as one lap from the `trial_start` sample, inclusive, to the `teleport` sample, exclusive. Trial arrays are sliced with those start/stop indices.

ii.
```python
starts = np.where(tstart == 1)[0]
stops = np.where(teleport == 1)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
stops = np.minimum(stops, nframes)
```

```python
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. In the trajectory, the agent inspected trial structure directly (`step 48`) and in its final summary described trials as “`trial_start` (inclusive) to `teleport` (exclusive).”

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials that look like stuck lick-sensor trials: if more than 30% of samples in the trial have lick values greater than 2, the whole trial is skipped. It does not apply a separate minimum-trial-length filter in `convert_session`.

ii.
```python
LICK_ERROR_FRAC = 0.3
LICK_ERROR_COUNT = 2
...
for t, (s, e) in enumerate(zip(starts, stops)):
    licks = lick[s:e]
    if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
        n_lick_error += 1
        continue
```

iii. The trajectory shows the agent reading the paper code’s lick-error logic (`step 50`), validating zone/env/lick-error statistics across all files (`step 52`), and then saying in its final summary that it dropped 81 stuck-sensor trials because lick is a decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the raw `Fluorescence` and `Neuropil` ROI traces, not from the NWB `Deconvolved` dataset.

ii.
```python
planes = sorted(f['processing/ophys/Fluorescence'].keys())
Fs, Fneus, roi_ids = [], [], []
for p in planes:
    Fs.append(f[f'processing/ophys/Fluorescence/{p}/data'][:].T)
    Fneus.append(f[f'processing/ophys/Neuropil/{p}/data'][:].T)
    roi_ids.append(f[f'processing/ophys/Fluorescence/{p}/rois'][:])
```

iii. The trajectory includes an explicit inspection of `Fluorescence`, `Neuropil`, and `Deconvolved` in an NWB file (`step 20`) and a read of the paper’s preprocessing code (`steps 16, 18`). The final summary states that stored `Deconvolved` was not used.

## 2-b. How is the `neural` data processed?

i. The agent recomputes dF/F and events itself. Outside-trial samples are masked out. Within each trial it subtracts `0.7 * Fneu`, adds the per-trial neuropil mean back, computes a maximin baseline using Gaussian smoothing then min/max filters, converts to dF/F, smooths again, and deconvolves with OASIS using `tau = 0.7`. It always excludes teleport/ITI samples from the baseline windows.

ii.
```python
def compute_dff_events(F, Fneu, starts, stops, fs):
    f = np.full(F.shape, np.nan, dtype=np.float64)
    fneu = np.full(F.shape, np.nan, dtype=np.float64)
    for s, e in zip(starts, stops):
        f[:, s:e] = F[:, s:e]
        fneu[:, s:e] = Fneu[:, s:e]

    f -= NEU_COEF * fneu

    dff = np.zeros(F.shape, dtype=np.float64)
    events = np.zeros(F.shape, dtype=np.float32)
    for s, e in zip(starts, stops):
        trial = f[:, s:e] + NEU_COEF * np.nanmean(fneu[:, s:e], axis=1, keepdims=True)
        base = gaussian_filter1d(trial, BASELINE_SMOOTH, axis=1)
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        d = (trial - base) / np.abs(base)
        d = gaussian_filter1d(d, DFF_SMOOTH, axis=1)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(d, dtype=np.float32),
                                    2000, TAU, fs)
    return dff, events
```

iii. The trajectory shows the agent reading `preprocessing.py` and `utilities.py`, testing the dF/F pipeline on one session (`steps 16, 18, 56, 58`), and then stating in its final summary that it recomputed neural activity from raw `F/Fneu` “exactly as in `reward_relative.preprocessing.dff`,” with teleport periods excluded from the baseline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only suite2p ROIs labeled `iscell`, then removes putative interneurons whose dF/F is correlated with running speed above `r > 0.5`.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:, 0].astype(bool)
...
keep = iscell[roi_ids]
F, Fneu = F[keep], Fneu[keep]
```

```python
d = dff[:, inside]
sp = speed[:nframes][inside]
...
with np.errstate(invalid='ignore', divide='ignore'):
    r = (d @ spc) / denom
good = ~(r > INTERNEURON_R)
events = events[good]
```

iii. The trajectory shows the agent inspecting multi-plane ROI ordering (`steps 46, 91`), reading the paper’s interneuron filter (`steps 70, 71`), and summarizing the same two filters in the final message.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by cutting each trial from `trial_start` to `teleport`. No further offset is applied.

ii.
```python
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The trajectory’s final summary explicitly says the temporal alignment event is trial start and that the trial window is the lap itself.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native frame rate and does not rebin. It uses the NWB sampling rate, divided by the number of simultaneously imaged planes, and records a nominal time bin size of `1000.0 / 15.5078125` ms.

ii.
```python
fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
           .attrs['rate']) / len(planes)
```

```python
inp[0] = np.arange(T, dtype=np.float32) / fs
...
'metadata': {
    ...
    'time_bin_size': 1000.0 / 15.5078125,
    ...
    'sampling_rate_hz': 15.5078125,
}
```

iii. The trajectory includes explicit inspection of imaging rates and multi-plane sessions (`steps 20, 46`) and the final summary says the dataset stays at the native imaging rate with 64.48 ms bins.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The agent derives this variable from the trial length and the inferred imaging-frame sampling rate `fs`, rather than directly from the raw behavior timestamps.

ii.
```python
stamps = beh['position/timestamps'][:]
...
fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
           .attrs['rate']) / len(planes)
...
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The trajectory shows the agent inspected timestamps and found the behavior streams already interpolated to the imaging clock (`steps 20, 46`). In the final summary it described the behavior streams as already on the imaging frame clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial the agent creates a regularly spaced vector starting at 0 and increasing by `1/fs` seconds for each sample.

ii.
```python
T = e - s
...
inp = np.empty((4, T), dtype=np.float32)
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The trajectory does not contain a separate written justification for this exact choice beyond the general claim that the behavior signals were already on the imaging frame clock.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: both the neural matrix and the time vector use the same trial length `T = e - s` from the same `[s:e)` slice.

ii.
```python
T = e - s
...
inp[0] = np.arange(T, dtype=np.float32) / fs
...
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The agent’s exploration focused on confirming common frame-clock alignment between behavior and imaging streams (`steps 20, 46`), and its final summary said the behavior streams were already interpolated to the imaging frame clock.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the session `identifier` string (`scene`) rather than from the per-frame `environment` behavior series.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
def parse_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.fullmatch(r'Env(\d)_Location([ABC])', scene)
    ...
    m = re.fullmatch(r'Env(\d)_([ABC])_to_Env(\d)_([ABC])', scene)
```

```python
zones, envs = parse_scene(scene, ntrials)
...
inp[1] = envs[t]
```

iii. The trajectory shows the agent reading `behavior.py` trial-type functions (`steps 24, 25`), validating scene-derived zones and environments against the behavior data (`step 52`), and then stating in the final summary that reward zone and environment came from the scene name.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses the scene name with regexes and, for environment-switch sessions, assigns one environment for the first 30 trials and the new environment thereafter.

ii.
```python
def parse_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    ...
    m = re.fullmatch(r'Env(\d)_([ABC])_to_Env(\d)_([ABC])', scene)
    if m:
        zones = [m.group(2)] * change_trial + [m.group(4)] * (ntrials - change_trial)
        envs = ([int(m.group(1)) - 1] * change_trial
                + [int(m.group(3)) - 1] * (ntrials - change_trial))
        return zones[:ntrials], envs[:ntrials]
```

iii. The trajectory justification is that this mirrors the paper’s own scene-based task-description logic and was validated against the reward-zone occupancy and environment streams (`step 52` and final summary).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `t` over the trial start/stop pairs.

ii.
```python
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = t
```

iii. The trajectory does not give a separate justification for this field, but the code uses the per-session trial order produced by the trial segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond setting the scalar trial index and broadcasting it across all time points in the trial.

ii.
```python
inp = np.empty((4, T), dtype=np.float32)
...
inp[2] = t
```

iii. No additional justification appears in the trajectory beyond the general trial-by-trial organization of the dataset.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The agent derives previous trial outcome from a session-level `rewarded` array. That array is itself built from `Reward/timestamps`, `position/timestamps`, and whether the `reward_zone` signal was active inside each trial.

ii.
```python
rzone = beh['reward_zone/data'][:]
stamps = beh['position/timestamps'][:]
reward_times = beh['Reward/timestamps'][:]
...
rewarded = np.zeros(ntrials, dtype=np.int64)
for t, (s, e) in enumerate(zip(starts, stops)):
    in_zone = np.any(rzone[s:e] > 0)
    got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
    rewarded[t] = int(in_zone and got)
```

iii. The trajectory shows the agent validating reward-zone and reward statistics (`steps 48, 52`) and in the final summary it justified reward outcome as “reward delivered and zone active,” following the paper’s rule.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `t > 0`, the value is `rewarded[t - 1]`. For the first imaged trial of each session, the agent sets the value to `1` on the assumption that the unseen warm-up predecessor was probably rewarded.

ii.
```python
# previous trial outcome; on the first imaged trial of a session the
# preceding (warm-up) trial is not in the file, so it is set to rewarded,
# the outcome of ~85% of trials
prev = 1 if t == 0 else int(rewarded[t - 1])
...
inp[3] = prev
```

iii. The final summary contains the full justification: the imaging session is preceded by about 30 warm-up trials in the same condition and about 85% of trials are rewarded, so the agent filled the first previous-outcome value with `1`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the `position` behavior series and the current trial’s reward-zone label parsed from the session scene name.

ii.
```python
pos = beh['position/data'][:]
...
zones, envs = parse_scene(scene, ntrials)
...
p = pos[s:e]
out[0] = bin_reward_distance(p, zones[t])
```

```python
def bin_reward_distance(pos, zone):
    start, stop = REWARD_ZONES[zone]
```

iii. The trajectory shows the agent reading the paper’s behavior code (`steps 24, 25`), validating scene-derived zone labels against observed reward-zone occupancy (`step 52`), and reporting “0 mismatches” in the final summary.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes a signed distance to the nearest reward-zone boundary: negative before the zone, zero inside the zone, positive after the zone.

ii.
```python
def bin_reward_distance(pos, zone):
    start, stop = REWARD_ZONES[zone]
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
    out = np.full(pos.shape, 3, dtype=np.int64)
```

iii. The trajectory justification is indirect: the final summary says the task uses the paper’s reward-zone definitions and that this variable is one of the decoded outputs.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The agent converts the signed distances into seven classes using explicit comparisons that correspond to the requested bins.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The trajectory does not separately discuss the bin thresholds; this appears to be a direct implementation of the task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same trial slice `[s:e)` as the neural activity for each trial.

ii.
```python
for t, (s, e) in enumerate(zip(starts, stops)):
    p = pos[s:e]
    ...
    out[0] = bin_reward_distance(p, zones[t])
    ...
    neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The agent’s exploration established that the behavior streams were already sampled on the imaging frame clock, so it treated common indexing as sufficient alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the `position` behavior series.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[s:e]
out[1] = bin_position(p)
```

iii. The trajectory includes direct inspection of the behavior variables in NWB (`step 20`) and task-structure inspection (`step 48`).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent bins position by dividing by 90 cm (450 cm track / 5 bins), converting to integers, and clipping to `[0, 4]`.

ii.
```python
def bin_position(pos):
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

```python
out[1] = bin_position(p)
```

iii. The trajectory does not provide a separate narrative justification beyond following the decoder-task binning.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five bins: effectively `<90`, `90-180`, `180-270`, `270-360`, and `>360` cm, via the `bin_position` helper.

ii.
```python
def bin_position(pos):
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The binning matches the instruction-defined five equal bins over a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same `[s:e)` trial slice as the neural data.

ii.
```python
p = pos[s:e]
...
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The trajectory treats shared imaging-clock indexing as sufficient alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the `lick` behavior series.

ii.
```python
lick = beh['lick/data'][:]
...
licks = lick[s:e]
```

iii. The agent inspected the raw lick values during NWB exploration (`steps 20, 48`) and later read the paper’s lick-error handling code (`step 50`).

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick values are binarized: values greater than 0 become 1, otherwise 0.

ii.
```python
out[3] = (licks > 0).astype(np.int64)
```

iii. The trajectory does not separately justify this beyond the decoder instruction that lick should be binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same trial slice `[s:e)` as the neural matrix.

ii.
```python
licks = lick[s:e]
...
out[3] = (licks > 0).astype(np.int64)
...
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The trajectory justification is again the common imaging-clock indexing of the behavior streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the session `identifier`/scene string via `parse_scene`, not from the `reward_zone` occupancy trace itself.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
zones, envs = parse_scene(scene, ntrials)
...
out[4] = ZONE_ORDER.index(zones[t])
```

iii. The trajectory shows the agent reading the paper’s behavior functions and then validating the scene-derived zone labels against the observed reward-zone occupancy (`steps 24, 25, 52`).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed to yield a per-trial zone label, with a switch after trial 30 on switch sessions, and then mapped to integers `A -> 0`, `B -> 1`, `C -> 2`.

ii.
```python
def parse_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    ...
    m = re.fullmatch(r'Env(\d)_Location([ABC])_to_([ABC])', scene)
    if m:
        zones = [m.group(2)] * change_trial + [m.group(3)] * (ntrials - change_trial)
        return zones[:ntrials], [env] * ntrials
```

```python
out[4] = ZONE_ORDER.index(zones[t])
```

iii. The final summary says this mirrors `reward_relative.behavior.get_reward_zones` and was verified against the occupancy signal.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward-event timestamps (`Reward/timestamps`) together with whether the reward-zone occupancy signal was active within that trial.

ii.
```python
rzone = beh['reward_zone/data'][:]
stamps = beh['position/timestamps'][:]
reward_times = beh['Reward/timestamps'][:]
...
in_zone = np.any(rzone[s:e] > 0)
got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
rewarded[t] = int(in_zone and got)
```

iii. The trajectory shows the agent examining reward timing and trial structure (`step 48`) and validating reward statistics across the full dataset (`step 52`). The final summary says it used the paper’s “reward delivered and zone active” rule.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial the agent checks whether the trial contains both a reward event and any reward-zone occupancy, stores that as `rewarded[t]`, and then writes the same scalar value across all time points in the trial.

ii.
```python
rewarded = np.zeros(ntrials, dtype=np.int64)
for t, (s, e) in enumerate(zip(starts, stops)):
    in_zone = np.any(rzone[s:e] > 0)
    got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
    rewarded[t] = int(in_zone and got)
...
out[5] = rewarded[t]
```

iii. The agent justified this in the final summary as matching the paper’s trial-type logic and reported an overall rewarded fraction consistent with the paper’s ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles a few issues defensively: it crops imaging to the overlap with behavior if their lengths differ, caps trial stops at `nframes`, asserts that trial starts and stops match, and discards trials flagged as stuck lick-sensor trials. It does not include the reference solution’s explicit short-trial filter or timestamp-alignment assertion for reward events.

ii.
```python
nframes = min(F.shape[1], len(pos))
F, Fneu = F[:, :nframes], Fneu[:, :nframes]
...
assert len(starts) == len(stops) and np.all(stops > starts)
stops = np.minimum(stops, nframes)
```

```python
if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
    n_lick_error += 1
    continue
```

iii. The trajectory shows the agent surveying the dataset for mismatches and lick errors (`steps 44, 52`) and then documenting the lick-error trial drop in the final summary.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work in the agent’s code is per-session NWB I/O plus the dF/F and OASIS deconvolution, which is why the code parallelizes conversion across sessions with a spawn-based process pool.

ii.
```python
with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
    for i, res in enumerate(ex.map(convert_session, files)):
        results.append(res)
```

```python
dff, events = compute_dff_events(F, Fneu, starts, stops, fs)
```

iii. The trajectory includes explicit tests of the dF/F pipeline (`step 58`) and multiprocessing behavior (`steps 66, 68`), and the final summary reports a full 9.6 GB converted dataset.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent leaves several trial-by-trial loops in place that could have been further vectorized: copying fluorescence into trial windows, marking in-trial samples, computing per-trial reward flags, and constructing per-trial input/output arrays.

ii.
```python
for s, e in zip(starts, stops):
    f[:, s:e] = F[:, s:e]
    fneu[:, s:e] = Fneu[:, s:e]
```

```python
for s, e in zip(starts, stops):
    inside[s:e] = True
...
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
    rewarded[t] = int(in_zone and got)
...
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
    inputs.append(inp)
    outputs.append(out)
```

iii. The trajectory does not explicitly discuss vectorization tradeoffs for these loops; this is mostly evident from the final implementation.

## 13-c. What processing does the code repeat multiple times?

i. The code walks the same trial boundaries multiple times: once to mask/copy trial frames for dF/F, once to build the `inside` mask and per-trial reward labels, and once more to emit the final per-trial neural/input/output arrays.

ii.
```python
for s, e in zip(starts, stops):
    f[:, s:e] = F[:, s:e]
    fneu[:, s:e] = Fneu[:, s:e]
```

```python
for s, e in zip(starts, stops):
    inside[s:e] = True
...
for t, (s, e) in enumerate(zip(starts, stops)):
    rewarded[t] = int(in_zone and got)
...
for t, (s, e) in enumerate(zip(starts, stops)):
    neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. No explicit trajectory justification was given for this repetition; it is a direct consequence of how the session conversion code is structured.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main discarded computation is the full `dff` array: it is computed so the speed-correlation interneuron filter can run, but only the deconvolved `events` are saved into the output dataset. The code also stores session `info` metadata that the decoder itself does not consume.

ii.
```python
dff, events = compute_dff_events(F, Fneu, starts, stops, fs)
...
d = dff[:, inside]
...
events = events[good]
...
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

```python
info = {
    'subject': subject,
    ...
    'trial_indices': kept_trials,
}
```

iii. The trajectory does not explicitly call this out, but its final summary emphasizes that the saved neural signal is the deconvolved activity, not dF/F.
