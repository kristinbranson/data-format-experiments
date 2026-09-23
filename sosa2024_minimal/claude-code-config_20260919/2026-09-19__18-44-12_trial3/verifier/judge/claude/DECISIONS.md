# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `sub-*/*.nwb` under the data directory, sorted by subject number and session number. Each file is loaded using `h5py` (not `pynwb`). All NWB files are processed -- one per session.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
               key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))
```
```python
with h5py.File(path, 'r') as f:
    scene = f['identifier'][()].decode().split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    ...
```

iii. The agent explored the data directory structure early on and found all NWB files following the `sub-<id>/sub-<id>_ses-<N>_behavior+ophys.nwb` naming pattern. It used h5py from its very first data exploration and never switched to pynwb.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected across all sessions and sorted by numeric ID.

ii.
```python
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
```

iii. The subject ID is extracted directly from the NWB metadata rather than parsed from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session/experiment day is extracted from `general/session_id` inside the NWB file.

ii.
```python
day = int(f['general/session_id'][()].decode())
```

iii. The agent verified that each NWB file represents a single imaging session on a particular day.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined by `trial_start` (==1) for the start and `teleport` (==1) for the end. The teleport frame is exclusive (the ITI/teleport period is not included in the trial).

ii.
```python
starts = np.where(tstart == 1)[0]
stops = np.where(teleport == 1)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
```

iii. The agent inspected the behavior time series and determined that `trial_start` and `teleport` define the lap boundaries. It verified that the number of starts equals the number of stops and that stops always come after starts.

## 1-e. How are trials filtered based on quality controls?

i. Trials with a stuck lick sensor are dropped: if >30% of samples have cumulative lick count > 2, the trial is discarded entirely. 81 trials were dropped this way. No minimum trial length filter is applied.

ii.
```python
LICK_ERROR_FRAC = 0.3
LICK_ERROR_COUNT = 2

if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
    n_lick_error += 1
    continue
```

iii. The agent read the reference code's `correct_lick_sensor_error()` function and adapted it. It chose to drop trials rather than NaN-ing lick values because lick is a decoder output. The 81 dropped trials matched the paper's reported number.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB files. The NWB's `Deconvolved` field is NOT used, as it is suite2p's own deconvolution, not the paper's pipeline.

ii.
```python
Fs.append(f[f'processing/ophys/Fluorescence/{p}/data'][:].T)
Fneus.append(f[f'processing/ophys/Neuropil/{p}/data'][:].T)
```

iii. The agent verified that the NWB `Deconvolved` array has only ~0.41 correlation with the paper's pipeline output, confirming the need to recompute from raw F and Fneu.

## 2-b. How is the `neural` data processed?

i. dF/F is computed following the paper's `preprocessing.dff` pipeline: (1) mask non-trial frames as NaN, (2) subtract 0.7 * Fneu, (3) per-trial, add trial-mean neuropil back, (4) maximin baseline (Gaussian smooth sigma=15, then 300-sample min filter, then 300-sample max filter), (5) dF/F = (F - baseline) / |baseline|, (6) smooth with 2-sample Gaussian, (7) deconvolve with OASIS (tau=0.7, per-plane frame rate). The baseline computation only uses within-trial data (ITI excluded).

ii.
```python
def compute_dff_events(F, Fneu, starts, stops, fs):
    f = np.full(F.shape, np.nan, dtype=np.float64)
    fneu = np.full(F.shape, np.nan, dtype=np.float64)
    for s, e in zip(starts, stops):
        f[:, s:e] = F[:, s:e]
        fneu[:, s:e] = Fneu[:, s:e]
    f -= NEU_COEF * fneu
    ...
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
```

iii. The agent read the paper's `preprocessing.dff` and replicated all steps. Parameters match the paper: neu_coef=0.7, baseline_method='maximin', tau=0.7. The 2000 batch size for OASIS is from the original code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) only suite2p manually curated cells (`iscell[:,0] == True`) are kept, (2) putative interneurons are dropped (cells with Pearson r > 0.5 between dF/F and running speed). The interneuron filter uses only in-trial frames.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
keep = iscell[roi_ids]
F, Fneu = F[keep], Fneu[keep]
...
d = dff[:, inside]
sp = speed[:nframes][inside]
d = d - d.mean(axis=1, keepdims=True)
spc = sp - sp.mean()
denom = np.sqrt((d ** 2).sum(axis=1) * (spc ** 2).sum())
with np.errstate(invalid='ignore', divide='ignore'):
    r = (d @ spc) / denom
good = ~(r > INTERNEURON_R)
events = events[good]
```

iii. The agent found the interneuron threshold (0.5) from the paper's `dayData.py` and the `is_putative_interneuron` function. 402 cells total were excluded across all sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start, which requires no additional processing -- the neural data is simply sliced from trial start to teleport.

ii.
```python
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. Since the alignment event is trial start, and the trial data begins at trial start by definition, no temporal shifting is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native imaging frame rate (~15.5 Hz, time bin ~64.5 ms). For multi-plane mice (m17, m18), the per-plane rate is used (scanner rate / n_planes). No temporal rebinning is applied.

ii.
```python
fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
           .attrs['rate']) / len(planes)
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. The agent verified that the stored rate attribute is the scanner rate, which must be divided by the number of planes for multi-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame count within the trial and the per-plane imaging frame rate (fs). Does NOT use behavior timestamps.

ii.
```python
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The agent computed time from frame indices divided by the per-plane frame rate, rather than using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Frame indices (0, 1, 2, ..., T-1) are divided by the per-plane frame rate to get seconds. Time starts at 0 for each trial.

ii.
```python
T = e - s
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. Simple division of frame index by sampling rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices in the NWB files, so no alignment is needed. They are sliced with the same start/stop indices.

ii.
```python
p = pos[s:e]
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
inp[0] = np.arange(T, dtype=np.float32) / fs
```

iii. The agent verified that behavior and imaging share the same time base in the NWB files.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the scene name stored in the NWB `identifier` field, parsed using regex to extract the environment number (Env1 or Env2).

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
def parse_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    m = re.fullmatch(r'Env(\d)_Location([ABC])', scene)
    if m:
        env = int(m.group(1)) - 1
        return [m.group(2)] * ntrials, [env] * ntrials
```

iii. The agent found that the scene name encodes the environment and reward zone. It verified zero mismatches between parsed environment and stored behavior data across all sessions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment number is extracted from the scene name (Env1 -> 0, Env2 -> 1). For switch sessions that change environment (e.g., `Env1_B_to_Env2_C`), the environment changes after trial 30.

ii.
```python
m = re.fullmatch(r'Env(\d)_([ABC])_to_Env(\d)_([ABC])', scene)
if m:
    envs = ([int(m.group(1)) - 1] * change_trial
            + [int(m.group(3)) - 1] * (ntrials - change_trial))
```

iii. This follows the paper's `behavior.get_trial_types` logic, which the agent read from the reference code.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the sequential index (0, 1, 2, ...) within a session, derived from the loop counter over trials.

ii.
```python
for t, (s, e) in enumerate(zip(starts, stops)):
    ...
    inp[2] = t
```

iii. The agent uses the loop counter `t` which enumerates trials within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing -- the sequential trial index is directly assigned as a constant across all timepoints in the trial.

ii.
```python
inp[2] = t
```

iii. Simple assignment of the loop index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and `reward_zone` behavior time series. A trial is considered rewarded if both a reward timestamp falls within the trial AND the reward zone signal is active.

ii.
```python
reward_times = beh['Reward/timestamps'][:]
...
for t, (s, e) in enumerate(zip(starts, stops)):
    in_zone = np.any(rzone[s:e] > 0)
    got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
    rewarded[t] = int(in_zone and got)
```

iii. The agent followed the paper's `behavior.get_trial_types` which defines reward as requiring both zone entry and reward delivery.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward status is used. For the first trial of a session (t==0), the value defaults to 1 (rewarded), since ~85% of trials are rewarded and there are warm-up trials before imaging.

ii.
```python
prev = 1 if t == 0 else int(rewarded[t - 1])
inp[3] = prev
```

iii. The agent noted that the first imaging trial is preceded by ~30 warm-up trials in the same condition, and ~85% of trials are rewarded, justifying a default of 1.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone identity for the current trial. Reward zone identity comes from the scene name parsed by `parse_scene()`.

ii.
```python
p = pos[s:e]
out[0] = bin_reward_distance(p, zones[t])
```

iii. The agent determined the reward zone from the session's scene name, verified against the reward_zone occupancy signal with zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from position to the nearest edge of the reward zone. If inside the zone, distance = 0. If before the zone, distance is negative. If past, positive.

ii.
```python
def bin_reward_distance(pos, zone):
    start, stop = REWARD_ZONES[zone]
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
```

iii. This follows the reward zone boundaries from the paper (A: 80-130, B: 200-250, C: 320-370 cm).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 categories using explicit boolean indexing: 0 (<-50), 1 (-50 to -10), 2 (-10 to 0), 3 (in zone, d==0), 4 (0 to +10), 5 (+10 to +50), 6 (>+50).

ii.
```python
out = np.full(pos.shape, 3, dtype=np.int64)     # 3: inside the zone (d == 0)
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. Matches the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as neural data within each trial -- no additional alignment needed.

ii.
```python
p = pos[s:e]
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. Both neural and behavioral data use the same time indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[s:e]
out[1] = bin_position(p)
```

iii. Direct use of the stored position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond discretization into 5 equal 90-cm bins via division by 90 and flooring/clipping.

ii.
```python
def bin_position(pos):
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. The 450 cm track divided by 5 gives 90 cm per bin.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position / 90 is cast to int and clipped to [0, 4], giving bins: 0 (<90), 1 (90-180), 2 (180-270), 3 (270-360), 4 (>=360).

ii.
```python
return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)
```

iii. Matches the instruction specification of 5 equal bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data -- no additional alignment needed.

ii. Same indexing: `p = pos[s:e]`

iii. Both use the same time base.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh['lick/data'][:]
...
licks = lick[s:e]
out[3] = (licks > 0).astype(np.int64)
```

iii. Direct use of the stored lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (licks > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data -- no additional alignment needed.

ii. Same indexing: `licks = lick[s:e]`

iii. Both use the same time base.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session's scene name (NWB `identifier` field), parsed by `parse_scene()` to determine per-trial reward zone (A, B, or C). On switch sessions, the zone changes after trial 30.

ii.
```python
zones, envs = parse_scene(scene, ntrials)
...
out[4] = ZONE_ORDER.index(zones[t])
```

iii. The agent verified this against the `reward_zone` occupancy signal with zero mismatches across all sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A, B, C) is mapped to an integer (0, 1, 2) using `ZONE_ORDER.index()`. The value is constant across all timepoints within a trial.

ii.
```python
ZONE_ORDER = ['A', 'B', 'C']
out[4] = ZONE_ORDER.index(zones[t])
```

iii. Simple mapping from zone name to index.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from two sources: `Reward` timestamps and the `reward_zone` behavior time series. A trial is rewarded only if both a reward event timestamp falls within the trial AND the reward zone signal is active during the trial.

ii.
```python
reward_times = beh['Reward/timestamps'][:]
...
in_zone = np.any(rzone[s:e] > 0)
got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
rewarded[t] = int(in_zone and got)
```

iii. This dual-condition check follows `behavior.get_trial_types` in the reference code, matching the ~15% omission rate described in the paper.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if the trial had both an active reward zone and a reward delivery event, 0 otherwise. Constant across all timepoints in the trial.

ii.
```python
rewarded[t] = int(in_zone and got)
...
out[5] = rewarded[t]
```

iii. The resulting ~84.2% reward rate matches expectations from the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Imaging/behavior frame count mismatch**: Truncated to the minimum of the two (`nframes = min(F.shape[1], len(pos))`).
- **Stuck lick sensor**: Trials with >30% frames having lick count >2 are dropped entirely (81 trials).
- **Multi-plane mice (m17, m18)**: Planes are pooled; frame rate is divided by number of planes.
- **m11 missing days 1-2**: These days were never imaged, so no files exist and no handling is needed.

ii.
```python
nframes = min(F.shape[1], len(pos))
F, Fneu = F[:, :nframes], Fneu[:, :nframes]
stops = np.minimum(stops, nframes)
```

iii. The agent noted 10 sessions with 1-frame differences, all in multi-plane mice.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** (reading large arrays from disk)
2. **dF/F computation and OASIS deconvolution** (per-plane, per-trial signal processing)
3. **Interneuron filtering** (correlation computation across all cells)

ii. N/A

iii. The agent used multiprocessing (12 workers) to parallelize session conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over trials sequentially, but the discretization functions (`bin_reward_distance`, `bin_position`, `bin_speed`) are already vectorized within each trial. The reward computation loop could potentially be vectorized across trials.

ii. N/A

iii. The variable trial lengths make full vectorization awkward, and the multiprocessing approach handles the main bottleneck (per-session I/O and processing).

## 13-c. What processing does the code repeat multiple times?

i. Unlike the reference solution (which has a separate survey step), the AI's code processes each NWB file only once. There is no repeated loading or processing.

ii. N/A

iii. The AI's approach is more efficient in this regard, processing everything in a single pass per session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `dff` (delta F/F) array is computed but only the `events` (deconvolved) array is used as neural data. The dff is needed intermediately to compute interneuron filtering and then for deconvolution, so it is not entirely unnecessary, but the full dff array is not saved.

ii.
```python
dff, events = compute_dff_events(F, Fneu, starts, stops, fs)
# dff used only for interneuron filtering, then discarded
```

iii. The dff is a necessary intermediate for computing events and interneuron correlations.
