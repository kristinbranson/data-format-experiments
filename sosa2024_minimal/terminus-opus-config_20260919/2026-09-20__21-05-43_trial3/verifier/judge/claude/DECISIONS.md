# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `/app/data` directory using `h5py` (not `pynwb`). It uses `glob.glob` to find all `.nwb` files across all `sub-*` subdirectories, then processes each file with `h5py.File(fn, 'r')` reading data directly via HDF5 paths. Each NWB file corresponds to one session for one subject.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with h5py.File(fn, 'r') as f:
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    # ... reads F, Fneu, behavior data via HDF5 paths
```

iii. The agent chose h5py over pynwb explicitly for speed: "Let me inspect the NWB structure of a small file using h5py (fast, avoids loading data)." The agent verified 152 sessions across 11 subjects were found, matching the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file, then sorted numerically by the numeric portion of the subject ID (e.g., m3, m4, ..., m19).

ii.
```python
subjects = sorted({r[3]['subject'] for r in results},
                  key=lambda s: int(s[1:]))
data['subject_idx'] = np.array([subjects.index(r[3]['subject']) for r in results])
```

iii. Subject IDs are read directly from each NWB file's metadata rather than from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from `general/session_id` in the NWB file.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# Each file = one session
session_id = f['general/session_id'][()].decode()
```

iii. The agent found 152 NWB files total, each representing a single recording session.

## 1-d. How are the data split into trials?

i. Trials are defined as laps from the `trial_start` frame to the `teleport` frame. The `trial_start` indices are found by `np.where(trial_start_data > 0)` and `teleport` indices by `np.where(teleport_data > 0)`. The teleport/ITI period is excluded.

ii.
```python
tstart = np.where(f[B + 'trial_start/data'][:] > 0)[0]
teleport = np.where(f[B + 'teleport/data'][:] > 0)[0]
# ...
npairs = min(len(tstart), len(teleport))
tstart, teleport = tstart[:npairs], teleport[:npairs]
```

iii. The agent stated: "Trials = laps from trial_start to teleport (ITI/teleport excluded, as the paper's dF/F is computed on-track only)."

## 1-e. How are trials filtered based on quality controls?

i. Four filtering criteria are applied:
1. **First trial dropped**: The loop starts at `i=1`, dropping the first trial of each session because the previous trial outcome is undefined.
2. **Lick sensor errors**: Trials where >30% of frames have a cumulative lick count > 2 are dropped.
3. **Environment filter**: Trials where the environment is not 0 or 1 are excluded.
4. **Minimum trial length**: Trials with fewer than 2 frames are excluded.
5. **Finite check**: Trials with non-finite neural data are excluded.

ii.
```python
for i in range(1, ntrials):  # skip first trial
    if lick_error[i]:
        continue
    if env_trial[i] not in (0, 1):
        continue
    if T < 2:
        continue
    if not np.all(np.isfinite(ev)):
        continue
```
```python
lick_error[i] = (np.sum(seg > 2) / len(seg)) > LICK_ERROR_FRAC  # LICK_ERROR_FRAC = 0.3
```

iii. The agent verified: "The lick-sensor-error criterion (>30% of frames with cumulative lick>2) yields 81 bad trials out of 12,216, exactly matching the paper (81 trials)." The first trial is dropped because "the previous trial's outcome (a decoder input) is unknown for it."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) traces stored in the NWB file under `processing/ophys/Fluorescence` and `processing/ophys/Neuropil`. The NWB's pre-computed `Deconvolved` field is NOT used.

ii.
```python
F = np.concatenate([f['processing/ophys/Fluorescence/' + p + '/data'][:].T
                    for p in planes], axis=0)
Fneu = np.concatenate([f['processing/ophys/Neuropil/' + p + '/data'][:].T
                       for p in planes], axis=0)
```

iii. The agent noted: "NWB has raw suite2p F, Fneu, deconvolved (on raw F), iscell" and chose to recompute dF/F and deconvolution from F and Fneu to match the paper's pipeline.

## 2-b. How is the `neural` data processed?

i. The dF/F and deconvolution pipeline is computed per trial: subtract 0.7*Fneu, add trial-mean neuropil back, compute maximin baseline (Gaussian smooth sigma=15 frames, then 300-frame minimum and maximum filters), compute dF/F as (F-baseline)/|baseline|, smooth with Gaussian sigma=2 frames, then deconvolve with OASIS (tau=0.7, fs=rate/nplanes). Cells from multiple planes are concatenated.

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

iii. The agent described: "dF/F: neuropil subtraction (0.7xFneu, trial-mean neuropil added back), per-trial maximin baseline (Gaussian sd 15 frames -> 300-frame ~20 s min then max filter), dF/F = (F-F0)/|F0|, smoothed with 2-frame s.d. Gaussian. Neural signal = OASIS-deconvolved events."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only suite2p-curated ROIs (`iscell`) are kept. (2) Putative interneurons (dF/F-speed Pearson r > 0.5) are excluded.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0
F = F[iscell].astype(np.float32)
Fneu = Fneu[iscell].astype(np.float32)
# ...
r = (dsub @ spc) / denom
is_int = np.nan_to_num(r, nan=0.0) > INT_R_THRESH  # 0.5
keep = ~is_int
events = events[keep]
```

iii. The agent cited the paper's Methods for the r > 0.5 threshold and confirmed: "interneuron exclusion is often 0, consistent with the paper's 0.42 +/- 0.85% mean."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by extracting frames from `tstart[i]` to `teleport[i]`. Since the trial start frame is index 0, alignment is implicit.

ii.
```python
ev = events[:, s:e]  # s = tstart[i], e = teleport[i]
neural_trials.append(ev.astype(np.float32))
```

iii. Trial start alignment requires no additional processing since trial boundaries already define the start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The data is kept at the native per-plane sampling rate. The time bin size is computed as `1000.0 / 15.5078125` ms (approximately 64.5 ms), derived from the scanner rate divided by the number of planes.

ii.
```python
'time_bin_size': 1000.0 / (15.5078125),
```

iii. The agent determined the sampling rate from the NWB metadata: `rate / nplanes` gives ~15.5 Hz per plane.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from trial start is derived from the frame count within the trial and the per-plane sampling rate `fs`. It does NOT use the behavior timestamps.

ii.
```python
t_in_trial = np.arange(T, dtype=np.float32) / fs
```

iii. The agent computed time as frame index divided by sampling rate, giving evenly-spaced time values.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple arange of frame indices divided by the per-plane sampling rate. Each frame is assigned a time value starting from 0.

ii.
```python
t_in_trial = np.arange(T, dtype=np.float32) / fs
```
where `T = e - s` (number of frames in the trial) and `fs = rate / nplanes`.

iii. This produces evenly-spaced time values starting at 0 for each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time values are computed directly from the neural frame indices, so they are inherently aligned.

ii.
```python
T = e - s  # same frame count as neural data
t_in_trial = np.arange(T, dtype=np.float32) / fs
```

iii. Since both neural and time use the same frame count `T`, alignment is automatic.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = f[B + 'environment/data'][:]
# ...
env_trial[i] = int(np.round(np.median(env[s:e])))
```

iii. The environment variable records the VR environment type at each timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of the environment values within the trial is computed and rounded to get a single per-trial value (0 or 1). This is then broadcast to all timepoints.

ii.
```python
env_trial[i] = int(np.round(np.median(env[s:e])))
# ...
np.full(T, env_trial[i], dtype=np.float32),
```

iii. The median is used presumably to handle any edge effects at trial boundaries.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index from the loop counter, not from any raw data variable.

ii.
```python
np.full(T, i, dtype=np.float32),  # i is the trial loop index
```

iii. The loop variable `i` ranges from 1 to ntrials-1 (since the first trial is skipped).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — the trial loop index `i` is used directly and broadcast to all timepoints.

ii.
```python
np.full(T, i, dtype=np.float32),
```

iii. The trial number starts at 1 (since the first trial is dropped) and increments by 1 for each trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` flag in the behavior data. A trial is considered rewarded only if a reward timestamp falls within the trial AND the reward zone flag was triggered.

ii.
```python
reward_ts = f[B + 'Reward/timestamps'][:]
# ...
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. The agent combined reward timestamps with reward zone flags to determine if a trial was truly rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the previous trial (`rewarded[i-1]`) is used. The first trial is dropped entirely so there is always a valid previous trial.

ii.
```python
np.full(T, rewarded[i - 1], dtype=np.float32),
```

iii. The agent dropped the first trial specifically so that `rewarded[i-1]` is always defined.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location (from the scene name). The reward zone boundaries come from hardcoded constants: A=(80,130), B=(200,250), C=(320,370).

ii.
```python
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
# ...
p = pos[s:e]
zone = REWARD_ZONES[zone_lab[i]]
dist = signed_distance_to_zone(p, zone)
```

iii. Reward zones are determined from the session scene name (e.g., `Env1_LocationB_to_A`), switching after trial 30 on switch sessions. The agent verified this against the reward_zone flag with "0/10,394 mismatches."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before the zone, 0 inside it, positive past it. The distance is then discretized into 7 bins.

ii.
```python
def signed_distance_to_zone(pos, zone):
    start, stop = zone
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
    return d
```

iii. This follows the standard signed distance to an interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic rather than `np.digitize`.

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
    return out
```

iii. The bin boundaries match the instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data use the same frame indices within each trial, so alignment is implicit.

ii.
```python
p = pos[s:e]  # same s, e as neural data
```

iii. Both neural and behavioral data share the same time indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = f[B + 'position/data'][:]
# ...
p = pos[s:e]
```

iii. The position variable records the animal's position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing into 5 equal bins.

ii.
```python
def bin_position(pos):
    edges = np.array([90.0, 180.0, 270.0, 360.0])
    return np.digitize(pos, edges).astype(np.int16)
```

iii. The 450 cm track is divided into five 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized using `np.digitize` with edges [90, 180, 270, 360], producing bins 0-4 corresponding to <90, 90-180, 180-270, 270-360, >360 cm.

ii.
```python
edges = np.array([90.0, 180.0, 270.0, 360.0])
return np.digitize(pos, edges).astype(np.int16)
```

iii. Matches the instructions' 5 equal-sized bins over 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data — no additional alignment needed.

ii.
```python
p = pos[s:e]  # same s, e as neural data
```

iii. Implicit alignment via shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = f[B + 'lick/data'][:]
# ...
lk = (lick[s:e] > 0).astype(np.int16)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int16)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data — no additional alignment needed.

ii.
```python
lk = (lick[s:e] > 0).astype(np.int16)  # same s, e as neural data
```

iii. Implicit alignment via shared frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session scene name (extracted from the NWB `identifier` field), not from the `reward_zone` behavior time series directly.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
# ...
def zone_labels_for_session(scene, ntrials):
    if '_to_' in scene:
        pre, post = scene.split('_to_')
        z0, z1 = pre[-1], post[-1]
        labels = [z0] * min(SWITCH_TRIAL, ntrials) + [z1] * max(0, ntrials - SWITCH_TRIAL)
    else:
        labels = [scene[-1]] * ntrials
    return labels
```

iii. The agent verified that the scene-based rule matched the reward_zone flag data with 0/10,394 mismatches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine the reward zone letter (A, B, or C). For switch sessions (containing `_to_`), the zone switches from the first to the second letter after trial 30 (`SWITCH_TRIAL`). The letter is mapped to an index: A=0, B=1, C=2.

ii.
```python
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}
SWITCH_TRIAL = 30
# ...
np.full(T, ZONE_IDX[zone_lab[i]], dtype=np.int16),
```

iii. The agent confirmed the switch-at-30 rule from the Methods section.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` flag. A trial is rewarded only if a reward timestamp falls within the trial time window AND the reward zone flag was triggered during the trial.

ii.
```python
got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
in_zone = np.any(rzone_flag[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. The agent added the `in_zone` condition to distinguish true reward deliveries from other reward-like events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if the trial was rewarded (reward timestamp present AND reward zone flag active), 0 otherwise. The value is broadcast to all timepoints in the trial.

ii.
```python
rewarded[i] = int(got_reward and in_zone)
# ...
np.full(T, rewarded[i], dtype=np.int16),
```

iii. The dual condition ensures only genuine reward events are counted.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is truncated to the minimum of neural frames and behavior samples.
- **Trial start/teleport count mismatch**: Truncated to the minimum count.
- **Teleport indices beyond data**: Filtered out with `teleport < nframes`.
- **Lick sensor errors**: Trials with >30% of frames having lick > 2 are dropped.
- **Non-finite neural data**: Trials with non-finite events are excluded.

ii.
```python
nframes = min(F.shape[1], len(pos))
F = F[:, :nframes]; Fneu = Fneu[:, :nframes]
# ...
npairs = min(len(tstart), len(teleport))
keep_tr = teleport < nframes
# ...
if not np.all(np.isfinite(ev)):
    continue
```

iii. The agent noted: "a few dual-plane sessions have one more imaging frame than VR samples."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files**: Reading large F and Fneu arrays from disk.
2. **dF/F computation and deconvolution**: Per-trial signal processing.
3. **Saving the pickle file** (~9.4 GB).

The agent mitigated this with `ProcessPoolExecutor(max_workers=12)` for parallel session processing.

ii.
```python
ctx = mp.get_context('spawn')
with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as ex:
    for fn, res in zip(files, ex.map(process_session, files)):
        results.append(res)
```

iii. The agent reported "Per-session processing works and is fast (0.4-1.4 s)" with 152 sessions processed in parallel.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial dF/F loop processes each trial sequentially. The interneuron correlation computation is already vectorized (matrix multiply `dsub @ spc`). The per-trial binning could theoretically be done on the whole session array before splitting, but variable trial lengths make this straightforward.

ii. N/A

iii. The code is already relatively well optimized, with the main bottleneck being I/O.

## 13-c. What processing does the code repeat multiple times?

i. Each session is processed independently and only once. There is no survey step — the code processes everything in a single pass, unlike some approaches that might scan data twice (once for statistics, once for conversion).

ii. N/A

iii. The single-pass design is more efficient than a two-pass approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `dff` array is computed for the full session but only used to compute the interneuron correlation and the deconvolved events. The dF/F values themselves are not stored in the output. Additionally, `speed` is decoded as an output but was not requested in the instructions (speed is an output in the code but the reference only lists it in decoder outputs).

ii.
```python
dff = np.full(F.shape, np.nan, dtype=np.float32)
# dff is used for interneuron filtering and deconvolution, but not saved
```

iii. The dF/F computation is necessary for the pipeline even though only events are stored.
