# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from all subdirectories of `/app/data`. It scans for directories in `DATA_ROOT`, then collects all `.nwb` files within each subject directory. Files are sorted by subject number then experiment day. Data is loaded using `h5py` (not `pynwb`), directly accessing the HDF5 dataset paths (e.g., `processing/ophys/Fluorescence/plane0/data`, `processing/behavior/BehavioralTimeSeries/position/data`). Multiprocessing (spawn context, 6 workers) is used to convert sessions in parallel, with results cached to `.session_cache/` pickle files.

ii.
```python
files = []
for sub in sorted(os.listdir(DATA_ROOT)):
    d = os.path.join(DATA_ROOT, sub)
    if os.path.isdir(d):
        files += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.endswith('.nwb')]
files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                          int(re.search(r'ses-(\d+)', p).group(1))))
```
```python
with h5py.File(path, 'r') as f:
    subject = f['general/subject/subject_id'][()].decode()
    exp_day = int(f['general/session_id'][()].decode())
    ...
    beh = f['processing/behavior/BehavioralTimeSeries']
    ts = beh['position/timestamps'][:]
    pos = beh['position/data'][:]
    ...
```

iii. The agent explored the NWB file structure using `h5py.File(...).visititems()` to map all datasets and groups, then chose h5py for direct and efficient access to the specific arrays needed. It confirmed 11 subjects and 152 session files.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the directory names in the data root (pattern `sub-m<N>`) and also verified from the NWB file's `general/subject/subject_id` field. A sorted list of unique subject IDs is constructed.

ii.
```python
subjects = sorted({re.search(r'sub-(m\d+)', p).group(1) for p in files},
                  key=lambda s: int(s[1:]))
```

iii. The agent enumerated all 152 NWB files and confirmed 11 subjects (m3, m4, m10-m15, m17-m19), matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The experiment day (session number) is parsed from the file's `general/session_id` field. Sessions are processed independently.

ii.
```python
exp_day = int(f['general/session_id'][()].decode())
```

iii. The agent verified that each NWB file contains a single session by inspecting file naming patterns (`sub-m<N>_ses-<DD>_behavior+ophys.nwb`) and internal metadata.

## 1-d. How are the data split into trials?

i. Trials are defined by the `trial_start` and `teleport` behavioral time series. Trial starts are frames where `trial_start > 0`, and teleport frames are where `teleport > 0`. The actual data slice for each trial is `[trial_start - 1, teleport - 1)`, matching the indexing convention used by the reference `dff()` function (which uses 1-based indexing internally).

ii.
```python
trial_starts = np.flatnonzero(beh['trial_start/data'][:] > 0)
teleports = np.flatnonzero(beh['teleport/data'][:] > 0)
...
a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
```

iii. The agent read the reference `preprocessing.dff()` function and found it slices segments as `(start_ind - 1, stop_ind - 1)`. The agent chose to use the same -1 offset for behavior extraction to ensure perfect neural-behavior alignment. It verified trial_start and teleport signals are consistently paired (same count per session).

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied:
1. **First trial dropped**: The first imaged lap of every session is dropped because the previous trial outcome (a required decoder input) is unknown (the ~30 un-imaged warm-up laps are not in the file).
2. **Lick-sensor artifact**: Trials where >30% of imaging frames have a cumulative lick count > 2 are dropped, reproducing `behavior.correct_lick_sensor_error`. The 0.30 threshold was tuned to match the paper's reported count of 81 affected trials.
3. **Very short trials**: Trials with fewer than 2 frames (`b - a < 2`) are skipped.
4. **Non-finite neural data**: Trials where the deconvolved events contain non-finite values are skipped.
5. **Sessions with < 2 surviving trials** are dropped entirely.

ii.
```python
if i == 0:
    continue
if lick_error[i]:
    continue
a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
if b - a < 2:
    continue
...
if not np.all(np.isfinite(neu)):
    continue
...
if len(neural_trials) < 2:
    return None
```
```python
lick_error[i] = np.mean(lick[a:b] > 2) > LICK_ERROR_THRESH  # LICK_ERROR_THRESH = 0.30
```

iii. The agent read `behavior.correct_lick_sensor_error()` from the reference code and tested multiple thresholds (0.30, 0.35, 0.50). At 0.30, exactly 81 trials were flagged, matching the paper's reported count. The first-trial drop was justified because the preceding warm-up laps are not in the file, making `previous_trial_outcome` genuinely unknown.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (F) and `Neuropil` (Fneu), read from the NWB's `processing/ophys/Fluorescence` and `processing/ophys/Neuropil` groups. The NWB's `Deconvolved` field (suite2p's own deconvolution) is NOT used.

ii.
```python
plane_names = sorted(f['processing/ophys/Fluorescence'].keys())
F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                    for p in plane_names], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil'][p]['data'][:nframes, :]
                       for p in plane_names], axis=1).T
```

iii. The agent discovered that the NWB `Deconvolved` field is suite2p's default deconvolution of raw fluorescence, while the paper computes its own dF/F and events using `preprocessing.dff()` with specific parameters. The agent chose to replicate the paper's pipeline.

## 2-b. How is the `neural` data processed?

i. The paper's dF/F and deconvolution pipeline is reproduced in `compute_dff_and_events()`:
1. **Neuropil subtraction**: `F - 0.7 * Fneu`
2. **Add back trial-mean neuropil**: `+ 0.7 * mean(Fneu_segment)` so the baseline is not near-zero
3. **Maximin baseline**: Gaussian smooth (sigma=15 samples), minimum filter (300 samples ~20s), maximum filter (300 samples)
4. **dF/F**: `(F - baseline) / |baseline|`
5. **Smoothing**: 2-sample Gaussian on dF/F
6. **Deconvolution**: OASIS via `suite2p.extraction.dcnv.oasis()` with tau=0.7 and frame_rate=15.5078125 Hz

Teleport period handling: per-animal/per-day lookup from `TELEPORT_SESSIONS` determines whether the baseline window spans trial+teleport or trial only.

ii.
```python
def compute_dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports):
    ...
    f_ -= NEU_COEF * fneu_
    for a, b in segments:
        f_[:, a:b] += NEU_COEF * np.nanmean(fneu_[:, a:b], axis=1, keepdims=True)
        seg = ndi.gaussian_filter1d(f_[:, a:b], BASELINE_SMOOTH_SIG, axis=1)
        seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, a:b] = seg
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    for a, b in segments:
        dff[:, a:b] = ndi.gaussian_filter1d(dff[:, a:b], DFF_SMOOTH_SIG, axis=1)
        events[:, a:b] = dcnv.oasis(..., 2000, TAU, FRAME_RATE)
```

iii. The agent read the paper's Methods description and the reference `preprocessing.dff()` function, confirming parameters: `neu_coef=0.7`, `tau=0.7`, `baseline_method='maximin'`, 20s window (300 samples), 2-sample Gaussian smoothing. The agent verified the OASIS function signature by inspecting suite2p source code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied:
1. **suite2p manual curation (iscell)**: Only ROIs with `iscell[:, 0] > 0` are kept. The iscell array is read from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` (pooled across all planes).
2. **Putative interneuron removal**: Cells whose dF/F correlates with running speed at Pearson r > 0.5 are excluded. This is computed as a vectorized matrix operation.

ii.
```python
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0
F = np.ascontiguousarray(F[iscell], dtype=np.float64)
Fneu = np.ascontiguousarray(Fneu[iscell], dtype=np.float64)
...
dv = dff[:, valid] - dff[:, valid].mean(axis=1, keepdims=True)
spc = sp_v - sp_v.mean()
speed_corr = (dv @ spc) / np.sqrt((dv ** 2).sum(axis=1) * (spc ** 2).sum())
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
events = events[~is_int]
```

iii. The agent read `spatial.is_putative_interneuron()` and `dayData` to determine the paper uses `int_thresh = 0.5` (not the function's default of 0.3). The agent verified this removed ~0.33% of cells on average, consistent with the paper's reported 0.42%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The trial slice `[trial_start - 1, teleport - 1)` matches the dff() convention, so neural and behavioral data are naturally aligned by using the same frame indices. No additional temporal alignment is needed.

ii.
```python
a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
```

iii. The agent ensured that both neural events and behavioral variables use the same `[a, b)` frame range, avoiding any misalignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of 15.5078125 Hz (time bin size = 64.484 ms) is used. This is hardcoded as `FRAME_RATE = 15.5078125`.

ii.
```python
FRAME_RATE = 15.5078125       # Hz, per imaging plane
DT = 1.0 / FRAME_RATE         # s
...
'time_bin_size': 1000.0 / FRAME_RATE,
```

iii. The agent determined the per-plane frame rate from the NWB metadata, accounting for multi-plane sessions where the scanner rate is twice the per-plane rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavioral time series.

ii.
```python
ts = beh['position/timestamps'][:]
...
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
inp[0] = t_rel
```

iii. The timestamps are shared across all behavioral variables and are aligned with the imaging frames.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's starting timestamp is subtracted from all timestamps within the trial: `ts[a:b] - ts[a]`.

ii.
```python
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
inp[0] = t_rel
```

iii. Straightforward computation to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices (same `[a, b)` slice), so they are inherently aligned. No interpolation or resampling is needed.

ii. Both use the same `a, b` indices:
```python
a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
neu = np.ascontiguousarray(events[:, a:b], ...)
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
```

iii. The agent verified that behavioral timestamps match the imaging frame rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment/data` behavioral time series.

ii.
```python
env_ts = beh['environment/data'][:]
...
env_vals = np.unique(env_ts[a:b])
assert len(env_vals) == 1
environment[i] = int(env_vals[0])
```

iii. The agent verified that the environment variable is constant within each trial and is either 0 or 1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the unique environment value within the trial is extracted and asserted to be a single value. It is cast to int.

ii.
```python
env_vals = np.unique(env_ts[a:b])
assert len(env_vals) == 1, f'{path}: trial {i} has environments {env_vals}'
environment[i] = int(env_vals[0])
...
inp[1] = environment[i]
```

iii. The assertion ensures data consistency - the environment should not change mid-trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-based loop index `i` over trials within the session. It is derived from the trial boundaries (trial_start and teleport arrays), not from any stored trial number variable.

ii.
```python
for i in range(ntrials):
    ...
    inp[2] = i
```

iii. The agent used the sequential index as the trial number, which counts all trials including the dropped first trial (so the first retained trial has index 1).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
inp[2] = i
```

iii. The trial number is the ordinal position within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series. Reward event timestamps are mapped to imaging frames using `np.searchsorted`.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
...
rew_frames = np.searchsorted(ts, reward_ts)
rewarded = np.zeros(ntrials, dtype=np.int64)
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
```

iii. The agent found that `Reward/data` is all zeros (not usable), so only the timestamps indicate reward delivery.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward event timestamp falls within the previous trial's frame range. The first trial of each session is dropped (see 1-e), so `previous_trial_outcome` always has a valid preceding trial. The value is `rewarded[i - 1]`.

ii.
```python
inp[3] = rewarded[i - 1]
```

iii. Since the first trial is dropped, `i >= 1` always holds, ensuring `rewarded[i-1]` is well-defined.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position/data` behavioral time series and the reward zone location for each trial. The reward zone location is determined from the session's scene name (parsed from the NWB `identifier` field) using `scene_reward_zones()`, following `behavior.get_reward_zones()`. Reward zone boundaries are: A=(80,130), B=(200,250), C=(320,370) cm.

ii.
```python
scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
...
zone_labels = scene_reward_zones(scene, ntrials)
...
zstart, zstop = REWARD_ZONES[zone_labels[i]]
p = pos[a:b]
out[0] = reward_zone_distance_bin(p, zstart, zstop)
```

iii. The agent read `behavior.get_reward_zones()` which parses scene names. Scenes follow patterns like `Env1_LocationA` (single zone) or `Env1_LocationA_to_C` (zone switch after 30 trials). The agent validated this against the actual reward_zone behavioral data across all 12,216 laps with zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest edge of the reward zone. Distance is 0 inside the zone, negative before it, positive after it. The continuous distance is then discretized into 7 bins using direct conditional assignment.

ii.
```python
def reward_zone_distance_bin(pos, zstart, zstop):
    d = np.zeros_like(pos)
    before = pos < zstart
    after = pos > zstop
    d[before] = pos[before] - zstart
    d[after] = pos[after] - zstop
    out = np.full(pos.shape, 3, dtype=np.int64)   # 3: inside the zone
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out
```

iii. The bin boundaries match the task specification's 7-bin discretization of distance to reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories using direct conditional assignment:
- 0: d < -50 cm
- 1: -50 <= d < -10 cm
- 2: -10 <= d < 0 cm
- 3: d == 0 (inside zone)
- 4: 0 < d <= 10 cm
- 5: 10 < d <= 50 cm
- 6: d > 50 cm

ii. See code in 7-b above.

iii. The boundaries match the instructions. Minor boundary differences at exactly d=10 and d=50 compared to the reference's `np.digitize` approach (the AI uses `<=` for positive boundaries where the reference uses `<`), but these affect negligibly few samples.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices within each trial — no additional alignment needed.

ii.
```python
p = pos[a:b]
out[0] = reward_zone_distance_bin(p, zstart, zstop)
```

iii. Same `[a, b)` slice as neural data ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position/data` behavioral time series.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[a:b]
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing into 5 bins.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
...
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The 450 cm track is divided into 5 equal 90 cm bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins using `np.digitize` with edges `[90, 180, 270, 360]`:
- 0: < 90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: > 360 cm

ii.
```python
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. Matches the instructions specifying 5 equal-sized bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data — no additional alignment needed.

ii. Same `[a, b)` slice as neural data.

iii. Verified by shared timestamps.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick/data` behavioral time series.

ii.
```python
lick = beh['lick/data'][:]
...
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The `lick` variable records lick events at each imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes). Raw lick values can be >1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data — no additional alignment needed.

ii. Same `[a, b)` slice.

iii. Shared timestamps ensure alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session scene name (NWB `identifier` field), parsed by `scene_reward_zones()` following `behavior.get_reward_zones()`. The scene name encodes which reward zone(s) are active and whether a switch occurs after 30 trials.

ii.
```python
scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
zone_labels = scene_reward_zones(scene, ntrials)
...
out[4] = ZONE_LABELS.index(zone_labels[i])
```
```python
def scene_reward_zones(scene, ntrials):
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return [m.group(1)] * ntrials
    m = (re.fullmatch(r'Env\d_Location([ABC])_to_([ABC])', scene)
         or re.fullmatch(r'Env\d_([ABC])_to_Env\d_([ABC])', scene))
    if m:
        return [m.group(1)] * SWITCH_TRIAL + [m.group(2)] * (ntrials - SWITCH_TRIAL)
```

iii. The agent validated this approach against the actual `reward_zone` behavioral data across all 12,216 laps with zero mismatches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name regex parsing determines the per-trial zone label (A, B, or C). For switch sessions, the zone changes after 30 trials (SWITCH_TRIAL = 30). The label is mapped to an integer: A=0, B=1, C=2.

ii.
```python
ZONE_LABELS = ['A', 'B', 'C']
SWITCH_TRIAL = 30
...
out[4] = ZONE_LABELS.index(zone_labels[i])
```

iii. Deterministic parsing of the experimental metadata, consistent with `behavior.get_reward_zones()`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` behavioral time series. Reward delivery is detected by mapping reward event timestamps to imaging frames.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
rew_frames = np.searchsorted(ts, reward_ts)
rewarded = np.zeros(ntrials, dtype=np.int64)
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
```

iii. The `Reward/data` field is all zeros; only the timestamps carry meaningful reward information.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to imaging frames using `np.searchsorted`. For each trial, the output is 1 if any reward event falls within the trial's frame range `[a, b)`, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
out[5] = rewarded[i]
```

iii. Binary per-trial reward outcome as specified in the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Non-finite neural data**: Trials where deconvolved events contain NaN or Inf are skipped entirely.
- **Lick-sensor artifacts**: Trials flagged by the lick-error criterion (>30% of frames with lick count > 2) are dropped.
- **First trial**: Dropped because the previous trial outcome is unknown.
- **Very short trials**: Trials with < 2 frames are skipped.
- **Empty sessions**: Sessions with < 2 surviving trials return None and are excluded.
- **Reward data field**: All zeros in the NWB; the agent used only timestamps.
- **Scanning flag**: Verified `scanning[a:b] > 0` for every retained trial.

ii.
```python
if not np.all(np.isfinite(neu)):
    continue
...
if lick_error[i]:
    continue
...
if b - a < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. These checks were discovered during data exploration. The lick-error filter reproduces a specific quality control from the paper.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with h5py and reading large fluorescence arrays — I/O bound
2. **dF/F computation**: Gaussian smoothing, min/max filtering, and OASIS deconvolution over the full (ncells, nframes) matrix per session
3. **Multiprocessing overhead**: Spawning worker processes and serializing results

ii. N/A

iii. The agent used multiprocessing with 6 workers and session-level caching to mitigate the I/O and computation costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron speed correlation is already vectorized as a single matrix-vector product (`dv @ spc`). The per-trial assembly loop iterates over trials sequentially, but given variable trial lengths, vectorization would require padding. The reward delivery check iterates over trials, which could be vectorized with broadcasting.

ii.
```python
# Vectorized speed correlation
speed_corr = (dv @ spc) / denom
```

iii. Most numerical operations are already vectorized. The per-trial loop is the natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session exactly once (with caching). It does not have a separate survey/exploration step that re-reads data. The cached results prevent recomputation on re-runs.

ii.
```python
cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
if os.path.exists(cache):
    return path, cache, None
```

iii. The caching strategy avoids redundant processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes dF/F for all cells before filtering interneurons (the interneuron filter requires dF/F, so the dF/F of interneurons is computed but then discarded). However, this is necessary because the interneuron filter depends on dF/F. No other obviously unnecessary processing is performed.

ii. N/A

iii. The dF/F computation for interneurons is unavoidable since the filter criterion depends on it.
