# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `pynwb.NWBHDF5IO` to load each NWB file. All NWB files are discovered via `glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb'))`, which finds every `.nwb` file under all `sub-*` directories. Each file is opened with `NWBHDF5IO(path, 'r', load_namespaces=True)`, and all behavior timeseries, ophys fluorescence, neuropil, and segmentation data are read. Multiprocessing (`spawn` context) is used for parallel session processing.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
    b = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
    beh = {k: np.asarray(b[k].data[:], dtype=np.float64) for k in
           ['position', 'speed', 'lick', 'reward_zone', 'environment',
            'trial number', 'trial_start', 'teleport', 'scanning']}
    reward_times = np.asarray(b['Reward'].timestamps[:], dtype=np.float64)
    ...
    f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
    fneu_list.append(np.asarray(nrs.data[:, :], dtype=np.float32)[:, sel].T)
```

iii. The CONVERSION_NOTES document that there are 152 NWB files across 11 subjects, matching the paper's description. The glob pattern finds all files. `pynwb` is used as required by the instructions.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file metadata (`nwb.subject.subject_id`). After all sessions are processed, unique subjects are collected and sorted.

ii.
```python
subject = nwb.subject.subject_id
...
subjects = sorted({r['info']['subject'] for r in results},
                  key=lambda s: int(s[1:]))
```

iii. Subject IDs are read directly from the NWB metadata rather than parsed from directory names. The result is 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from `nwb.session_id`.

ii.
```python
session_id = nwb.session_id
```

iii. One file per session is the NWB convention for this dataset.

## 1-d. How are the data split into trials?

i. Trials are delimited by the `trial_start` and `teleport` behavior variables. `trial_start==1` marks the start and `teleport==1` marks the end. However, the AI uses a shifted window `[start-1, teleport-1)` to match the reference code's `preprocessing.dff` windowing convention, which internally applies a `-1` offset. This means each extracted trial starts one frame before the `trial_start` flag and ends one frame before the `teleport` flag.

ii.
```python
si = np.where(raw['trial_start'] == 1)[0]
ti = np.where(raw['teleport'] == 1)[0]
...
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
```

iii. The AI justifies this shift by noting that the reference code `preprocessing.dff` uses `start-1:stop-1` windowing, and that without the shift, the last sample of every trial comes out NaN. The AI also notes that the reference `behavior.py:57` uses a different window from `dff`, so there is an upstream disagreement in the reference code about the exact window.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied:
1. **Lick-sensor error trials** (>30% of frames with cumulative lick count > 2) are dropped. This removes exactly 81 trials, matching the paper's reported count.
2. Trials with `scanning != 1` inside the `[start-1, end-1)` window are dropped (0 in practice).
3. Trials with < 2 samples or non-finite neural activity are dropped (0 in practice).

ii.
```python
LICK_ERROR_FRAC_THR = 0.3
LICK_ERROR_COUNT_THR = 2
...
lick_error[i] = (np.sum(L > LICK_ERROR_COUNT_THR) / len(L)) > LICK_ERROR_FRAC_THR
...
if lick_error[i]:
    n_dropped_lick += 1
    continue
if scan_bad[i]:
    n_dropped_scan += 1
    continue
if act.shape[1] < 2 or not np.all(np.isfinite(act)):
    n_dropped_nan += 1
    continue
```

iii. The CONVERSION_NOTES explain that lick-error trials are dropped (rather than NaN-ed as in the reference code) because `lick` is a required decoder output and NaN values are not allowed. The 81-trial count exactly matches the paper. The lick threshold of 0.30 was calibrated to reproduce the paper's reported count.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` and `Neuropil` traces stored in the NWB ophys processing module, NOT from the NWB `Deconvolved` field.

ii.
```python
oph = nwb.processing['ophys'].data_interfaces
plane_names = sorted(oph['Fluorescence'].roi_response_series.keys())
for pn in plane_names:
    rrs = oph['Fluorescence'].roi_response_series[pn]
    ...
    f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
    nrs = oph['Neuropil'].roi_response_series[pn]
    fneu_list.append(np.asarray(nrs.data[:, :], dtype=np.float32)[:, sel].T)
```

iii. The CONVERSION_NOTES explain that the NWB `Deconvolved` array is suite2p's own deconvolution of raw fluorescence, not the paper's events. The paper recomputes dF/F and events from raw F/Fneu.

## 2-b. How is the `neural` data processed?

i. dF/F is computed from F and Fneu using the paper's `preprocessing.dff` algorithm: neuropil subtraction (coefficient 0.7) with per-trial mean neuropil added back, per-trial maximin baseline (Gaussian smooth sigma=15, then 300-frame minimum and maximum filters), dF/F = (F - baseline)/|baseline|, smoothed with a 2-sample Gaussian, then optionally OASIS deconvolution. The `keep_teleports` flag is respected based on a per-mouse per-day lookup table copied from the reference code. **dF/F is used as the final neural signal** rather than the deconvolved events.

ii.
```python
def dff_and_events(f, f_neu, trial_starts, teleports, frame_rate, ...):
    f_ -= neu_coef * f_neu_
    for sl in slices:
        f_[:, sl] = f_[:, sl] + neu_coef * np.nanmean(f_neu_[:, sl], axis=1, keepdims=True)
        base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        flow[:, sl] = base
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    for sl in slices:
        smoothed = nansmooth(dff[:, sl], DFF_SMOOTH_SIG, axis=1)
        dff[:, sl] = smoothed
        spks[:, sl] = dcnv.oasis(...)
```

iii. The AI tested both dF/F and events on the sample data and found dF/F decoded better for all 5 of 6 outputs with the per-timepoint decoder. The rationale is that OASIS deconvolution strips temporal integration that the per-timepoint decoder relies on.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters:
1. **iscell curation**: Only ROIs with `iscell[:,0] == 1` are retained (suite2p manual curation).
2. **Putative interneuron removal**: Cells with Pearson r(dF/F, running speed) > 0.5 are excluded.

ii.
```python
iscell = np.asarray(seg['iscell'].data[:])[:, 0].astype(bool)
...
sel = iscell[rid]
f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
...
r_speed = (d_c @ sp_c) / denom
keep_cells = r_speed <= INTERNEURON_SPEED_CORR_THR
dff = dff[keep_cells]
events = events[keep_cells]
```

iii. Both filters match the paper's methods: "putative interneurons ... Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed". The interneuron removal rate (0.29% overall) is consistent with the paper's reported 0.42 +/- 0.85%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The trial window `[start-1, stop-1)` means the first frame is one imaging frame before `trial_start`, so `off_start = -0.0645 s`. Time from trial start is computed as `ts[sl] - ts[s]`, making the first timepoint negative.

ii.
```python
sl = slice(s - 1, e - 1)
act = (events if signal == 'events' else dff)[:, sl]
...
inp[0] = ts[sl] - ts[s]  # t = 0 at the trial_start frame
```

iii. The AI notes that `off_start` is set to the negative of the frame period (~-64.5ms) to reflect that the first sample precedes trial_start by one frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~64.5 ms, 15.5 Hz) is preserved. No temporal rebinning is applied. All sessions have the same frame period.

ii.
```python
dt = float(np.median(np.diff(ts)))
frame_rate = 1.0 / dt
...
'time_bin_size': float(np.median(dts) * 1000.0),
```

iii. The frame period is consistent across all 152 sessions (64.4836 ms). For 2-plane mice, the stored rate is 31 Hz but the per-plane rate is 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps (`b['position'].timestamps[:]`), which serve as the common timestamp vector for all behavioral time series.

ii.
```python
ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
...
inp[0] = ts[sl] - ts[s]
```

iii. All behavior time series share the same timestamps; `position.timestamps` was chosen as the canonical source.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the `trial_start` frame is subtracted from each timestamp within the trial window. Because the window starts at `start-1`, the first value is negative (~-0.0645 s).

ii.
```python
inp[0] = ts[sl] - ts[s]  # t = 0 at the trial_start frame
```

iii. This gives time in seconds relative to trial start, with the first sample at approximately -64.5 ms.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indexing (VR data was already interpolated onto imaging frames in the NWB files). Both use the same `sl = slice(s-1, e-1)` index, ensuring alignment.

ii. Same `sl` index used for both:
```python
act = (events if signal == 'events' else dff)[:, sl]
...
inp[0] = ts[sl] - ts[s]
```

iii. The NWB files already contain VR-aligned data sampled at the imaging frame rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series (morph: 0=ENV1, 1=ENV2).

ii.
```python
morph = raw['environment']
...
mvals = np.unique(morph[s:e])
trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
```

iii. The environment variable is 0 or 1 within a trial, matching the binary ENV1/ENV2 distinction.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique environment value within the trial is taken. If there are multiple values (unlikely), the median is used. The value is broadcast to all timepoints in the trial.

ii.
```python
trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
...
inp[1] = trial_morph[i]
```

iii. The reference code `behavior.get_trial_types` similarly computes a per-trial morph value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the 0-based trial loop index `i` over all trials in the session (before filtering).

ii.
```python
for i, (s, e) in enumerate(zip(si, ti)):
    ...
    inp[2] = i
```

iii. The trial number is the sequential index within the session, matching the reference code's `trial_ids`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing — the loop index `i` is directly used. The index counts all trials including those later dropped, so the trial number can be non-contiguous if trials are filtered.

ii.
```python
inp[2] = i
```

iii. This preserves the original trial ordering in the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavior flag. For each trial, `isreward[i] = has_reward AND any(rzone_flag > 0)`. `previous_trial_rewarded` is then `isreward[i-1]`.

ii.
```python
reward_idx = np.searchsorted(ts, raw['reward_times'])
...
has_reward = np.any((reward_idx >= s) & (reward_idx < e))
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
...
prev_reward = np.zeros(ntrials_all, dtype=np.int64)
prev_reward[1:] = isreward[:-1]
```

iii. The reward zone flag check matches the reference code's `behavior.get_trial_types` which requires both reward delivery and zone-entry. In practice, the rzone flag is 0 exactly on omission trials, so the check is equivalent to just checking reward delivery.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial `isreward` is computed over ALL trials before any filtering. Previous trial outcome is set by shifting: `prev_reward[1:] = isreward[:-1]`. The first trial of each session gets 0. The value is broadcast to all timepoints.

ii.
```python
prev_reward = np.zeros(ntrials_all, dtype=np.int64)
prev_reward[1:] = isreward[:-1]
...
inp[3] = prev_reward[i]
```

iii. Computing isreward before trial filtering ensures that dropped trials still correctly inform the previous-trial outcome of the next trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone identity. The reward zone for each trial is determined from the **scene name** in `nwb.identifier` using `zone_labels_from_scene`, which parses the scene string for zone letters and applies a switch at trial 30.

ii.
```python
scene = nwb.identifier.split('/')[-1]
...
zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
zone_start = np.array([REWARD_ZONE_DICT[z][0] for z in zone_lab])
zone_end = np.array([REWARD_ZONE_DICT[z][1] for z in zone_lab])
...
out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. This matches `behavior.get_reward_zones` which derives zone identity from the scene name with a switch at trial 30. Cross-validated: scene-derived zones agreed with measured reward-zone-entry positions on all 10,394 trials with the flag set.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone: 0 inside [zone_start, zone_end], negative before the zone, positive after.

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

iii. Standard signed-distance computation matching the decoder task specification.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. 7 classes via a manual comparison-based approach:
- 0: d < -50, 1: -50 <= d < -10, 2: -10 <= d < 0, 3: d == 0, 4: 0 < d <= 10, 5: 10 < d <= 50, 6: d > 50

ii.
```python
def discretize_distance(d):
    out = np.full(d.shape, 3, dtype=np.int64)     # d == 0 -> in the reward zone
    out[d < 0] = 2                                # -10 <= d < 0
    out[d < DIST_EDGES[1]] = 1                    # -50 <= d < -10
    out[d < DIST_EDGES[0]] = 0                    # d < -50
    out[d > 0] = 4                                # 0 < d <= 10
    out[d > DIST_EDGES[3]] = 5                    # 10 < d <= 50
    out[d > DIST_EDGES[4]] = 6                    # d > 50
```

iii. The bin edges match the decoder task specification. The approach uses overwriting in sequence rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `sl = slice(s-1, e-1)` index is used for position and neural data, ensuring alignment.

ii.
```python
p = pos[sl]
out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
```

iii. Both neural and position data use the same trial window.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
beh = {k: np.asarray(b[k].data[:], dtype=np.float64) for k in ['position', ...]}
...
p = pos[sl]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
out[1] = np.digitize(p, POS_EDGES)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins using `np.digitize` with edges `(90, 180, 270, 360)`, giving bins 0-4 (< 90, 90-180, 180-270, 270-360, > 360).

ii.
```python
POS_EDGES = (90.0, 180.0, 270.0, 360.0)
out[1] = np.digitize(p, POS_EDGES)
```

iii. Matches the decoder task specification of 5 equal-sized bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices within each trial as neural data.

ii.
```python
p = pos[sl]  # same sl as neural
```

iii. Verified by using the same slice index.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = raw['lick']
...
lk = lick[sl]
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lk > 0).astype(np.int64)
```

iii. Matches the decoder task spec (binary no/yes) and the reference code's `licks[licks>0]=1`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii.
```python
lk = lick[sl]  # same sl as neural
```

iii. Same slice indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the **scene name** in the NWB file identifier, parsed by `zone_labels_from_scene`. The scene name encodes the initial zone and, for switch sessions, the post-switch zone.

ii.
```python
scene = nwb.identifier.split('/')[-1]
...
zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab], dtype=np.int64)
...
out[4] = zone_code[i]
```

iii. This matches `behavior.get_reward_zones` which parses the scene name. The mapping X->A, Y->B, Z->C is handled via the scene string.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. `zone_labels_from_scene` parses the scene name: for fixed-zone sessions it returns the same zone for all trials; for switch sessions the zone changes after trial 30 (`CHANGE_TRIAL=30`). Zones are mapped to integers: A=0, B=1, C=2.

ii.
```python
def zone_labels_from_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    for z in ZONE_ORDER:
        if scene.endswith('Location' + z):
            return np.array([z] * ntrials)
    first = None
    for z in ZONE_ORDER:
        if z + '_to' in scene:
            first = z
    last = scene[-1]
    n0 = min(change_trial, ntrials)
    return np.array([first] * n0 + [last] * (ntrials - n0))
```

iii. The AI verified this against the measured reward-zone-entry positions on all 10,394 rewarded trials with zero mismatches.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and `reward_zone` behavior flag.

ii.
```python
reward_idx = np.searchsorted(ts, raw['reward_times'])
...
has_reward = np.any((reward_idx >= s) & (reward_idx < e))
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
...
out[5] = isreward[i]
```

iii. Per-trial binary value: 1 if reward was delivered and the reward_zone flag was active, 0 otherwise.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to the nearest behavior timepoint using `searchsorted`. For each trial, check if any reward timestamp falls within the trial window AND the reward_zone flag is active. The value is broadcast to all timepoints.

ii.
```python
isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
...
out[5] = isreward[i]
```

iii. Matches `behavior.get_trial_types`. The rzone flag check is functionally redundant (it's 0 exactly on omission trials) but follows the reference code.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: 10 of 152 files (2-plane mice) have one more ophys frame than VR samples; all streams are trimmed to the shorter length.
- **Lick-sensor error trials**: 81 trials with corrupted lick data are dropped.
- **Scanning flag**: Trials with `scanning != 1` would be dropped (0 in practice).
- **Non-finite neural data**: Trials with NaN/Inf in the neural data after dF/F computation would be dropped (0 in practice).
- **Trial boundaries out of range**: Trials whose window would start before frame 0 or end past the last frame are skipped (0 in practice).

ii.
```python
n = min(len(ts), f.shape[1])
n_trimmed = max(len(ts), f.shape[1]) - n
ts = ts[:n]
beh = {k: v[:n] for k, v in beh.items()}
f, f_neu = f[:, :n], f_neu[:, :n]
...
keep = (si >= 1) & (ti <= len(ts))
...
if lick_error[i]: continue
if scan_bad[i]: continue
if act.shape[1] < 2 or not np.all(np.isfinite(act)): continue
```

iii. These defensive checks were developed during data exploration and documented in CONVERSION_NOTES.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **dF/F + OASIS deconvolution** (~10-16 s per session for ~1000 cells) — dominated by per-trial Gaussian/min/max filters and OASIS.
2. **NWB file I/O** (~1-3.5 s per session reading Fluorescence + Neuropil arrays).
3. **Pickle serialization** (~14 s for the 9.6 GB output file).

ii. N/A

iii. Timing is reported per session. Total conversion with 24-way multiprocessing: ~2.4 min.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` (lines 357-392) iterates over trials sequentially. Some operations like digitize and distance computation could be applied to full session arrays, but variable trial lengths make this complex. The interneuron speed-correlation was already vectorized over all cells.

ii. N/A

iii. Variable-length trials are the main obstacle to full vectorization.

## 13-c. What processing does the code repeat multiple times?

i. The `dff_and_events` function computes both dF/F and events (full OASIS deconvolution) even when only dF/F is used as the neural signal. The deconvolution step adds computational cost that is discarded when `--signal dff` (the default) is used.

ii.
```python
dff, events = dff_and_events(raw['f'], raw['f_neu'], si, ti, frame_rate)
...
act = (events if signal == 'events' else dff)[:, sl]
```

iii. Both signals are computed because both are needed for the interneuron removal (which uses dF/F) and the neural output (which uses whichever is selected).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **OASIS deconvolution** is always computed even when the default dF/F signal is used. The `events` array is computed but discarded when `--signal dff`. Additionally, the full dF/F and events arrays are computed for the entire session before splitting into trials; only in-trial samples are used by the decoder but the computation covers all frames including inter-trial gaps (which are NaN).
