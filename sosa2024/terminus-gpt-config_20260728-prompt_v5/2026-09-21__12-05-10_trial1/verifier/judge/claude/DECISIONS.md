# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by recursively globbing `*.nwb` under the data root directory. Each NWB file is opened with `pynwb.NWBHDF5IO`, and the neural (ophys) and behavioral data are extracted in a single pass per file.

ii.
```python
def list_sessions(data_root: Path):
    return sorted(data_root.rglob('*.nwb'))

def read_session(nwb_path: Path):
    with NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        ...
        deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
        ...
        beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
```

iii. The AI notes in CONVERSION_NOTES.md that data are stored as DANDI-style NWB files under subject folders in `/app/data`, and uses pynwb for loading. The recursive glob finds all 152 NWB files across all 11 subjects.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file's `subject.subject_id` field, falling back to the parent directory name (stripping `sub-`). A dictionary maps subject IDs to sequential indices.

ii.
```python
subj = nwb.subject.subject_id if nwb.subject is not None else nwb_path.parent.name.replace('sub-', '')
...
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
```

iii. The AI uses the NWB metadata for subject ID. This gives the same subject identifiers as parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session ID is read from the NWB's `session_id` field.

ii.
```python
sess_id = getattr(nwb, 'session_id', nwb_path.stem)
```

iii. The AI treats each NWB file as a separate session, consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trials are identified by finding timepoints where `trial_start > 0` (within the valid mask), then using `contiguous_segments` to create segments from one trial start to the next trial start (or end of recording). Within each segment, a validity mask (`trial number >= 0` and `scanning > 0`) is applied to select only valid timepoints.

ii.
```python
trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)
segs = contiguous_segments(trial_start_idx, len(t))

def contiguous_segments(start_idxs, n_time):
    starts = list(start_idxs)
    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_time
        if e > s:
            segs.append((s, e))
    return segs
```

iii. The AI segments trials using trial_start events and uses the valid mask to filter within. However, this approach does not use the `teleport` signal to define trial ends. The reference uses `teleport` transitions to find trial boundaries, excluding inter-trial intervals (teleport periods). The AI's approach includes timepoints between the teleport and the next trial start if they pass the valid mask.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) applying a valid mask (`trial number >= 0` and `scanning > 0`) within each segment, requiring at least 5 valid timepoints; (2) requiring at least 5 valid position samples within [0, 450] cm; (3) requiring at least 2 kept trials per session.

ii.
```python
valid = np.isfinite(t) & (b['trial number'] >= 0)
if 'scanning' in b:
    valid &= (b['scanning'] > 0)
...
mask = valid[s:e]
if mask.sum() < 5:
    continue
pos = np.asarray(b['position'][s:e])[mask]
if np.sum((pos >= 0) & (pos <= 450)) < 5:
    continue
...
if len(kept_segs) < 2:
    return [], [], [], ...
```

iii. The AI uses a minimum of 5 valid timepoints per trial. The reference uses a minimum of 50 timepoints (`min_ntimepoints=50`).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the `Deconvolved` ROI response series from the NWB's ophys processing module, specifically hardcoded to `plane0` only.

ii.
```python
ophys = nwb.processing['ophys']
deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
```

iii. The AI's CONVERSION_NOTES Step 5 states: "Use `Deconvolved` ROIResponseSeries as neural input: This is the most directly usable processed neural activity stream in NWB." However, the paper's Methods describe computing their own dF/F and deconvolved events from raw Fluorescence and Neuropil traces, not using the stored Deconvolved data.

## 2-b. How is the `neural` data processed?

i. No processing is applied. The AI reads the `Deconvolved` array directly and uses it as-is (cast to float32).

ii.
```python
deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
...
nn = neural[idx, :].T.astype(np.float32)
```

iii. The AI chose not to recompute neural signals, instead using the stored Deconvolved data. The reference computes dF/F from Fluorescence and Neuropil using the paper's pipeline (neuropil subtraction with coef=0.7, maximin baseline, Gaussian smoothing, OASIS deconvolution with tau=0.7).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural quality filtering is applied. The AI does not filter by `iscell` (suite2p's manual curation) and does not remove putative interneurons. Additionally, only `plane0` is read, missing neurons from `plane1` in two-plane sessions (mice m17, m18).

ii.
```python
deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
roi_count = deconv.shape[1]
```

iii. The AI does not mention iscell filtering or interneuron removal in its CONVERSION_NOTES. The reference applies both: iscell filtering (suite2p's manual curation) and putative interneuron removal (cells whose dF/F correlates with running speed at r > 0.5).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by splitting into trial segments. The first timepoint of each trial corresponds to the trial start, so no additional temporal shifting is needed.

ii.
```python
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
```

iii. Trial start alignment is achieved by taking the segment of neural data corresponding to each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate is used without rebinning. The time bin size is computed as the median of inter-frame intervals across sessions.

ii.
```python
dts = np.diff(sess['timestamps'])
dts = dts[np.isfinite(dts) & (dts > 0)]
if len(dts):
    dt_list.append(float(np.median(dts)))
...
time_bin = float(np.median(dt_list) * 1000.0) if dt_list else np.nan
```

iii. No rebinning is applied, consistent with the reference approach.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `trial_start` behavior time series timestamps, which serve as the common time base for all behavior data.

ii.
```python
frame_timestamps = np.asarray(beh.time_series['trial_start'].timestamps[:], dtype=np.float64)
...
tt = t[idx] - t[idx[0]]
inp = np.vstack([tt.astype(np.float32), ...])
```

iii. The AI uses the trial_start timestamps as the frame timestamps. The reference uses the `trial number` timestamps, but both are the same underlying behavior sampling grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in the trial.

ii.
```python
tt = t[idx] - t[idx[0]]
```

iii. Standard approach; matches the reference.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (frame-aligned), so both are indexed by the same `idx` array.

ii.
```python
idx = np.flatnonzero(mask) + s
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
```

iii. Since neural and behavior data are both indexed at the imaging frame rate, they are inherently aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = np.asarray(b['environment'][s:e])[mask]
env_per_trial.append(np.nanmedian(env))
```

iii. The environment variable records the VR environment type at each timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial environment value is computed as the median of the environment values within the trial. Then across all trials, unique values are sorted and mapped to binary 0/1 codes.

ii.
```python
env_per_trial.append(np.nanmedian(env))
...
env_codes, env_map = map_binary_environment(np.asarray(env_per_trial, dtype=float))

def map_binary_environment(env_trial_vals):
    vals = env_trial_vals[~np.isnan(env_trial_vals)]
    uniq = sorted(np.unique(vals).tolist())
    mapping = {v: i for i, v in enumerate(uniq_sorted[:2])}
    out = np.array([mapping.get(v, 0) for v in env_trial_vals], dtype=np.int64)
    return out, mapping
```

iii. The reference simply casts the environment value to int. Since environment is already 0/1 per timepoint and constant within trials, both approaches yield the same result.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the sequential index of the kept trial within the session (the loop variable `i` over `kept_segs`).

ii.
```python
for i, (s, e) in enumerate(kept_segs):
    ...
    inp = np.vstack([
        ...
        np.full(len(idx), i, dtype=np.float32),
        ...
    ])
```

iii. The AI uses the sequential index over kept trials. The reference uses the loop counter over all trials (including skipped ones), so filtered trials create gaps in numbering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index as a constant across all timepoints in the trial.

ii.
```python
np.full(len(idx), i, dtype=np.float32)
```

iii. The trial number is constant per trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward events are aligned to frame timestamps, and a per-trial reward flag is computed.

ii.
```python
rw = np.asarray(b['Reward'][s:e])[mask]
rew_per_trial.append(int(np.any(rw > 0)))
```

iii. The `Reward` time series records reward delivery events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward event (Reward > 0) occurred within the trial's valid timepoints. The previous trial's outcome is then used as the current trial's input. For the first trial, set to 0.

ii.
```python
rew_per_trial.append(int(np.any(rw > 0)))
...
prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)
...
np.full(len(idx), prev_rew[i], dtype=np.float32)
```

iii. The reference uses a similar approach: checking if any reward event occurred in the previous trial's time range.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavior time series. The reward zone interval is inferred per trial by finding the min and max position where `reward_zone > 0`. The zone location (A/B/C) is then assigned based on the center of this interval.

ii.
```python
def infer_zone_interval_and_location(rz_vals, pos_vals):
    m = (rz_vals > 0) & np.isfinite(pos_vals) & (pos_vals >= 0) & (pos_vals <= 450)
    if np.any(m):
        lo = float(np.min(pos_vals[m]))
        hi = float(np.max(pos_vals[m]))
    else:
        c = float(np.nanmedian(pos_vals[np.isfinite(pos_vals)]))
        lo, hi = c - 10.0, c + 10.0
    center = 0.5 * (lo + hi)
    if center < 200:
        loc = 0
    elif center < 300:
        loc = 1
    else:
        loc = 2
    return lo, hi, loc
```

iii. The AI infers per-trial reward zone boundaries from the data. The reference uses a Viterbi algorithm to assign zones and then uses the canonical zone ranges from the paper (A=[80,130], B=[200,250], C=[320,370]).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, compute signed distance from position to the inferred zone interval (per-trial [lo, hi]). Distance is 0 when inside, negative before, positive after. Position is clipped to [0, 450] before distance computation.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
zlo, zhi = rz_interval_per_trial[i]
dist = signed_distance_to_interval(pos_clip, zlo, zhi)

def signed_distance_to_interval(pos, lo, hi):
    dist = np.zeros_like(pos, dtype=np.float32)
    dist[pos < lo] = pos[pos < lo] - lo
    dist[pos > hi] = pos[pos > hi] - hi
    return dist
```

iii. The reference computes distance to the canonical zone ranges (from `reward_zone_dict`), not per-trial inferred intervals. The AI's per-trial interval inference can be noisy.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Binned into 7 categories using explicit comparisons:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to < 0 cm
- 3: exactly 0 cm
- 4: >0 to 10 cm
- 5: 10 to 50 cm
- 6: > 50 cm

ii.
```python
def bin_distance(dist):
    out = np.full(dist.shape, 6, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. The bin edges match the instructions. The reference uses `np.digitize` with edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]` achieving similar bin assignments.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data share the same time indices within each trial.

ii.
```python
idx = np.flatnonzero(mask) + s
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. Both indexed by the same `idx` array.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = np.asarray(b['position'][idx], dtype=np.float32)
pos_clip = np.clip(pos, 0, 450)
```

iii. The `position` variable records the animal's position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm, then discretized into 5 bins.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
bin_position(pos_clip)
```

iii. The reference does not clip position; it uses `np.digitize` with open first and last bins. The clipping is a minor difference since very few samples fall outside [0, 450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: 0 (<90), 1 (90-180), 2 (180-270), 3 (270-360), 4 (>=360).

ii.
```python
def bin_position(pos):
    out = np.full(pos.shape, 0, dtype=np.int64)
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
    return out
```

iii. Matches the instructions for 5 equal-sized bins spanning 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. Same `idx` indexing as neural data.

iii. Inherently aligned via shared frame timestamps.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. Matches the reference approach and instruction specification.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii. Same `idx` indexing.

iii. Inherently aligned.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavior time series. Per-trial zone intervals are inferred from positions where `reward_zone > 0`, and the center of the interval determines A (0), B (1), or C (2).

ii.
```python
zlo, zhi, zloc = infer_zone_interval_and_location(rz, pos)
rz_per_trial.append(zloc)
...
np.full(len(idx), rz_codes[i], dtype=np.int64)
```

iii. The AI infers zone location from per-trial position data. The reference uses a Viterbi algorithm with canonical zone ranges from the paper.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The center of the inferred zone interval determines the zone: center < 200 cm maps to A (0), center < 300 cm maps to B (1), else C (2). For trials without reward zone activity, a fallback based on median position is used.

ii.
```python
center = 0.5 * (lo + hi)
if center < 200:
    loc = 0
elif center < 300:
    loc = 1
else:
    loc = 2
```

iii. The reference uses Viterbi with Gaussian emission and transition probabilities to robustly assign zone labels across trials, handling noise and missing data better.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series.

ii.
```python
rw = np.asarray(b['Reward'][s:e])[mask]
rew_per_trial.append(int(np.any(rw > 0)))
...
np.full(len(idx), rew_per_trial[i], dtype=np.int64)
```

iii. Reward events indicate whether the animal received a reward during the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any Reward value > 0 within the trial's valid timepoints, else 0.

ii.
```python
rew_per_trial.append(int(np.any(rw > 0)))
```

iii. The reference similarly checks for any reward event within the trial. The AI's approach of checking `rw > 0` within the valid-masked segment may include some teleport-period reward events if those pass the valid mask.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Sparse events**: The `align_series_to_frame` function handles sparse event-like time series (e.g., Reward) by finding nearest frame timestamps.
- **NaN/invalid timepoints**: A validity mask (`trial number >= 0`, `scanning > 0`, `isfinite(timestamps)`) excludes invalid data.
- **Short trials**: Trials with < 5 valid timepoints or < 5 valid position samples are dropped.
- **Missing reward zone**: When no reward_zone > 0 in a trial, a fallback uses median position +/- 10 cm.
- **Position clamping**: Position is clipped to [0, 450] for discretization.

ii.
```python
valid = np.isfinite(t) & (b['trial number'] >= 0)
if 'scanning' in b:
    valid &= (b['scanning'] > 0)
...
if mask.sum() < 5:
    continue
...
if np.sum((pos >= 0) & (pos <= 450)) < 5:
    continue
```

iii. The reference handles neural/behavior length mismatch by cropping, uses min 50 timepoints, and uses the Viterbi algorithm to handle noisy reward zone assignments.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with pynwb (I/O bound)
2. **Reading large arrays** from the NWB (deconvolved neural data, behavior time series)

ii. N/A

iii. The AI's code is simpler (no dff computation), so loading dominates.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial sequentially for constructing input/output arrays. The trial-level operations (bin_distance, bin_position, etc.) are already vectorized within each trial. The outer trial collection loop could potentially be replaced with whole-session vectorized operations followed by splitting.

ii. N/A

iii. The variable trial lengths make full vectorization awkward.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session in a single pass (no separate survey step), so there is minimal repeated processing. The `align_series_to_frame` function is called for each behavior variable, but the alignment computation is lightweight.

ii. N/A

iii. The single-pass design avoids the double-loading issue present in the reference's survey + conversion approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `align_series_to_frame` function includes interpolation fallback logic and sparse event alignment that may not be needed for all behavior variables (many are already frame-aligned). The code also extracts `scanning` and `trial number` behavior variables that are only used for filtering, not as decoder inputs or outputs.

ii. N/A

iii. These are minor inefficiencies that don't significantly impact runtime.
