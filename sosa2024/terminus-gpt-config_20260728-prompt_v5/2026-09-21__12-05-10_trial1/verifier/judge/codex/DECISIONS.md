# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script recursively finds every `.nwb` file under `/app/data`, treats each file as a session, opens it with `pynwb`, reads the `behavior` and `ophys` processing modules, and later processes every discovered session in a single pass.

ii.
```python
def list_sessions(data_root: Path):
    return sorted(data_root.rglob('*.nwb'))

def read_session(nwb_path: Path):
    with NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
        ophys = nwb.processing['ophys']
        ...

def convert(data_root: Path, out_path: Path, sample=False, show_processing=False):
    files = list_sessions(data_root)
    ...
    for j, f in enumerate(files):
        sess = read_session(f)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent justified treating the full archive as the dataset and treating each NWB file as one session. In the trajectory it also noted that `nwb.trials` was absent, so the NWB behavior and ophys interfaces had to be read directly.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id` when present, with a fallback to the parent folder name stripped of the `sub-` prefix. The final `subjects` list is the order of first appearance while iterating through session files.

ii.
```python
subj = nwb.subject.subject_id if nwb.subject is not None else nwb_path.parent.name.replace('sub-', '')
...
subjects = []
subject_to_idx = {}
...
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
```

iii. The notes say the archive has 11 subject folders, but the implemented converter uses the NWB metadata first. That matches the agent's general preference in the trajectory for relying on NWB content instead of only directory naming.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Session identity is taken from `nwb.session_id` when available, otherwise from the file stem.

ii.
```python
def list_sessions(data_root: Path):
    return sorted(data_root.rglob('*.nwb'))

sess_id = getattr(nwb, 'session_id', nwb_path.stem)
...
for j, f in enumerate(files):
    sess = read_session(f)
```

iii. In Step 4 and Step 5 of `CONVERSION_NOTES.md`, the agent explicitly decided that one NWB file corresponds to one session because the archive is organized per subject-session file and standard NWB trial tables are absent.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from positive samples in the aligned `trial_start` series. Each trial begins at a positive `trial_start` index and extends to the next `trial_start` index, or to the session end for the last trial. Within that window, only samples passing the `valid` mask are retained.

ii.
```python
valid = np.isfinite(t) & (b['trial number'] >= 0)
if 'scanning' in b:
    valid &= (b['scanning'] > 0)

trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)

def contiguous_segments(start_idxs, n_time):
    starts = list(start_idxs)
    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_time
        if e > s:
            segs.append((s, e))
    return segs
```

iii. In the trajectory, the agent noted that `trial_start` was a clean binary pulse and that `trial number` was unreliable during off-track periods. It therefore chose `trial_start`-based segmentation plus validity masking instead of using the reference script's `teleport` end points.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has fewer than 5 valid samples, or fewer than 5 valid position samples within `[0, 450]`. After filtering, a session is dropped if fewer than 2 trials remain.

ii.
```python
for (s, e), trn in zip(segs, trial_nums):
    mask = valid[s:e]
    if mask.sum() < 5:
        continue
    pos = np.asarray(b['position'][s:e])[mask]
    if np.sum((pos >= 0) & (pos <= 450)) < 5:
        continue
    ...
    kept_segs.append((s, e))

if len(kept_segs) < 2:
    return [], [], [], np.zeros(sess['roi_count'], dtype=np.int64)
```

iii. The trajectory shows the agent added these minimal filters after observing off-track samples and sparse streams. The notes frame this as a pragmatic QC step to ensure decoder-compatible trials and sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB `ophys/Deconvolved/plane0` ROI response series only.

ii.
```python
ophys = nwb.processing['ophys']
deconv = np.asarray(
    ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:],
    dtype=np.float32
)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the agent chose `Deconvolved` because it was the "most directly usable processed neural activity stream" and aligned naturally to imaging frames. The trajectory repeats that `Deconvolved/plane0` had the expected time-by-ROI shape.

## 2-b. How is the `neural` data processed?

i. The script does not recompute dF/F or deconvolution. It takes the stored `Deconvolved/plane0` array, keeps the rows for each trial's valid time indices, transposes from time-by-neuron to neuron-by-time, and casts to `float32`.

ii.
```python
deconv = np.asarray(..., dtype=np.float32)
...
nn = neural[idx, :].T.astype(np.float32)
...
neural_trials.append(nn)
```

iii. The notes justify this as a simpler decoder-ready signal than raw fluorescence. The agent did inspect `Fluorescence` and `Neuropil` in exploration, but decided not to reproduce the paper's full preprocessing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no neuron-level quality filter in the final script. The script does not apply `iscell`, does not remove putative interneurons, and does not combine multiple planes. The only filtering around neural data is indirect timepoint filtering through the `valid` mask used for trial extraction.

ii.
```python
deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
roi_count = deconv.shape[1]
...
valid = np.isfinite(t) & (b['trial number'] >= 0)
if 'scanning' in b:
    valid &= (b['scanning'] > 0)
```

iii. The notes do not claim a neuron curation step. Instead, Step 5 and the trajectory justify the choice as using a directly stored processed neural signal, with trial/sample QC but not paper-style cell QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by segmenting trials on the `trial_start` event and then expressing each trial from its own first kept sample onward.

ii.
```python
trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)
...
for i, (s, e) in enumerate(kept_segs):
    mask = valid[s:e]
    idx = np.flatnonzero(mask) + s
    nn = neural[idx, :].T.astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly says the decoder conversion should align to `trial_start`, which the agent treated as the required event from the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sample spacing of the aligned frame timestamps and does not apply explicit rebinning. It records the dataset time bin size as the median inter-frame interval across sessions, in milliseconds.

ii.
```python
dts = np.diff(sess['timestamps'])
dts = dts[np.isfinite(dts) & (dts > 0)]
if len(dts):
    dt_list.append(float(np.median(dts)))
...
time_bin = float(np.median(dt_list) * 1000.0) if dt_list else np.nan
```

iii. In the trajectory, the agent argued that imaging-frame timestamps were the best common time base and that keeping the native resolution avoided interpolation complexity beyond aligning sparse behavior streams to those frames.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `frame_timestamps`, which the script takes from `behavior/BehavioralTimeSeries/trial_start.timestamps`.

ii.
```python
frame_timestamps = np.asarray(beh.time_series['trial_start'].timestamps[:], dtype=np.float64)
...
t = sess['timestamps']
...
tt = t[idx] - t[idx[0]]
```

iii. The notes and trajectory emphasize using imaging-frame-aligned behavior timestamps as the common session time base. The agent therefore used those timestamps directly instead of a separate behavioral clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first kept timestamp is subtracted from the aligned timestamps to make time start at zero.

ii.
```python
tt = t[idx] - t[idx[0]]
...
inp = np.vstack([
    tt.astype(np.float32),
    ...
])
```

iii. The justification is implicit in the trial-start alignment requirement and made explicit in Step 5 of the notes: the decoder input should be time from trial onset on the same grid as the trial neural data.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same trial index array `idx` is used to slice both timestamps and neural data. Before that, behavior streams are aligned to the chosen frame timestamp grid with `align_series_to_frame`.

ii.
```python
beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
idx = np.flatnonzero(mask) + s
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
```

iii. The trajectory shows the agent discovered that `Reward` and some other streams were sparse or off-grid, so it introduced `align_series_to_frame` specifically to put all behavior variables onto the neural/frame time base.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the aligned `behavior/BehavioralTimeSeries/environment` series.

ii.
```python
ts_names = ['Reward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']
beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
env = np.asarray(b['environment'][s:e])[mask]
```

iii. In Step 5 the agent mapped `environment` directly to the decoder input and planned to turn it into a binary per-trial variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment values within each kept trial are summarized by `np.nanmedian`, then the set of observed trial-level values is sorted and the first two unique values are remapped to `0` and `1`. The per-trial code is broadcast across all timepoints in the trial.

ii.
```python
env_per_trial.append(np.nanmedian(env))
...
def map_binary_environment(env_trial_vals):
    vals = env_trial_vals[~np.isnan(env_trial_vals)]
    uniq_sorted = sorted(np.unique(vals).tolist())
    mapping = {v: i for i, v in enumerate(uniq_sorted[:2])}
    out = np.array([mapping.get(v, 0) for v in env_trial_vals], dtype=np.int64)
    return out, mapping
...
np.full(len(idx), env_codes[i], dtype=np.float32)
```

iii. The notes say the output should be binary `ENV1` vs `ENV2`. In the trajectory, the agent reported seeing values like `-1` and `0` in sampled data and therefore chose an explicit remapping step rather than trusting the raw coding.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The final `trial_number` input is derived from the loop index over kept trial segments, not from the raw `trial number` stream. The raw `trial number` is only used earlier to build the `valid` mask and to confirm that a segment contains in-trial samples.

ii.
```python
valid = np.isfinite(t) & (b['trial number'] >= 0)
...
trial_nums = []
for s, e in segs:
    tr = b['trial number'][s:e]
    tr_valid = tr[tr >= 0]
    if len(tr_valid) == 0:
        continue
    trial_nums.append(int(np.round(np.median(tr_valid))))
...
np.full(len(idx), i, dtype=np.float32)
```

iii. The notes state that the decoder should use a per-trial quantity. In the trajectory, the agent said `trial_start` looked cleaner than relying on the raw `trial number` values, so the final exported quantity became a sequential kept-trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond assigning the kept-trial index and broadcasting it across all timepoints in the trial.

ii.
```python
inp = np.vstack([
    tt.astype(np.float32),
    np.full(len(idx), env_codes[i], dtype=np.float32),
    np.full(len(idx), i, dtype=np.float32),
    np.full(len(idx), prev_rew[i], dtype=np.float32),
])
```

iii. This follows the Step 5 note that per-trial variables would be broadcast over time so all decoder inputs share the same `(n_input, n_timepoints)` shape.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the aligned `Reward` series.

ii.
```python
rw = np.asarray(b['Reward'][s:e])[mask]
...
rew_per_trial.append(int(np.any(rw > 0)))
...
prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)
```

iii. The trajectory shows the agent discovered that `Reward` was a sparse event-like stream with its own timestamps. That led directly to the frame-alignment step and then to per-trial reward detection.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, the current-trial outcome is computed as whether any aligned reward sample in the trial is positive. Then the previous-trial outcome vector is formed by shifting those trial outcomes by one, prepending 0 for the first trial, and broadcasting the result across each trial.

ii.
```python
rew_per_trial.append(int(np.any(rw > 0)))
...
prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)
...
np.full(len(idx), prev_rew[i], dtype=np.float32)
```

iii. The notes describe previous reward outcome as a binary per-trial contextual variable, and the trajectory records that the sparse `Reward` stream had to be aligned before this could be done correctly.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` and `reward_zone`. The script infers a per-trial reward-zone interval from positions where `reward_zone > 0`, and then uses position relative to that interval.

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
    ...

pos = np.asarray(b['position'][idx], dtype=np.float32)
zlo, zhi = rz_interval_per_trial[i]
dist = signed_distance_to_interval(pos_clip, zlo, zhi)
```

iii. The trajectory shows the agent initially misread the raw `reward_zone` codes, then concluded they were sparse local annotations near the reward zone rather than the requested output itself. It therefore switched to inferring a zone interval from `reward_zone > 0` samples.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to `[0, 450]`, the signed distance to the inferred reward-zone interval is computed, with negative values before the interval, zero inside it, and positive values after it.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
...
def signed_distance_to_interval(pos, lo, hi):
    dist = np.zeros_like(pos, dtype=np.float32)
    dist[pos < lo] = pos[pos < lo] - lo
    dist[pos > hi] = pos[pos > hi] - hi
    return dist
```

iii. The notes say the first implementation was replaced after sanity checks. The revised justification was that distance should be computed from an inferred interval, not from the raw multi-valued `reward_zone` code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The script manually thresholds the continuous signed distance into the 7 requested bins.

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

iii. The trajectory explicitly says the interval-based fix was chosen to make all 7 classes appear in sensible proportions matching the task bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The same per-trial index array `idx` is used to slice neural activity and position. Distance is computed on those trial-aligned position samples, so it stays on the neural time base.

ii.
```python
idx = np.flatnonzero(mask) + s
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
...
dist = signed_distance_to_interval(pos_clip, zlo, zhi)
```

iii. The notes and trajectory repeatedly justify one common frame-based time base for neural and behavioral streams; this output follows that design.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the aligned `position` series.

ii.
```python
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `position` directly to the absolute-position decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the physical track range `[0, 450]` and then discretized.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
...
bin_position(pos_clip)
```

iii. The notes metadata explicitly states that position was clipped to `[0,450]` for discretization. In the trajectory, this was justified by the presence of off-track/teleport values outside the corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five 90 cm bins: `<90`, `90-180`, `180-270`, `270-360`, and `>=360`.

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

iii. This follows the decoder-task specification directly; the notes describe it as one of the requested discretizations.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position` with the same trial/sample indices used for the neural matrix.

ii.
```python
idx = np.flatnonzero(mask) + s
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. The common frame-time-base design in `align_series_to_frame` is the AI's stated reason that extra alignment steps were unnecessary at trial extraction time.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the aligned `lick` series.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `lick` directly to the decoder output, and the trajectory notes that the raw values could exceed 1, motivating binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The aligned lick values are binarized: values `> 0` become `1`, else `0`.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. In the trajectory, the agent explicitly noted that raw lick values sometimes exceeded 1 and that binary decoding required thresholding them.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced on the same trial/sample indices as the neural data after being aligned to the session frame timestamps.

ii.
```python
beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
nn = neural[idx, :].T.astype(np.float32)
```

iii. The trajectory indicates this shared alignment machinery was added after discovering that not every behavior series naturally had the same sample count as the frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`, via the same inferred per-trial reward-zone interval used for the distance output.

ii.
```python
rz = np.asarray(b['reward_zone'][s:e])[mask]
pos = np.asarray(b['position'][s:e])[mask]
zlo, zhi, zloc = infer_zone_interval_and_location(rz, pos)
rz_per_trial.append(zloc)
```

iii. The trajectory shows the agent rejected using the raw `reward_zone` codes directly and instead treated nonzero `reward_zone` samples as evidence for the interval's location along the track.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The interval center is converted to an A/B/C-like code by thresholding its midpoint: `<200` becomes 0 (A), `200-300` becomes 1 (B), and `>300` becomes 2 (C). That per-trial code is then broadcast across the trial.

ii.
```python
center = 0.5 * (lo + hi)
if center < 200:
    loc = 0
elif center < 300:
    loc = 1
else:
    loc = 2
...
np.full(len(idx), rz_codes[i], dtype=np.int64)
```

iii. The notes and trajectory justify this as a simple way to map inferred interval locations to the three canonical reward zones after observing that `reward_zone > 0` clustered around narrow position bands.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the aligned `Reward` series.

ii.
```python
rw = np.asarray(b['Reward'][s:e])[mask]
rew_per_trial.append(int(np.any(rw > 0)))
...
np.full(len(idx), rew_per_trial[i], dtype=np.int64)
```

iii. The trajectory identifies `Reward` as a sparse event series and says the frame-alignment code was introduced partly so current-trial and previous-trial reward outcomes could be computed on the same time base as the neural data.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. After alignment to frame timestamps, a trial is labeled rewarded if any sample in that trial has `Reward > 0`; otherwise it is labeled 0. The resulting scalar is broadcast across the trial.

ii.
```python
rew_per_trial.append(int(np.any(rw > 0)))
...
out = np.vstack([
    ...,
    np.full(len(idx), rew_per_trial[i], dtype=np.int64),
]).astype(np.int64)
```

iii. The notes describe this as a per-trial binary decoder output. The trajectory says sparse reward timing had to be resolved before this could be done reliably.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles irregularities with permissive fallbacks rather than hard failures. `align_series_to_frame` pads, rasterizes sparse events, or interpolates onto the frame grid depending on the series shape. Empty time series become zeros. Trials with too few valid or in-track samples are skipped. If no nonzero `reward_zone` samples exist in a trial, the script falls back to a 20 cm interval centered on the median position. Sessions with fewer than two kept trials are discarded.

ii.
```python
if ts_t is None:
    if len(data) == len(frame_timestamps):
        return data
    out = np.full(len(frame_timestamps), np.nan, dtype=np.float64)
    n = min(len(data), len(out))
    out[:n] = data[:n]
    return out
...
if len(data) == 0:
    return np.zeros(len(frame_timestamps), dtype=np.float64)
...
return np.interp(frame_timestamps, ts_t, data)
...
if mask.sum() < 5:
    continue
...
if np.any(m):
    lo = float(np.min(pos_vals[m]))
    hi = float(np.max(pos_vals[m]))
else:
    c = float(np.nanmedian(pos_vals[np.isfinite(pos_vals)]))
    lo, hi = c - 10.0, c + 10.0
```

iii. The notes explicitly mention two fixes found during validation: sparse `Reward` needed explicit frame alignment, and reward-zone distance needed interval inference. The general pattern in the trajectory is to patch around observed data issues so conversion can proceed on the full archive.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive parts are session-level NWB reads, per-session alignment of all behavior series to the frame grid, and the two per-trial passes inside `process_session` (one to decide which trials to keep and infer per-trial metadata, one to materialize neural/input/output arrays). Optional plotting also adds overhead for sampled sessions.

ii.
```python
for j, f in enumerate(files):
    sess = read_session(f)
    ...
    ntr, itr, otr, bri = process_session(sess, make_plot=show_processing and j < 2)

beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
for (s, e), trn in zip(segs, trial_nums):
    ...
for i, (s, e) in enumerate(kept_segs):
    ...
```

iii. `CONVERSION_NOTES.md` Step 9 reports about 98 seconds for the full 152-session run, and the trajectory repeatedly attributes most runtime to loading and processing large session files rather than to model logic.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The code uses Python loops over sessions, over trial segments, and within `contiguous_segments`. The two main opportunities for vectorization are the first pass over trials that computes `env_per_trial`, `rz_interval_per_trial`, and `rew_per_trial`, and the second pass that slices and stacks the per-trial arrays.

ii.
```python
for j, f in enumerate(files):
    ...

for (s, e), trn in zip(segs, trial_nums):
    ...

for i, (s, e) in enumerate(kept_segs):
    ...
```

iii. The agent's notes say the first-pass implementation was already "vectorized per-trial masking" compared with per-timepoint loops, but the structure still centers on explicit variable-length trial loops.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code walks the trial segments twice: once to decide which segments to keep and to infer per-trial context, and again to build the actual `neural`, `input`, and `output` arrays. It also reuses the same segment masks to look up `position`, `environment`, `reward_zone`, and `Reward` in both passes.

ii.
```python
for (s, e), trn in zip(segs, trial_nums):
    mask = valid[s:e]
    ...
    env = np.asarray(b['environment'][s:e])[mask]
    rz = np.asarray(b['reward_zone'][s:e])[mask]
    rw = np.asarray(b['Reward'][s:e])[mask]
    ...

for i, (s, e) in enumerate(kept_segs):
    mask = valid[s:e]
    idx = np.flatnonzero(mask) + s
    ...
    pos = np.asarray(b['position'][idx], dtype=np.float32)
    speed = np.asarray(b['speed'][idx], dtype=np.float32)
    lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. The trajectory and notes focus more on correctness than on removing this duplication. The repeated pass is the straightforward way the agent chose to separate trial filtering/metadata inference from final tensor construction.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes several values that are not used downstream: `trial_nums`, `env_map`, `rz_map`, and aligned `teleport` values. It also supports optional sample-session plotting that is not used by the saved dataset. More broadly, it aligns some behavior series solely for gating or inspection rather than because they are exported.

ii.
```python
trial_nums = []
...
env_codes, env_map = map_binary_environment(np.asarray(env_per_trial, dtype=float))
rz_codes = np.asarray(rz_per_trial, dtype=np.int64)
rz_map = {0: 0, 1: 1, 2: 2}
...
ts_names = ['Reward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']
...
if make_plot and neural_trials:
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
```

iii. The notes show the script evolved through debugging and visualization, so some helper computations remained even after the final export format stabilized.
