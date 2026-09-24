# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every NWB file under `/app/data`, opens each with `NWBHDF5IO`, and loads behavior plus the `Deconvolved/plane0` ROI series. Full mode processes all 152 files; sample mode takes the first two.

ii.
```python
def list_sessions(data_root: Path):
    return sorted(data_root.rglob('*.nwb'))
...
with NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True) as io:
    nwb = io.read()
...
files = list_sessions(data_root)
```

iii. The notes say that the archive contains 11 subject folders and 152 session NWBs and justify treating each file as a session. They state that the full archive was retained instead of a paper-analysis subset.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `nwb.subject.subject_id`, with the parent directory name as fallback. Subjects are added on first encounter and each retained session receives its subject index.

ii.
```python
subj = nwb.subject.subject_id if nwb.subject is not None else nwb_path.parent.name.replace('sub-', '')
...
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
subject_idx.append(subject_to_idx[sess['subject']])
```

iii. The notes identify 11 subject folders and report 11 subjects after full conversion.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Sessions with fewer than two retained trials are omitted from the output.

ii.
```python
for j, f in enumerate(files):
    sess = read_session(f)
    ...
    if len(ntr) >= 2:
        neural_all.append(ntr)
```

iii. The agent reasoned that the reference workflow and paper are session-centric and that the archive supplies one NWB per subject-session.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples that also pass validity checks. Each trial runs until the next start, or the end of the recording for the last trial; invalid/scanning-off samples are removed within that interval. The `teleport` stream is loaded but not used as the trial end.

ii.
```python
trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)
segs = contiguous_segments(trial_start_idx, len(t))
...
for i, s in enumerate(starts):
    e = starts[i + 1] if i + 1 < len(starts) else n_time
```

iii. The notes say trials were reconstructed from `trial_start` plus `trial number`, because standard NWB trial tables are absent. They do not justify using the next start rather than teleport transitions.

## 1-e. How are trials filtered based on quality controls?

i. A trial is retained only if at least five samples pass finite-time, nonnegative-trial-number, and positive-scanning checks and at least five retained positions are within 0–450 cm. Sessions need at least two retained trials.

ii.
```python
valid = np.isfinite(t) & (b['trial number'] >= 0)
valid &= (b['scanning'] > 0)
...
if mask.sum() < 5: continue
if np.sum((pos >= 0) & (pos <= 450)) < 5: continue
```

iii. The notes call these vectorized per-trial checks, but do not connect the five-sample threshold or scanning/position filters to the paper. The reference short-trial threshold was not identified.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is read directly from the NWB `ophys/Deconvolved/plane0` series. Other planes, fluorescence, and neuropil are not used.

ii.
```python
deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
```

iii. The agent described this as the most directly usable event-like activity stream and later claimed it was the confirmed reference source. This conflicts with the paper code, which recomputes events from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. No signal processing is applied; a time-by-ROI slice is transposed to ROI-by-time and cast to float32.

ii.
```python
nn = neural[idx, :].T.astype(np.float32)
```

iii. The notes justify the stored deconvolved series as already processed and suitable for decoding. They did not reproduce neuropil correction, per-trial maximin dF/F, smoothing, or OASIS deconvolution from the paper.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not neuron-filtered. Every ROI in `plane0` is retained; `iscell` curation and speed-correlated putative-interneuron exclusion are absent.

ii.
```python
roi_count = deconv.shape[1]
...
brain_region_idx = np.zeros(sess['roi_count'], dtype=np.int64)
```

iii. The agent equated matching the raw archive ROI count with consistency. The notes explicitly report 260,091 converted neurons but never establish paper cell-level quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials begin at the selected `trial_start` frame, and the corresponding neural rows are sliced with the same indices after removing invalid frames.

ii.
```python
idx = np.flatnonzero(mask) + s
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
```

iii. The notes identify `trial_start` as the required alignment event and report an exact first-trial spot check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is done. Metadata uses the median positive difference of behavior timestamps across sessions, in milliseconds.

ii.
```python
dts = np.diff(sess['timestamps'])
dt_list.append(float(np.median(dts)))
...
time_bin = float(np.median(dt_list) * 1000.0)
```

iii. The agent says imaging-frame timestamps form the common time base. It does not account for per-plane sampling rate when using only `plane0`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses the timestamps attached to the `trial_start` behavioral time series.

ii.
```python
frame_timestamps = np.asarray(beh.time_series['trial_start'].timestamps[:], dtype=np.float64)
```

iii. The agent states that behavioral streams share the imaging-frame time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first retained sample’s timestamp is subtracted from every retained timestamp in the trial.

ii.
```python
tt = t[idx] - t[idx[0]]
```

iii. This implements trial-start-relative time; the notes report a range starting at 0 seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time and neural activity use the identical `idx` array, so they have matching columns.

ii.
```python
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
```

iii. The agent reports `np.allclose` spot checks for raw neural and time alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the per-frame `environment` series; the median retained value in each trial is used.

ii.
```python
env = np.asarray(b['environment'][s:e])[mask]
env_per_trial.append(np.nanmedian(env))
```

iii. The mapping plan identifies `environment` as the NWB source and notes that it should encode ENV1 versus ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Unique trial medians within each session are sorted and mapped to 0 and 1, then the code broadcasts the mapped value over all trial samples.

ii.
```python
mapping = {v: i for i, v in enumerate(uniq_sorted[:2])}
...
np.full(len(idx), env_codes[i], dtype=np.float32)
```

iii. The notes say per-trial variables are broadcast for consistent array shapes. No evidence is given that remapping is necessary; for raw 0/1 values it is identity.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The raw `trial number` stream is used only for validity and to compute an unused median list. The saved input is the zero-based index `i` of the retained trial.

ii.
```python
tr = b['trial number'][s:e]
trial_nums.append(int(np.round(np.median(tr_valid))))
...
np.full(len(idx), i, dtype=np.float32)
```

iii. The notes claim trial structure comes from `trial_start` plus `trial number`, but the final value is a sequential within-session retained-trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. It broadcasts the retained-trial loop index over time; skipped trials therefore cause renumbering.

ii.
```python
np.full(len(idx), i, dtype=np.float32)
```

iii. Broadcasting is justified as a consistent representation of a per-trial variable. Renumbering after filtering is not discussed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives current reward outcomes from the sparse `Reward` series aligned onto behavior-frame timestamps, then shifts that retained-trial outcome vector by one.

ii.
```python
rw = np.asarray(b['Reward'][s:e])[mask]
rew_per_trial.append(int(np.any(rw > 0)))
prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)
```

iii. The agent discovered that `Reward` is sparse and documented nearest-frame alignment as a required correction.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Sparse events are assigned to nearest frames; each retained trial is rewarded if any aligned value is positive. The first retained trial is 0 and later trials receive the preceding retained trial’s outcome, broadcast over time.

ii.
```python
idx = np.searchsorted(frame_timestamps, ts_t)
...
out[idx] = data
...
prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)
```

iii. The notes say this fixed sparse-event misalignment. They do not address that after a trial is filtered, “previous” means previous retained trial rather than the actual preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses `position` and the binary/positive `reward_zone` time series. For each trial it takes the minimum and maximum positions at positive reward-zone samples as that trial’s interval.

ii.
```python
m = (rz_vals > 0) & np.isfinite(pos_vals) & (pos_vals >= 0) & (pos_vals <= 450)
lo = float(np.min(pos_vals[m])); hi = float(np.max(pos_vals[m]))
```

iii. The notes say raw reward-zone codes were unsuitable and that inferring per-trial intervals made distributions sensible. They did not use the known A/B/C ranges or the reference’s cross-trial Viterbi correction for missing/noisy annotations.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to 0–450 cm. Distance is negative before the inferred interval, zero within it, and positive after it.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
dist[pos < lo] = pos[pos < lo] - lo
dist[pos > hi] = pos[pos > hi] - hi
```

iii. The agent explains this as signed distance to an inferred reward-zone interval. Clipping is noted in metadata but not tied to the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven requested categories, with exactly zero as class 3.

ii.
```python
out[dist < -50] = 0
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
```

iii. The notes report checking the resulting distribution and a first-trial exact spot check against the same implemented rule.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and reward-zone values are selected by the same retained indices as neural activity, producing one distance category per neural column.

ii.
```python
idx = np.flatnonzero(mask) + s
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. The agent reports a sample distance-bin match check and treats the behavior streams as frame-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral series.

ii.
```python
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. The mapping plan identifies position as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] before categorization.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
```

iii. Metadata explicitly records clipping. The notes do not explain why raw slight out-of-range values should be changed instead of absorbed by open-ended bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Boolean masks create five 90-cm bins with boundaries 90, 180, 270, and 360 cm.

ii.
```python
out[(pos >= 90) & (pos < 180)] = 1
out[(pos >= 180) & (pos < 270)] = 2
out[(pos >= 270) & (pos < 360)] = 3
out[pos >= 360] = 4
```

iii. The agent says this follows the decoder’s five equal track bins and verified a sample against the raw position.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same `idx` selects both neural frames and position samples.

ii.
```python
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. The agent’s raw-data spot check found matching trial lengths and position categories.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the `lick` behavioral time series after generic frame alignment.

ii.
```python
beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. The notes identify lick as the paper’s licking measure and the required binary decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive value becomes 1; all other values, including NaN comparisons, become 0.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. This is justified by the requested no/yes encoding.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The generic helper aligns the full lick series to frame timestamps, and then the same trial indices used for neural data select lick samples.

ii.
```python
beh_data = {k: align_series_to_frame(...) for k in ts_names}
...
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. The agent assumes common imaging-frame timestamps and reports no validator warnings.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses `reward_zone` positivity together with `position` to infer the active interval and its center.

ii.
```python
zlo, zhi, zloc = infer_zone_interval_and_location(rz, pos)
```

iii. The notes say raw zone codes could not directly supply the requested categories, motivating spatial inference.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The interval center is labeled A below 200 cm, B from 200 to below 300 cm, and C at 300 cm or above. If no positive zone sample exists, a ±10-cm interval around the trial’s median position is used. The class is broadcast over time.

ii.
```python
if center < 200: loc = 0
elif center < 300: loc = 1
else: loc = 2
...
np.full(len(idx), rz_codes[i], dtype=np.int64)
```

iii. The agent calls the inferred distributions sensible. It gives no paper-based basis for the fallback or the 200/300 thresholds and does not stabilize noisy/missing trials across the session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse `Reward` behavior series and its event timestamps.

ii.
```python
beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
rw = np.asarray(b['Reward'][s:e])[mask]
```

iii. The notes identify and correct the separate sparse reward-event time base.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward values are placed on nearest behavior frames, and a trial is 1 if any retained aligned reward value is positive, otherwise 0; the result is broadcast over time.

ii.
```python
choose_left = np.abs(frame_timestamps[left] - ts_t) < np.abs(frame_timestamps[idx] - ts_t)
out[idx] = data
...
rew_per_trial.append(int(np.any(rw > 0)))
```

iii. The agent reports exact agreement in a sampled raw reward-outcome check. Unlike the reference, it does not assert a maximum alignment error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/mismatched streams are silently copied and NaN-padded, zero-filled when empty, nearest-frame placed when shorter, or interpolated when longer. Invalid timestamps/trial numbers/scanning-off frames are removed; position is clipped; weak trials and sessions are skipped. Missing reward-zone activity falls back to the median trial position.

ii.
```python
out = np.full(len(frame_timestamps), np.nan, dtype=np.float64)
...
if len(data) == 0:
    return np.zeros(len(frame_timestamps), dtype=np.float64)
...
return np.interp(frame_timestamps, ts_t, data)
```

iii. The agent describes these as defensive alignment and filtering choices and says verification passed. It does not log most repairs or quantify interpolation/padding, and its reward-zone fallback is not reference-derived.

## 13-a. What are the most time-consuming steps of the code?

i. The code times each session and the total conversion. Its notes identify full NWB array extraction and session-by-session processing as the principal cost; the full 25-GB pickle write is also necessarily costly, although not separately timed.

ii.
```python
t0 = time.time()
sess = read_session(f)
...
print(f'processed ... elapsed={time.time()-t0:.2f}s')
```

iii. The agent estimated about 0.5 seconds per sample session and 1–2 minutes overall, while noting that full session arrays are loaded into memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Python loops remain over sessions, trial segments, per-trial summaries, and final per-trial construction. Most calculations inside a trial are vectorized. The two trial passes could be consolidated or some full-session binning could be performed once.

ii.
```python
for (s, e), trn in zip(segs, trial_nums):
    ...
for i, (s, e) in enumerate(kept_segs):
```

iii. The notes claim the implementation avoids per-timepoint loops and uses vectorized per-trial masking; they only generally acknowledge session-by-session iteration.

## 13-c. What processing does the code repeat multiple times?

i. Each behavior stream passes through the same alignment helper. Each trial is traversed once to summarize/filter it and again to build arrays; masks, indices, positions, and reward-zone information are computed again. Full conversion itself does not repeat a separate survey pass.

ii.
```python
for (s, e), trn in zip(segs, trial_nums):
    mask = valid[s:e]
    pos = np.asarray(b['position'][s:e])[mask]
...
for i, (s, e) in enumerate(kept_segs):
    mask = valid[s:e]
    pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. The agent’s notes do not explicitly identify this duplication; they emphasize direct extraction and vectorization.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes raw median trial numbers but only uses their existence, creates unused `env_map` and `rz_map`, loads `teleport` without using it, and optionally creates diagnostic plots. It also aligns all listed streams before deciding whether a session will survive.

ii.
```python
trial_nums.append(int(np.round(np.median(tr_valid))))
env_codes, env_map = map_binary_environment(...)
rz_map = {0: 0, 1: 1, 2: 2}
ts_names = [..., 'teleport', ...]
```

iii. The agent justifies optional plots as sanity checks but does not document the unused mappings, trial-number medians, or unused teleport data.
