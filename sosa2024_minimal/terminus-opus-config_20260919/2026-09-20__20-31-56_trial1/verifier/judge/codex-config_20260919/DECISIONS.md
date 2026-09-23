# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates every NWB file one directory below `/app/data`, processes each file as one session, writes an intermediate session pickle, and assembles all valid session pickles into one dataset. It reads NWB/HDF5 datasets directly with `h5py` and parallelizes session conversion with spawned workers.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
with h5py.File(fn, 'r') as f:
    ...
with ctx.Pool(nproc, maxtasksperchild=1) as p:
    for _ in p.imap_unordered(process_session, files):
        pass
```

iii. The trajectory reports 152 files/sessions and 11 mice. The agent chose this pattern after inspecting the directory and NWB structure, and used multiprocessing because processing the full 87 GB source set was I/O- and compute-intensive.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file's `general/subject/subject_id`. During assembly, unique IDs are accumulated in first-seen order and each session receives the corresponding `subject_idx`.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
if m['subject'] not in subjects:
    subjects.append(m['subject'])
data['subject_idx'].append(subjects.index(m['subject']))
```

iii. The agent inspected NWB metadata and verified that the complete collection contains 11 mice, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The filename, `session_id`, scene, subject, and other details are retained in session metadata; sessions with fewer than two retained trials are skipped at assembly.

ii.
```python
session_id = f['general/session_id'][()].decode()
res = dict(neural=neural, input=inp, output=out, meta=meta)
if len(r['neural']) < 2:
    continue
```

iii. The agent inferred this from the file organization and session metadata. All 152 processed sessions ultimately passed the two-trial requirement.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples in `trial_start`; trial stops are positive samples in `teleport`. Each retained trial uses the half-open slice `[start:stop)`, excluding the teleport sample and inter-trial interval.

ii.
```python
starts = np.where(trial_start > 0)[0]
stops = np.where(teleport > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
...
neural.append(events[:, s:e])
```

iii. The agent traced the paper's preprocessing and concluded that a lap is `trial_start` to `teleport`, with teleport periods excluded. It also checked trial positions and boundaries across sessions.

## 1-e. How are trials filtered based on quality controls?

i. It removes the first trial of every session because previous-trial outcome is considered undefined, and removes trials where more than 30% of samples have raw cumulative lick values greater than 2. It does not apply the reference solution's minimum-50-timepoint filter.

ii.
```python
lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH
...
if t == 0 or tr['lick_err']:
    continue
```

iii. The lick rule was taken from the paper's lick-sensor correction and produced exactly 81 rejected trials, matching the paper. Dropping trial zero was justified as avoiding an invented value for an undefined previous outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB `Fluorescence` (F) and `Neuropil` (Fneu) ROI response series, not the stored `Deconvolved` series. Planes are concatenated, and the ROI `iscell` field supplies initial cell curation.

ii.
```python
F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
iscell = ps['iscell'][:, 0] > 0
```

iii. The agent determined that NWB `Deconvolved` is Suite2p's signal from raw fluorescence, whereas the paper computes and deconvolves its own dF/F signal.

## 2-b. How is the `neural` data processed?

i. Within every trial it subtracts `0.7*Fneu`, adds back `0.7` times trial-mean neuropil, computes a Gaussian-smoothed/maximin baseline (sigma 15, 300-sample minimum then maximum), calculates dF/F, smooths it with sigma 2, and OASIS-deconvolves it with tau 0.7. Unlike the reference, it never lets a baseline window span teleport periods.

ii.
```python
f = f - NEU_COEF * fneu + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)
flow = gaussian_filter1d(f, BASELINE_SMOOTH, axis=1)
flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
d = gaussian_filter1d((f - flow) / np.abs(flow), DFF_SMOOTH, axis=1)
events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
```

iii. The agent copied parameters and operations from the paper repository and Methods. It interpreted `keep_teleports=False` as the paper's operative default; this missed the reference's session-specific teleport-imaging metadata.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps Suite2p-curated `iscell` ROIs and removes cells whose in-trial dF/F has Pearson correlation greater than 0.5 with speed. Correlations at or below 0.5, including negative correlations, are retained.

ii.
```python
F = F[iscell]
Fneu = Fneu[iscell]
...
keep_cells = r <= INT_R_THRESH
events = events[keep_cells]
```

iii. Both filters were identified in the Methods/paper code. The resulting 0.29% removal rate was compared with the paper's reported putative-interneuron rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays begin at the `trial_start` index, so column zero is the trial-start alignment event. No interpolation or shifting is applied.

ii.
```python
s, e = tr['s'], tr['e']
neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
```

iii. The agent found imaging and behavioral series to be frame-aligned and therefore used identical slices for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native behavioral/imaging sampling is retained, approximately 64.5 ms per sample (about 15.5 Hz). No rebinning or resampling is performed; the reported dataset bin size is the mean session timestamp interval.

ii.
```python
dt = float(np.median(np.diff(tstamps)))
fs = 1.0 / dt
...
time_bin_size=float(np.mean(dts)) * 1000.0
```

iii. The trajectory states that all recordings become approximately 15.5 Hz after accounting for planes, and native resolution was kept to match the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` time-series timestamps, indirectly through their median interval and the number of samples in the trial.

ii.
```python
tstamps = b['position/timestamps'][()]
dt = float(np.median(np.diff(tstamps)))
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. Inspection showed behavior and imaging at a shared regular frame rate, so the agent used a regular zero-based time vector.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The code multiplies integer sample offsets `0..T-1` by the session's median timestamp difference.

ii.
```python
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. This makes every trial start at exactly zero while preserving native temporal spacing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly `e-s` entries, matching neural columns sliced over `[s:e)`. Alignment is by shared sample index rather than explicit timestamp interpolation.

ii.
```python
T = e - s
neural.append(events[:, s:e])
time_s = np.arange(T, dtype=np.float32) * dt
```

iii. The agent verified that NWB behavior and neural data are already aligned at the imaging frame rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment` series.

ii.
```python
env = get('environment')
env=int(np.round(np.median(env[s:e])))
```

iii. The agent inspected the variable and found it constant within trials and binary as required.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial median is rounded and cast to integer, then repeated at every timepoint in the trial.

ii.
```python
env=int(np.round(np.median(env[s:e])))
np.full(T, tr['env'], dtype=np.float32)
```

iii. Median aggregation defensively enforces a per-trial value while preserving the observed binary environment coding.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index over trial boundaries derived from `trial_start` and `teleport`, not the NWB `trial number` series.

ii.
```python
for t in range(n_trials):
    ...
    np.full(T, t, dtype=np.float32)
```

iii. The agent followed the boundary-derived sequential trial definition; its exploration found inconsistencies between stored trial-number values and boundary signals.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond repeating the zero-based trial index across all trial samples. Indices retain gaps when earlier trials are filtered.

ii.
```python
np.full(T, t, dtype=np.float32)
```

iii. This represents original within-session trial order, including the paper's switch at index 30.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from timestamps in the behavior `Reward` series, mapped into the common position/behavior timestamp indices.

ii.
```python
reward_times = b['Reward/timestamps'][()]
reward_idx = np.searchsorted(tstamps, reward_times)
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
```

iii. Reward has event timestamps rather than a frame-aligned binary series, so the agent maps events to trial index ranges.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Every trial is marked rewarded if any mapped reward event falls within it. For retained trial `t`, the binary outcome from original trial `t-1` is repeated across time. Trial zero is dropped.

ii.
```python
np.full(T, trials[t - 1]['rewarded'], dtype=np.float32)
```

iii. The instructions require rewarded versus omitted for the immediately preceding trial. The agent dropped the first trial because that value has no predecessor.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance uses raw `position` and a reward-zone label inferred from the NWB identifier/scene name. Zone ranges are A=80–130, B=200–250, and C=320–370 cm; switch scenes change label at trial index 30. The recorded `reward_zone` series is used only as a sanity check.

ii.
```python
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
lo, hi = ZONES[tr['zone']]
```

iii. The agent traced the paper's `get_reward_zones(change_trial=30)` behavior and verified inferred zones against recorded zone-entry positions, finding zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the zone, position minus the lower edge before it, and position minus the upper edge after it.

ii.
```python
d = np.zeros(T)
d[p < lo] = p[p < lo] - lo
d[p > hi] = p[p > hi] - hi
```

iii. This implements distance to any point in the zone and preserves direction relative to the zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It explicitly assigns seven integer categories with boundaries at -50, -10, 0, 10, and 50 cm, leaving exact zero as class 3.

ii.
```python
rd = np.full(T, 3, dtype=np.int64)
rd[d < -50] = 0
rd[(d >= -50) & (d < -10)] = 1
rd[(d >= -10) & (d < 0)] = 2
rd[(d > 0) & (d <= 10)] = 4
rd[(d > 10) & (d <= 50)] = 5
rd[d > 50] = 6
```

iii. The explicit masks were chosen to reproduce the decoder task's stated categories, especially its special exact-zero category.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the same `[s:e)` indices; the derived distance therefore has one entry per neural timepoint.

ii.
```python
neural.append(events[:, s:e])
p = pos[s:e]
```

iii. The agent verified the common frame-level alignment and performed behavioral sanity checks after conversion.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` series.

ii.
```python
pos = get('position')
p = pos[s:e]
```

iii. The NWB description and observed values identify this as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped into `[0, 450)` before discretization, so small out-of-track values are absorbed into endpoint categories.

ii.
```python
pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6),
                   [90.0, 180.0, 270.0, 360.0])
```

iii. The agent treated 450 cm as the fixed track length and used clipping to guarantee valid five-class outputs.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` uses boundaries 90, 180, 270, and 360 cm, yielding five equal-width categories.

ii.
```python
np.digitize(..., [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
```

iii. Five bins of 90 cm exactly span the instructed 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both use the identical `[s:e)` trial slice, with no interpolation.

ii.
```python
neural.append(events[:, s:e])
p = pos[s:e]
```

iii. The agent found neural and behavior samples already synchronized.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` series.

ii.
```python
lick = get('lick')
lk = lick[s:e]
```

iii. Data inspection and the NWB description identify this as the lick sensor/cumulative lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Samples greater than zero are binarized to one. Before conversion, trials with widespread lick-sensor errors are removed.

ii.
```python
lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH
...
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. Binary thresholding meets the requested no/yes output, and trial removal follows the paper's handling of unusable lick recordings.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural samples are sliced with the same trial indices.

ii.
```python
neural.append(events[:, s:e])
lk = (lick[s:e] > 0).astype(np.int64)
```

iii. The common NWB frame grid makes additional alignment unnecessary.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived primarily from the scene string in the NWB identifier, together with trial index 30 for switch sessions. The raw `reward_zone` and `position` series only validate the assignment.

ii.
```python
scene = ident.split('/')[-1]
z0, z1 = scene_zones(scene)
zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
```

iii. The agent located the same scene naming and fixed change-trial logic in the paper code and observed zero validation mismatches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene text is parsed into pre/post-switch zone letters; letters are mapped A/B/C to 0/1/2 and repeated across each trial.

ii.
```python
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}
np.full(T, ZONE_IDX[tr['zone']], dtype=np.int64)
```

iii. This provides the categorical per-trial output required by the task.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from the timestamps of the behavior `Reward` event series and the position-series timestamp grid.

ii.
```python
reward_times = b['Reward/timestamps'][()]
reward_idx = np.searchsorted(tstamps, reward_times)
```

iii. The agent found that rewards are event-timestamped rather than provided as an aligned binary vector.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is one if any reward insertion index lies in `[s,e)`, otherwise zero; this scalar is repeated across all trial timepoints.

ii.
```python
rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
...
np.full(T, tr['rewarded'], dtype=np.int64)
```

iii. This directly implements rewarded versus omitted trials and yielded the expected omission rate during exploration.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. When imaging and behavior lengths differ, F and Fneu are truncated to the common length. Non-finite dF/F is converted to zero. Trial boundary counts/order are asserted, zone labels are checked against observed entries, lick-error trials are excluded, and sessions with fewer than two retained trials are skipped.

ii.
```python
n_common = min(F.shape[1], len(pos))
F = F[:, :n_common]
Fneu = Fneu[:, :n_common]
d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
assert len(starts) == len(stops) and np.all(stops > starts)
```

iii. The agent discovered a few one-frame imaging surpluses in two-plane sessions and patched them by truncation. Other checks were added after a full survey and conversion validation.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large F/Fneu arrays, per-trial filtering and OASIS deconvolution, serializing the roughly 9.5 GB result, and later loading/training on it dominate. Full conversion is parallelized by session.

ii.
```python
dff, events = compute_events(F, Fneu, starts, stops, fs)
...
with ctx.Pool(nproc, maxtasksperchild=1) as p:
    for _ in p.imap_unordered(process_session, files):
        pass
```

iii. The trajectory includes timing/profiling: a first multiprocessing attempt stalled, direct session conversion was fast, and switching to spawned workers allowed all 152 sessions to finish.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop marking in-trial samples can be replaced by interval/range construction, the zone-label list can be vectorized, and some per-trial reward/quality calculations could be computed over full-session arrays. The trial loop that creates variable-length output arrays is naturally retained; dF/F must remain trial-local.

ii.
```python
for s, e in zip(starts, stops):
    in_trial[s:e] = True
...
for t in range(n_trials):
    ...
```

iii. The trajectory does not explicitly discuss vectorization. Its focus was paper fidelity and session-level parallelism; trial-local baselines and ragged trial outputs constrain useful vectorization.

## 13-c. What processing does the code repeat multiple times?

i. It loops over trials once in `compute_events`, once to mark in-trial samples, once to build trial metadata/quality flags, and once to build final arrays. It also calculates reward-zone validation information that does not affect accepted outputs.

ii.
```python
for s, e in zip(starts, stops): ...       # compute_events
for s, e in zip(starts, stops): ...       # in_trial mask
for t in range(n_trials): ...             # metadata
for t in range(n_trials): ...             # arrays
```

iii. No explicit justification was recorded. The separation makes the stages readable and allows the cell filter to be computed before materializing per-trial neural arrays.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes dF/F for every curated cell, then discards dF/F after interneuron detection; it deconvolves cells that are subsequently classified as interneurons; it computes zone mismatch diagnostics retained only in metadata; and it creates per-session intermediate pickle files after assembly.

ii.
```python
dff, events = compute_events(...)
...
events = events[keep_cells]
del dff, d_, d_c
...
n_zone_mismatch += 1
```

iii. dF/F is required to reproduce the paper's interneuron criterion, so only its post-filter storage would be unnecessary. The diagnostics and intermediates supported verification, recovery, and parallel assembly, though downstream decoding does not use them.
