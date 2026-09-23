# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively gathers every NWB one directory below `/app/data/sub-*`, sorts the paths, and processes each file as one session, normally in spawned worker processes. Each file is opened with `pynwb.NWBHDF5IO`; behavioral and ophys arrays are read through the pynwb object model.

ii.
```python
files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))
with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes say this covers 152 NWB files from 11 subject directories and obeys the explicit pynwb-only requirement. Parallelism was added because session conversion is independent.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB's `nwb.subject.subject_id`; after processing, unique IDs are naturally sorted by mouse number and every session gets an index into that list.

ii.
```python
subject = nwb.subject.subject_id
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The agent reports 11 unique mice and uses NWB metadata rather than relying only on path parsing.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; `session_id` is read from NWB metadata. Sessions with fewer than two retained trials would be removed during assembly.

ii.
```python
session_id = nwb.session_id
results = [r for r in results if len(r['neural']) >= 2]
```

iii. This follows the file organization and the target requirement of at least two trials per session; the notes say all 152 sessions passed.

## 1-d. How are the data split into trials?

i. Trial starts are samples where `trial_start > 0`; stops are samples where `teleport > 0`. Trial arrays use the half-open slice `[start, stop)`, excluding the teleport/ITI sample.

ii.
```python
starts = np.where(g('trial_start') > 0)[0]
stops = np.where(g('teleport') > 0)[0]
for i, (s, e) in enumerate(zip(starts, stops)):
    ev = events[:, s:e]
```

iii. The notes justify this as the on-track lap definition used by the paper; position runs from about 0 to 450 cm and fluorescence was commonly blanked during ITIs.

## 1-e. How are trials filtered based on quality controls?

i. The first trial is dropped because previous outcome is undefined. The code also drops lick-sensor-error trials (>30% of samples with lick count >2), trials shorter than 10 frames, trials containing non-scanning frames, and trials with non-finite neural/position/speed/lick values. Sessions with fewer than two survivors are dropped.

ii.
```python
if i == 0: continue
if lick_error[i]: continue
if (e - s) < MIN_TRIAL_FRAMES: continue
if np.any(scanning[s:e] != 1): continue
if np.any(~np.isfinite(ev)) or np.any(~np.isfinite(p)) or np.any(~np.isfinite(sp)) or np.any(~np.isfinite(lk)): continue
```

iii. The agent says the first trial cannot supply the requested previous-trial value, the lick rule reproduces the paper's 81 bad trials, and the remaining checks prevent invalid decoder arrays. This is stricter than the paper/reference, which NaNs bad lick data for licking analyses rather than discarding whole trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is derived from per-plane raw `Fluorescence` and `Neuropil`, with ROI metadata `iscell` and `planeIdx`; speed is additionally used for interneuron filtering. Although OASIS events are computed, the actual full run used the default `dff`, so saved neural data are processed dF/F.

ii.
```python
Fp = np.asarray(oph['Fluorescence'].roi_response_series[name].data[:]).T
Np = np.asarray(oph['Neuropil'].roi_response_series[name].data[:]).T
keep = iscell[plane_idx == plane]
if signal == 'dff':
    events = dff
```

iii. The notes correctly distinguish these traces from the NWB `Deconvolved` field. In Step 12 the agent explicitly chose dF/F because it improved decoder accuracy, despite earlier notes saying paper-like events would be used.

## 2-b. How is the `neural` data processed?

i. Across each trial, the code subtracts `0.7*Fneu`, adds back the trial neuropil mean, computes a maximin baseline (Gaussian sigma 15, minimum then maximum filters of 300 frames), forms `(F-baseline)/abs(baseline)`, and smooths with sigma 2. It also computes OASIS (`tau=.7`, 15.5078 Hz), but the default/full run replaces those events with dF/F before saving. Planes are pooled.

ii.
```python
f_ -= NEU_COEF * fneu_
f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
seg = gaussian_filter1d(f_[:, s:e], BASELINE_SIGMA, axis=-1)
seg = maximum_filter1d(minimum_filter1d(seg, BASELINE_WIN, axis=-1), BASELINE_WIN, axis=-1)
dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])
dff[:, s:e] = gaussian_filter1d(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=-1)
events[:, s:e] = dcnv.oasis(...)
```

iii. Most processing reproduces the paper's pipeline. The final dF/F choice was an accuracy-driven deviation; the human reference instead saves OASIS events and also honors session-specific `keep_teleports` baseline windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only manually curated `iscell` ROIs are loaded. Cells with Pearson correlation `r(dF/F,speed)>0.5` are excluded as putative interneurons, and zero-variance/non-finite-correlation ROIs are also dropped.

ii.
```python
keep = iscell[plane_idx == plane]
r_speed = (dff_c @ sp_c) / denom
keep_cells = ~(r_speed > SPEED_CORR_THR)
keep_cells &= np.isfinite(r_speed)
```

iii. The first two rules follow the Methods. The finite-correlation rule is a defensive extra to prevent dead ROIs entering the decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is implicit: neural data are sliced starting at each `trial_start`, so column zero is the trial-start frame; trials end before `teleport`.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    ev = events[:, s:e]
```

iii. The agent notes that behavior was already interpolated onto the imaging-frame clock, so no resampling or shifting was required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 15.5078125 Hz frames are retained: 64.48 ms per bin, with no temporal rebinning.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. The paper's analyses operate at imaging-frame resolution, and the notes report identical behavior-frame spacing.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It comes from the timestamps attached to the behavior `position` series and the trial-start indices.

ii.
```python
t = np.asarray(beh['position'].timestamps[:])
tt = t[s:e] - t[s]
```

iii. The position timestamps are treated as the common behavior/imaging clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from every timestamp in that trial.

ii.
```python
tt = t[s:e] - t[s]
inp[0] = tt
```

iii. This makes each variable-duration trial begin at exactly zero seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `[s:e]` indices are used for timestamps and neural columns. If imaging and behavior lengths differ, both are truncated to their common total length first.

ii.
```python
n_frames = min(F.shape[1], len(t))
ev = events[:, s:e]
tt = t[s:e] - t[s]
```

iii. The notes say multi-plane files occasionally have one extra imaging frame and that truncation occurs after the final teleport, losing no trial data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii.
```python
env = g('environment')
env_vals = np.unique(env[s:e])
```

iii. The data use 0 for ENV1 and 1 for ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. A per-trial scalar is taken from the unique value; if unexpectedly nonconstant, the rounded median is used. It is broadcast over every timepoint.

ii.
```python
env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
inp[1] = env_trial[i]
```

iii. The notes observed environment is constant within normal trials; the median branch is defensive.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index of trials defined by `trial_start` and `teleport`, not the stored trial-number stream.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    inp[2] = i
```

iii. This matches the within-session sequential trial interpretation used by the reference.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform is applied; the scalar index is broadcast across the trial.

ii.
```python
inp[2] = i
```

iii. The agent deliberately preserves original trial indices even when some trials are dropped.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives reward delivery from `Reward.timestamps` and checks the `reward_zone` stream for in-zone occupancy in the preceding trial.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
reward_idx = np.searchsorted(t, reward_times)
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
```

iii. This mirrors the paper's `get_trial_types`: reward is valid when delivered during reward-zone occupancy.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are insertion-mapped to the common clock; each current trial receives the binary outcome of original trial `i-1`. The first trial is discarded rather than assigned zero.

ii.
```python
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
if i == 0: continue
inp[3] = rewarded[i - 1]
```

iii. Dropping the first trial avoids inventing an outcome. Importantly, after a bad intermediate trial is dropped, the next retained trial still references the true immediately preceding experimental trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavior `position` plus zone bounds A/B/C. Zone identity is parsed from `nwb.identifier`'s scene name, with switches after trial 30; `reward_zone` is used only as a validation signal.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = scene_reward_zones(scene, n_trials_raw)
p = pos[s:e]
z0, z1 = REWARD_ZONES[zlab]
```

iii. The notes say the scene-based method matches the paper code and agreed with recorded occupancy on every rewarded trial, while also covering omission trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the zone, position minus zone start before it, and position minus zone end after it.

ii.
```python
d = np.zeros_like(pos)
d[pos < zone_start] = pos[pos < zone_start] - zone_start
d[pos > zone_end] = pos[pos > zone_end] - zone_end
```

iii. This follows the task's “distance to any location in the reward zone,” rather than the paper's distance to zone start.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
b = np.full(d.shape, 3, dtype=np.int64)
b[(d < 0) & (d >= -10)] = 2
b[(d < -10) & (d >= -50)] = 1
b[d < -50] = 0
b[(d > 0) & (d <= 10)] = 4
b[(d > 10) & (d <= 50)] = 5
b[d > 50] = 6
```

iii. The masks directly implement the requested boundary semantics and avoid the human reference's `1e-6` approximation around zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural use the same trial slice `[s:e]`, producing equal-length columns.

ii.
```python
ev = events[:, s:e]
p = pos[s:e]
out[0] = discretize_distance(d)
```

iii. No interpolation is needed because behavior was exported on the imaging clock.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is directly derived from behavior `position`.

ii.
```python
pos = g('position')
p = pos[s:e]
```

iii. The NWB variable is already corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is digitized at four 90 cm boundaries; there is no other transform.

ii.
```python
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
out[1] = np.digitize(p, POS_EDGES)
```

iii. Five equal-width bins span the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` yields 0 below 90, 1 for 90–<180, 2 for 180–<270, 3 for 270–<360, and 4 at or above 360 cm.

ii.
```python
out[1] = np.digitize(p, [90.0, 180.0, 270.0, 360.0])
```

iii. This matches the requested five categories (with equality at 360 placed in category 4, as `np.digitize` specifies).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both arrays use exactly `[s:e]` on the common imaging-frame clock.

ii.
```python
ev = events[:, s:e]
p = pos[s:e]
```

iii. The agent's plots and length checks found no temporal offset.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavior `lick` series.

ii.
```python
lick = g('lick')
lk = lick[s:e]
```

iii. The notes characterize this as a cumulative-count-like stream requiring binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero become 1 and all others 0. Trials with the paper-defined sensor error are discarded entirely.

ii.
```python
lick_error[i] = (lick[s:e] > 2).mean() > LICK_ERROR_FRAC
out[3] = (lk > 0).astype(np.int64)
```

iii. Binarization matches the target. Whole-trial removal was chosen because the target cannot store NaN labels, though the reference only invalidates lick data.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays share `[s:e]` indices.

ii.
```python
ev = events[:, s:e]
lk = lick[s:e]
```

iii. Both streams are already on the common frame clock.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene embedded in `nwb.identifier`, trial index, and fixed A/B/C definitions. Recorded `reward_zone` plus position validate the derived labels.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = scene_reward_zones(scene, n_trials_raw)
```

iii. The agent followed the paper's scene-name logic because the recorded zone signal is absent on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex parsing assigns the single scene zone or pre/post-switch zones, changing at index 30; labels map A/B/C to 0/1/2 and are broadcast through each trial.

ii.
```python
labels[change_trial:] = m.group(2)
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
out[4] = ZONE_TO_IDX[zlab]
```

iii. This reproduces `behavior.get_reward_zones` and was checked against all available occupancy observations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward.timestamps` and the behavior `reward_zone` time series.

ii.
```python
reward_times = np.asarray(beh['Reward'].timestamps[:])
rz_series = g('reward_zone')
```

iii. The combination implements the paper's distinction between reward delivery and valid in-zone reward.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped with `searchsorted`; a trial is 1 only if at least one mapped reward index lies in `[s,e)` and reward-zone occupancy occurs. The scalar is broadcast across the trial.

ii.
```python
got_reward = np.any((reward_idx >= s) & (reward_idx < e))
in_zone = np.any(rz_series[s:e] > 0)
rewarded[i] = int(got_reward and in_zone)
out[5] = rewarded[i]
```

iii. The agent cites the reference `get_trial_types` behavior and reports an 84.7% rewarded fraction, consistent with ~15% omissions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. One-frame neural/behavior length mismatches are truncated to the common length; assertions reject mismatched trial boundaries; malformed trials, non-scanning data, and non-finite values are dropped. Nonconstant environment uses rounded median; dead ROIs are dropped. Plot failures only warn.

ii.
```python
n_frames = min(F.shape[1], len(t))
assert len(stops) == n_trials_raw
keep_cells &= np.isfinite(r_speed)
if np.any(scanning[s:e] != 1): continue
if np.any(~np.isfinite(ev)): continue
```

iii. These checks were motivated by observed extra frames and by producing finite, dimensionally valid decoder data; the notes state truncation does not remove trial samples.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large fluorescence/neuropil arrays and the dF/F/OASIS computation dominate; serialization of the large pickle is also material. Session work is parallelized.

ii.
```python
with ProcessPoolExecutor(max_workers=args.nworkers, mp_context=ctx) as pool:
    for i, res in enumerate(pool.map(_worker, tasks)):
```

iii. The notes estimate dF/F plus OASIS at roughly 1.5–2.5 seconds/session and explain that full contiguous HDF5 reads are faster than per-ROI fancy indexing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops used for reward/environment/lick-error summaries, zone validation, and final array construction could be partly vectorized. Per-trial baseline and OASIS loops are harder to remove because trials have variable lengths and processing must remain trial-local.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    got_reward = np.any((reward_idx >= s) & (reward_idx < e))
...
for i, (s, e) in enumerate(zip(starts, stops)):
    out[1] = np.digitize(p, POS_EDGES)
```

iii. The agent says the expensive operations are already vectorized across cells, and variable-length output naturally requires some trial iteration.

## 13-c. What processing does the code repeat multiple times?

i. Trial ranges are traversed repeatedly: to mask/correct F/Fneu, compute baselines and OASIS, summarize trial variables, validate zone identity, and finally construct trials. `compute_dff_and_events` also performs OASIS even when the selected/default saved signal is dF/F.

ii.
```python
for s, e in zip(starts, stops):  # mask
for s, e in zip(starts, stops):  # baseline
for s, e in zip(starts, stops):  # smoothing/OASIS
```

iii. The notes defend trial-local neural processing as reference-required, but do not identify the wasted OASIS work introduced after changing the default to dF/F.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. On the actual dF/F run, it computes the full OASIS event matrix and then discards it by assigning `events = dff`. It also computes `dff_keep` and `plane_of_cell`; the former is used only for optional plots and the latter only for metadata. Zone-mismatch validation and extensive summary concatenations do not affect saved trial values.

ii.
```python
dff, events = compute_dff_and_events(F, Fneu, starts, stops)
if signal == 'dff':
    events = dff
dff_keep = dff[keep_cells]
```

iii. Earlier planning assumed events would be saved, so OASIS was necessary then. After the Step 12 switch to dF/F, the computation remained, apparently for comparison/plotting convenience rather than downstream conversion.
