# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively glob-selects every subject NWB file, processes sessions in a spawned multiprocessing pool, and assembles successful results. It uses `h5py`, not `pynwb`. The completed run found and converted all 152 files.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
jobs = [(f, i < show_n, args.neural_signal) for i, f in enumerate(files)]
with ctx.Pool(args.nproc, maxtasksperchild=1) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
        if res is not None:
            results.append(res)
```

iii. The agent found 11 subject folders and 152 released NWB sessions. It chose direct HDF5 access for speed and parallelized session conversion; the final log confirms 152 converted sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, unique subjects are naturally sorted numerically, and each session receives an index into that list.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({r['subject'] for r in results},
                  key=lambda s: int(re.sub(r'\D', '', s)))
'subject_idx': np.array([subjects.index(r['subject']) for r in results])
```

iii. The agent verified 11 mice and used NWB metadata rather than relying only on directory names.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. Converted results are sorted by subject and NWB session ID.

ii.
```python
session_id = f['general/session_id'][()].decode()
results.sort(key=lambda r: (r['subject'], r['session_id']))
'neural': [r['neural'] for r in results]
```

iii. File organization and session metadata both support this mapping; 152 sessions were retained.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and ends are positive `teleport` samples. Each trial is the half-open interval `[start, teleport)`, excluding the teleport frame.

ii.
```python
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
for i, (s, e) in enumerate(zip(starts, stops)):
    act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. The agent observed that the teleport frame has an interpolated position and identified this half-open window with the reference trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped when more than 30% of their frames have cumulative lick count above 2. A session would be skipped if fewer than two trials remained. No speed or minimum-duration filter is applied.

ii.
```python
lick_bad = np.array([(beh['lick'][s:e] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for s, e in zip(starts, stops)])
if lick_bad[i]:
    continue
if len(neural) < 2:
    return None
```

iii. This reproduces the paper's 30% lick-sensor rule and removed 81 trials. Low-speed samples were retained because speed is an output and contiguous temporal alignment is required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from suite2p `Fluorescence` (F), `Neuropil` (Fneu), ROI `iscell`, and `planeIdx`; the stored NWB `Deconvolved` series is not used.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)
Fs.append(f[f'processing/ophys/Fluorescence/plane{p}/data'][:].T)
Fns.append(f[f'processing/ophys/Neuropil/plane{p}/data'][:].T)
return F[iscell], Fneu[iscell], plane_idx[iscell]
```

iii. The agent correctly distinguished suite2p's stored deconvolution from the paper pipeline starting with raw F/Fneu.

## 2-b. How is the `neural` data processed?

i. F is neuropil-corrected with coefficient 0.7; per-trial mean neuropil is restored; a sigma-15 Gaussian plus 300-frame minimum/maximum filter supplies the maximin baseline; dF/F is calculated and sigma-2 smoothed. OASIS events are computed, but the full run's default and saved signal are dF/F, not events. Baselines are always trial-restricted rather than using the reference's session-dependent `keep_teleports` rule.

ii.
```python
f_ -= NEU_COEF * fneu_
f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
seg = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=-1)
seg = ndimage.minimum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
seg = ndimage.maximum_filter1d(seg, BASELINE_FILTER_WIN, axis=-1)
dff[:, mask] = (f_[:, mask] - flow[:, mask]) / np.abs(flow[:, mask])
dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. The agent said both products belong to the reference pipeline but selected dF/F because its controlled decoder comparison was better on 5/6 targets. It considered teleport handling irrelevant because saved samples are within trials, an assertion that overlooks baseline-window effects documented by the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only manually curated `iscell` ROIs are loaded, then cells with Pearson correlation `r(dF/F, speed) > 0.5` over within-trial samples are excluded as putative interneurons.

ii.
```python
keep_cells = r_speed <= INTERNEURON_R_THRESH
dff_kept = dff[keep_cells]
```

iii. Both filters are explicitly described in the Methods; 402 cells (0.29%) were excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are sliced at `trial_start` and time zero is the first sample of that slice; there is no resampling.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    act = np.ascontiguousarray(activity[:, s:e], dtype=np.float32)
```

iii. Behavior is already interpolated to imaging frames, so common indices align neural data to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native per-plane imaging resolution is retained: 15.5078125 Hz or 64.4836 ms per bin. No temporal rebinning is performed.

ii.
```python
FRAME_RATE = 15.5078125
'time_bin_size': 1000.0 / FRAME_RATE
```

iii. The paper and behavior timestamps use approximately 15.5 Hz; retaining it preserves exact alignment.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the NWB `position` timestamps, stored as `beh['t']`.

ii.
```python
beh['t'] = b['position']['timestamps'][:]
tt = beh['t'][s:e] - beh['t'][s]
```

iii. All behavior streams were verified to share imaging-frame timing.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial.

ii.
```python
inp[0] = beh['t'][s:e] - beh['t'][s]
```

iii. This yields zero at trial start and advances by about 64.48 ms per frame.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[s:e]` frame slice and is checked to have the same time dimension as neural and output arrays.

ii.
```python
assert a.shape[1] == ii.shape[1] == oo.shape[1]
```

iii. The released behavioral data are already aligned to imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is the median of the raw per-frame `environment` stream within each trial. Scene-derived environments are calculated only as a cross-check.

ii.
```python
env_stream = np.array([np.median(beh['environment'][s:e])
                       for s, e in zip(starts, stops)])
inp[1] = env_stream[i]
```

iii. The stream is constant per trial, takes 0/1, and matched the scene-derived environment for every trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The within-trial median is taken and broadcast across all trial frames.

ii.
```python
inp[1] = env_stream[i]
```

iii. Median aggregation is robust and preserves the required binary ENV1/ENV2 value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index over trial-start/teleport pairs, not the raw `trial number` stream.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    inp[2] = i
```

iii. The original within-session index is retained even if an earlier lick-error trial is removed.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond broadcasting the zero-based original trial index across time.

ii.
```python
inp[2] = i
```

iii. This supplies the specified continuous per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Sparse `Reward` timestamps are mapped onto behavior frames, then combined with the raw `reward_zone` stream to define whether each trial was rewarded.

ii.
```python
idx = np.searchsorted(beh['t'], rew_t)
rew_bin[idx] = 1
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0)
                     for s, e in zip(starts, stops)], dtype=np.int64)
```

iii. This mirrors the paper's `get_trial_types` reward definition.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial receives the preceding original trial's binary outcome, broadcast over time. The first recorded trial is assigned 1, unlike the human reference's 0.

ii.
```python
inp[3] = isreward[i - 1] if i > 0 else 1
```

iii. The agent reasoned that 30 warm-up trials preceded imaging and were likely rewarded. That history is not present in the supplied data, so this is an assumption rather than an observed previous outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Position comes from the raw `position` stream. The applicable zone is parsed from the NWB identifier/scene name, with a switch after trial 30; fixed bounds are A 80–130, B 200–250, C 320–370 cm.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
zone = REWARD_ZONES[zones[i]]
dist = signed_distance_to_zone(pos, zone)
```

iii. This directly mirrors `behavior.get_reward_zones` and was validated against reward positions. The human converter instead infers zones from `reward_zone` and position with Viterbi smoothing.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is negative before the zone, zero inside, and positive after, measured to the nearest zone boundary.

ii.
```python
d[before] = pos[before] - lo
d[after] = pos[after] - hi
```

iii. This implements “distance to any location in the reward zone” more literally than distance from zone start.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit boolean masks implement the seven requested categories, with exact zero isolated as class 3.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[d == 0] = 3
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The edges follow the task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the same `[s:e]` imaging-frame slice.

ii.
```python
pos = beh['position'][s:e]
act = activity[:, s:e]
```

iii. No interpolation is needed because behavior was aligned before NWB release.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position`.

ii.
```python
pos = beh['position'][s:e].astype(np.float64)
```

iii. The stream records corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm, floored, and clipped into indices 0–4.

ii.
```python
np.clip(np.floor(pos / (TRACK_LENGTH / 5.0)), 0, 4).astype(np.int64)
```

iii. Clipping absorbs small values outside the nominal 0–450 cm range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins are used: <90, 90–180, 180–270, 270–360, and >360 cm.

ii.
```python
out[1] = bin_position(pos)
```

iii. This matches the requested five equal track bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and activity are indexed by the identical trial boundaries.

ii.
```python
pos = beh['position'][s:e]
act = activity[:, s:e]
```

iii. Shape assertions verify framewise alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw per-frame `lick` behavior stream.

ii.
```python
licks = beh['lick'][s:e].astype(np.float64)
```

iii. The agent identified this as a cumulative per-frame lick count.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count becomes 1; otherwise 0. Sensor-error trials are removed before output creation.

ii.
```python
out[3] = (licks > 0).astype(np.int64)
```

iii. Binarization matches the requested no/yes output, while removal avoids invalid lick trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks use the same `[s:e]` frame slice as neural data.

ii.
```python
licks = beh['lick'][s:e]
act = activity[:, s:e]
```

iii. Both streams share imaging-frame indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene name in the NWB `identifier`, not directly from the raw `reward_zone` stream.

ii.
```python
scene = f['identifier'][()].decode().split('/')[-1]
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
```

iii. The agent followed the paper repository's scene-based `get_reward_zones` logic and validated it against observed rewards.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene patterns are parsed; switch scenes change zone after 30 trials; A/B/C map to 0/1/2 and are broadcast over time.

ii.
```python
n1 = min(CHANGE_TRIAL, ntrials)
return ([z1] * n1 + [z2] * (ntrials - n1), ...)
out[4] = ZONE_TO_IDX[zones[i]]
```

iii. This matches the experiment's fixed switch-trial rule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse `Reward` timestamps and the per-frame `reward_zone` stream.

ii.
```python
rew_t = b['Reward']['timestamps'][:]
isreward = np.array([(beh['reward'][s:e].sum() > 0) and
                     (beh['reward_zone'][s:e].sum() > 0) ...])
```

iii. This is the reference paper code's definition of a rewarded trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are assigned to the next behavior-frame insertion index. A trial is 1 only if it contains both a reward event and reward-zone entry; the value is broadcast over the trial.

ii.
```python
idx = np.searchsorted(beh['t'], rew_t)
rew_bin[idx] = 1
out[5] = isreward[i]
```

iii. The binary definition mirrors `get_trial_types`; the agent checked a reward rate near the expected 85%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior/imaging length differences of at most five frames are truncated to their common length; trials cut by truncation are dropped. Failed sessions are loudly reported but skipped, sessions with fewer than two usable trials are skipped, reward indices are clipped, nonfinite saved arrays are rejected, and extensive assertions check dimensions. The final repaired run retained all sessions.

ii.
```python
nframes = min(n_beh, n_ophys)
assert abs(n_beh - n_ophys) <= 5
beh[k] = beh[k][:nframes]
F = F[:, :nframes]
valid = stops < nframes
starts, stops = starts[valid], stops[valid]
assert np.all(np.isfinite(act)) and np.all(np.isfinite(inp))
```

iii. A first full run exposed ten one-frame mismatches; the agent connected this to the reference's one-frame correction, fixed it, reran, and verified all 152 sessions.

## 13-a. What are the most time-consuming steps of the code?

i. OASIS deconvolution dominates most sessions, followed by dF/F filtering for large sessions, NWB array loading, and writing the 9.63 GB pickle.

ii.
```python
timings['load'] = time.time() - t0
timings['dff'] = time.time() - t1
timings['deconv'] = time.time() - t1
pickle.dump(data, fh, protocol=4)
```

iii. Measured per-session timings showed roughly 3.5–6 seconds for OASIS and lower loading/filtering costs; eight workers reduced full conversion to about 3.3 minutes.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in dF/F masking, baseline filtering, OASIS, trial summaries, and output assembly remain. The cellwise interneuron correlation was already vectorized. Variable trial lengths make full vectorization awkward, though interval masks and some full-session output transforms could be vectorized.

ii.
```python
for s, e in zip(starts, stops):
    f_[:, s:e] = F[:, s:e]
for i, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The agent explicitly replaced a neuron loop with a matrix product and considered the remaining trial loops a natural representation of variable-length data.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trial intervals repeatedly for masking, baseline construction, smoothing, deconvolution, interneuron masking, reward/lick summaries, per-trial conversion, plotting, and sanity checks. It also computes OASIS events even when the default saved signal is dF/F.

ii.
```python
for s, e in zip(starts, stops): ...  # repeated in compute_dff/deconvolve/convert_session
events = deconvolve(dff_kept, starts, stops)
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
```

iii. The agent prioritized fidelity, diagnostics, and selectable neural signals; it did not emphasize that default dF/F runs still pay the full deconvolution cost.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default full run, OASIS events are computed solely to obtain a shape/count and are then discarded because dF/F is saved. Scene-derived environment values are retained only diagnostically; plane indices and several session diagnostics are reduced to metadata. Plotting is optional.

ii.
```python
events = deconvolve(dff_kept, starts, stops)
activity = events if neural_signal == 'events' else np.nan_to_num(dff_kept)
zones, envs_scene = zone_per_trial(scene, ntrials_raw)
```

iii. The events option supported a controlled comparison and optional alternate output, but computing it unconditionally is unnecessary for the chosen default dataset and is the main avoidable cost.
