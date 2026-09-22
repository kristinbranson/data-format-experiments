# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `/app/data/sub-*/*.nwb`, sorted the paths, and processed each file as one session, normally with 12 spawned workers. Each NWB was opened through `pynwb.NWBHDF5IO`; failed sessions were reported but omitted from the result.

ii.
```python
files = sorted(glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
with NWBHDF5IO(fn, 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes state that this found 152 sessions in 11 subject folders and satisfied the required `pynwb` constraint. Multiprocessing was justified by the large full dataset and reduced conversion to about three minutes.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `nwb.subject.subject_id`. Unique IDs are numerically sorted and each successfully converted session gets an index into that list.

ii.
```python
subject = nwb.subject.subject_id
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The agent preferred the NWB metadata over parsing directory names and verified the resulting 11 mice against the paper and folders.

## 1-c. How are the data split into sessions?

i. Every NWB file is a session; its converted trials form one entry in each session-level list.

ii.
```python
return dict(neural=neural, input=inputs, output=outputs,
            subject=S['subject'], n_neurons=len(cells), info=info)
```

iii. File naming, scene/date metadata, and the 152-file inventory supported this interpretation.

## 1-d. How are the data split into trials?

i. Trial starts are all positive `trial_start` frames and ends are all positive `teleport` frames. Slices use `[start, teleport)`, excluding teleport/ITI.

ii.
```python
starts = np.where(trial_start > 0)[0]
teles = np.where(teleport > 0)[0]
for i, (a, b) in enumerate(zip(starts, teles)):
    sl = slice(a, b)
```

iii. The notes identify these as track entry and exit. The agent deliberately used zero-based `[start, stop)` rather than the paper code's shifted indices so behavior and neural frames remain aligned.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped when more than 30% of their frames have cumulative lick count greater than 2. No minimum-length filter is used.

ii.
```python
lick_err = np.array([(lick[a:b] > LICK_ERROR_COUNT).mean() > LICK_ERROR_FRAC
                     for a, b in zip(starts, teles)])
if lick_err[i]:
    continue
```

iii. The agent cites the paper's lick-sensor-error rule and reports reproducing exactly 81 flagged trials. It chose to drop them because their lick output would be corrupt.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from per-plane `Fluorescence` and `Neuropil` ROI series, plus `ImageSegmentation.iscell`; stored `Deconvolved` data is not used.

ii.
```python
Fseries = oph['Fluorescence'].roi_response_series
Nseries = oph['Neuropil'].roi_response_series
F = np.concatenate([np.asarray(Fseries[k].data[:nframes, :], dtype=np.float32).T
                    for k in plane_keys], axis=0)
```

iii. Investigation showed stored `Deconvolved` was suite2p output from raw fluorescence, whereas the paper analyzed events recomputed from its own maximin dF/F.

## 2-b. How is the `neural` data processed?

i. For `iscell` ROIs, the agent subtracts `0.7*Fneu`, restores each trial's mean neuropil, applies Gaussian-15 then 300-frame min/max baseline filters, computes `(F-F0)/abs(F0)`, smooths dF/F with Gaussian sigma 2, and runs OASIS (`tau=.7`) per trial. Planes are pooled. It always excludes teleport periods and uses `[start, teleport)`.

ii.
```python
f_ -= NEU_COEF * fneu_
seg = nansmooth_nd(f_[:, s:e], [0, BASELINE_SMOOTH])
seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, fs)
```

iii. This was presented as a port of the paper's `preprocessing.dff`. The agent documented its one-frame indexing deviation, but did not implement the paper/reference per-session `keep_teleports` metadata.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps `iscell==1` ROIs, then removes cells with Pearson correlation between dF/F and speed greater than 0.5.

ii.
```python
keep = np.where(S['iscell'])[0]
corr_speed = np.where(denom > 0, (Dz @ spz) / np.maximum(denom, 1e-12), 0.0)
cells = np.where(~(corr_speed > INTERNEURON_R))[0]
```

iii. Both filters are attributed to the paper: suite2p/manual ROI curation and putative-interneuron exclusion. The observed exclusion fraction was checked against the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials begin at the `trial_start` frame, so column zero is the alignment event; the same `[a,b)` indices are used for behavior.

ii.
```python
neu = events[np.ix_(cells, np.arange(a, b))].astype(np.float32)
```

iii. The agent states behavior was already interpolated onto imaging frames, making explicit resampling unnecessary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native per-plane imaging frames are retained at 15.5078125 Hz, or 64.484 ms; no rebinning occurs.

ii.
```python
fs = S['rate'] / S['n_planes']
'time_bin_size': 1000.0 / (15.5078125)
```

iii. The notes report a uniform frame period and one-to-one behavior/imaging samples across sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses timestamps attached to the raw `position` behavioral series.

ii.
```python
t = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
```

iii. Position timestamps were treated as the common imaging-aligned behavior clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from every timestamp in the trial and cast to float32.

ii.
```python
tt = (t[sl] - t[a]).astype(np.float32)
inp[0] = tt
```

iii. This directly implements seconds elapsed from the requested alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical slice and length as neural data; no interpolation is performed.

ii.
```python
T = b - a
inp = np.empty((4, T), dtype=np.float32)
neu = events[np.ix_(cells, np.arange(a, b))]
```

iii. The agent verified that NWB behavior was already frame-aligned to imaging.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment` series.

ii.
```python
env = np.asarray(beh['environment'].data[:], dtype=np.float64)
morph = np.array([np.unique(env[a:b])[0] for a, b in zip(starts, teles)])
```

iii. All trials were checked to be constant and limited to 0/1, matching ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The first unique value within each trial is validated as 0 or 1 and broadcast over time.

ii.
```python
assert np.all(np.isin(morph, [0.0, 1.0]))
inp[1] = morph[i]
```

iii. Broadcasting gives a uniform `(4,T)` input matrix while retaining a per-trial variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index over paired trial starts and teleports, not the NWB `trial number` series.

ii.
```python
for i, (a, b) in enumerate(zip(starts, teles)):
    inp[2] = i
```

iii. The index is a simple within-session sequential trial identifier.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The zero-based index is broadcast to every frame. Indices retain gaps when a lick-error trial is dropped.

ii.
```python
inp[2] = i
trial_ids.append(i)
```

iii. This preserves original session trial numbering after curation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` timestamps mapped to behavior frames, jointly requiring `reward_zone` activity when defining each trial's rewarded status.

ii.
```python
ridx = np.searchsorted(t, reward_t)
reward[ridx] = 1.0
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
```

iii. The notes link this to the paper's `isreward` trial type and use reward-zone entry as a consistency condition.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trials after the first, the preceding raw trial's `isreward` is broadcast. The first trial is set to 1. Dropped trials still count as the preceding trial.

ii.
```python
inp[3] = 1.0 if i == 0 else float(isreward[i - 1])
```

iii. The agent reasoned that imaging followed 30 rewarded warm-up trials, so rewarded was more plausible than omitted for the unobserved predecessor.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and a per-trial zone. Zone labels come primarily from the scene encoded in `nwb.identifier` with a switch at trial 30, checked against position at the first positive `reward_zone` frame; observed labels replace disagreements.

ii.
```python
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
fixed[valid] = emp[valid]
z0, z1 = REWARD_ZONES[zl]
```

iii. Scene rules mirror the paper code, and the empirical check reportedly agreed for 100% of observable trials across all sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the active zone, position minus zone start before it, and position minus zone end after it; the result is then categorized.

ii.
```python
d[before] = pos[before] - zone_start
d[after] = pos[after] - zone_end
out[0] = discretize_distance(d)
```

iii. This follows the requested distance to any location in the reward zone and the paper's reward-relative position concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement the seven requested intervals; exact zero remains class 3.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d < -10)] = 1
out[(d >= -10) & (d < 0)] = 2
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The explicit comparisons were chosen to match boundary wording exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural events use the same `[a,b)` slice and therefore the same columns.

ii.
```python
d = signed_distance_to_zone(pos[sl], z0, z1)
neu = events[np.ix_(cells, np.arange(a, b))]
```

iii. No resampling is needed because behavior is stored on imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` series.

ii.
```python
pos = np.asarray(beh['position'].data[:], dtype=np.float64)
out[1] = discretize_position(pos[sl])
```

iii. The series records corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped into `[0,450)` and divided into 90-cm widths using floor.

ii.
```python
p = np.clip(pos, 0.0, TRACK_LENGTH - 1e-9)
return np.floor(p / 90.0).astype(np.int8)
```

iii. Clipping safely absorbs small measurements outside the nominal track into the endpoint bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes are floor(position/90), producing 0 through 4 after clipping.

ii.
```python
return np.floor(p / 90.0).astype(np.int8)
```

iii. Five 90-cm bins exactly span the 450-cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The identical trial slice is applied to position and neural events.

ii.
```python
out[1] = discretize_position(pos[sl])
neu = events[np.ix_(cells, np.arange(a, b))]
```

iii. The raw behavioral samples are already aligned to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavior `lick` time series.

ii.
```python
lick = np.asarray(beh['lick'].data[:], dtype=np.float64)
```

iii. The agent interpreted positive cumulative counts as a lick in that frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero become 1 and all others 0; sensor-error trials are removed entirely.

ii.
```python
out[3] = (lick[sl] > 0).astype(np.int8)
```

iii. This implements the required binary output and avoids known corrupt trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural use the same trial indices.

ii.
```python
out[3] = (lick[sl] > 0).astype(np.int8)
neu = events[np.ix_(cells, np.arange(a, b))]
```

iii. Behavior and imaging are one-to-one by frame.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene name in `nwb.identifier`, trial index, `reward_zone`, and `position` used for empirical verification/fallback.

ii.
```python
scene = nwb.identifier.split('/')[-1]
zone_labels = scene_reward_zones(S['scene'], ntrials_raw)
emp = empirical_reward_zones(pos, rzone, starts, teles)
```

iii. The paper code defines zones from scene transitions; the raw-zone check confirmed all observable assignments.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed scenes retain one label; switch scenes change at index 30. Valid empirical labels replace any scene disagreement, then A/B/C map to 0/1/2 and are broadcast.

ii.
```python
n0 = min(change_trial, ntrials)
return np.array([m.group(1)] * n0 + [m.group(2)] * (ntrials - n0))
out[4] = ZONE_TO_IDX[zl]
```

iii. This is justified as a direct port plus an independent data check, and it supplies labels for omission trials where the zone signal is absent.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward` timestamps and the behavior `reward_zone` series.

ii.
```python
reward_t = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)
isreward = np.array([bool((reward[a:b] > 0).any() and (rzone[a:b] > 0).any())
                     for a, b in zip(starts, teles)])
```

iii. Reward delivery is the primary signal; zone entry was added as a consistency requirement based on the paper's trial-type logic.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. `Reward` timestamps are assigned to the first behavior timestamp not earlier than the event via `searchsorted` (then clipped). A trial is 1 only if it contains both a mapped reward event and positive reward-zone activity; the value is broadcast.

ii.
```python
ridx = np.clip(np.searchsorted(t, reward_t), 0, nframes - 1)
reward[ridx] = 1.0
out[5] = int(isreward[i])
```

iii. The agent reports an omission rate near the paper's 15% and treats the result as per-trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Ophys arrays are truncated to behavior length; missing empirical reward-zone observations retain scene-derived labels; zero-variance speed correlations become zero; unexpected trial pairing/environment values and neural NaNs assert. Worker exceptions are caught and the failed session is omitted after reporting. Reward indices beyond the clock are clipped.

ii.
```python
Fseries[k].data[:nframes, :]
if valid.any(): fixed[valid] = emp[valid]
except Exception as e: return dict(error=str(e), file=os.path.basename(fn))
results = [r for r in results if 'error' not in r]
```

iii. The notes identify ten sessions with one extra trailing ophys frame and say it occurs after the final teleport. Assertions and independent spot checks were used to catch other issues.

## 13-a. What are the most time-consuming steps of the code?

i. NWB loading, whole-session maximin dF/F plus OASIS, and writing the 9.52-GB pickle dominate. Timings are recorded per session.

ii.
```python
t_load = time.time() - t0
dff, events = compute_dff_events(...)
t_dff = time.time() - t0
pickle.dump(data, f, protocol=4)
```

iii. The notes measured loading at roughly 0.5-1.2 s/session and neural processing at 3.3-5.8 s in sample sessions; multiprocessing was added to reduce wall time.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops remain for copying segments, trial-specific baselines/deconvolution, empirical zones, reward/error summaries, and variable-length output construction. Cell-speed correlations were vectorized into a matrix-vector product; session work was parallelized.

ii.
```python
for s, e in zip(starts, teles):
    f_[:, s:e] = F[:, s:e]
corr_speed = (Dz @ spz) / np.maximum(denom, 1e-12)
```

iii. The agent says per-trial loops are natural because trials vary in length, while cell operations were the profitable vectorization target.

## 13-c. What processing does the code repeat multiple times?

i. Trial traversal and slicing are repeated across dF/F construction, baseline/deconvolution, zone inference, reward/environment/error summaries, output construction, and optional plotting. Gaussian/min/max/OASIS processing is necessarily performed separately for each trial.

ii.
```python
for s, e in zip(starts, teles): ...
for a, b in zip(starts, teles): ...
for i, (a, b) in enumerate(zip(starts, teles)): ...
```

iii. The notes emphasize that the production converter loads each session once and processes whole-session arrays in one pass; repeated trial loops preserve trial-specific reference processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full dF/F is retained only long enough to calculate interneuron correlations, after which downstream data uses only OASIS events. `plane_idx` is loaded but never used. Empirical zone labels and agreement are computed even though scene labels agreed everywhere. Optional plots repeat several discretizations but are disabled in normal full conversion.

ii.
```python
plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)
dff, events = compute_dff_events(...)
emp = empirical_reward_zones(...)
```

iii. dF/F and empirical checks are defensible validation/curation intermediates; `plane_idx` is genuinely unused. Plot-only work was intentionally optional.
