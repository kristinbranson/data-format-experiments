# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globbed every NWB file under `/app/data/sub-*`, sorted them, processed each file as one session (normally with a multiprocessing pool), and assembled all returned session lists into the pickle.

ii. ```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
with Pool(args.workers, initializer=_init, initargs=({},)) as pool:
    for i, res in enumerate(pool.imap(_worker, files, chunksize=1)):
        results.append(res)
```

iii. The notes say this found 11 mice, 152 NWB sessions, and 12,216 raw trials. The agent chose direct `h5py` reads for speed and parallelized independent sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, unique mouse IDs are numerically sorted, and each session receives an index into that list.

ii. ```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({r['info']['subject'] for r in results}, key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['info']['subject']) for r in results])
```

iii. The notes identify the subject field and verify 11 expected switch-task mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Results are sorted by numeric mouse ID and experiment-day `session_id`.

ii. ```python
session_id = f['general/session_id'][()].decode()
results.sort(key=lambda r: (int(r['info']['subject'][1:]), int(r['info']['session_id'])))
```

iii. The filename/session metadata and paper schedule supported this mapping; the notes explain why m11 has only days 3–14.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and stops are positive `teleport` samples; each trial is the half-open interval `[start, stop)`.

ii. ```python
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts)
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
```

iii. The agent states this is the behavioral convention used by the repository and excludes the teleport/ITI while aligning all streams to identical samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed when more than 30% of their frames have cumulative lick count greater than 2. No short-trial filter is used.

ii. ```python
lick_error = np.array([(l > 2).sum() / len(l) > LICK_CORRECTION_THR for l in lick_tr])
for t in range(ntrials):
    if lick_error[t]:
        continue
```

iii. This follows `correct_lick_sensor_error` and the Methods' 30% threshold, reproducing the paper's 81 affected trials. Because categorical outputs cannot contain NaNs, the agent drops whole trials rather than only invalidating lick samples.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from raw `Fluorescence/plane*/data` (F), `Neuropil/plane*/data` (Fneu), ROI references, and `iscell`; the stored `Deconvolved` stream is not used.

ii. ```python
Fs.append(f[f'{OPHYS_PATH}/Fluorescence/{p}/data'][:][:, keep])
Ns.append(f[f'{OPHYS_PATH}/Neuropil/{p}/data'][:][:, keep])
```

iii. The agent reasoned that NWB `Deconvolved` is Suite2p output from raw F, not the paper's processed signal.

## 2-b. How is the `neural` data processed?

i. Per trial, the agent subtracts `0.7*Fneu`, adds back `0.7` times mean trial neuropil, computes a Gaussian/maximin baseline (sigma 15; 300-frame minimum then maximum filters), forms `(F-F0)/abs(F0)`, smooths with sigma 2, and stores float32 dF/F. Unlike the human solution, it does not OASIS-deconvolve this dF/F.

ii. ```python
f = f - NEU_COEF * fneu
f = f + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)
flow = ndi.maximum_filter1d(ndi.minimum_filter1d(
    nan_gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA), BASELINE_FILTER_WIN), BASELINE_FILTER_WIN)
dff = nan_gaussian_filter1d((f - flow) / np.abs(flow), DFF_SMOOTH_SIGMA)
```

iii. The agent says this reproduces `preprocessing.dff`, but declines deconvolution because per-session Suite2p tau was absent and argues dF/F decoded better than guessed OASIS events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It first retains `iscell[:,0] > 0`, pools planes, then removes cells with Pearson correlation between dF/F and in-trial speed greater than 0.5.

ii. ```python
iscell = seg['iscell'][:, 0] > 0
keep = iscell[rois]
is_int, speed_r = find_putative_interneurons(dff_trials, speed_tr)
dff_trials = [d[~is_int] for d in dff_trials]
```

iii. Both filters are attributed to the reference pipeline; streaming sufficient statistics avoid concatenating all trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays are sliced at the same `[trial_start, teleport)` indices as behavior, making column zero the trial-start frame.

ii. ```python
for s, e in zip(starts, stops):
    f = F[:, s:e].astype(np.float64)
```

iii. The notes explicitly set alignment to trial start with `off_start=0` and explain the one-index convention of the original helper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging-frame resolution is retained without rebinning, approximately 64.4839 ms (15.5078 Hz).

ii. ```python
frame_rate = float(1.0 / np.median(np.diff(beh['time'])))
bin_ms = 1000.0 / float(np.mean(frame_rates))
```

iii. The behavior is already resampled to imaging frames, and all session rates were asserted equal.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps`.

ii. ```python
beh['time'] = B['position/timestamps'][:]
```

iii. The notes state all behavioral streams share the imaging-frame timestamp grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from every timestamp in the trial.

ii. ```python
time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
inp[0] = time_tr[t]
```

iii. This directly implements seconds since the alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector and neural matrix use the identical `[s:e]` frame slice, so lengths and columns correspond exactly.

ii. ```python
f = F[:, s:e]
time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
```

iii. The agent relies on the NWB's already aligned imaging-frame grid and does no interpolation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` stream.

ii. ```python
env_tr = [beh['environment'][s:e] for s, e in zip(starts, stops)]
```

iii. Exploration found values 0/1 corresponding to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique within-trial value is selected, asserted constant, and broadcast over trial time.

ii. ```python
env = np.array([np.unique(e)[0] for e in env_tr])
assert len(np.unique(e)) == 1
inp[1] = env[t]
```

iii. The notes report no trials with mixed environments.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index of the raw trial-start/teleport pairs, not the NWB `trial number` stream.

ii. ```python
for t in range(ntrials):
    inp[2] = t
```

iii. The agent preserves the original session ordinal even after exclusions so trial number and previous outcome retain their experimental meanings.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond broadcasting the integer trial index across all frames.

ii. ```python
inp = np.empty((4, T), dtype=np.float32)
inp[2] = t
```

iii. This supplies the requested continuous per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from `Reward/timestamps` mapped to frames and the per-frame `reward_zone` stream.

ii. ```python
rt = B['Reward/timestamps'][:]
reward[idx] = 1.0
isreward = np.array([float(np.any(r > 0) and np.any(z > 0)) for r, z in zip(rew_tr, rz_tr)])
```

iii. The conjunction replicates `behavior.get_trial_types` and avoids treating a reward event outside a valid zone trial as a rewarded outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward events are mapped to the nearest behavior frame. The immediately preceding raw trial's binary outcome is broadcast; trial zero receives 0.

ii. ```python
idx = np.where(np.abs(ts[left] - rt) < np.abs(ts[idx] - rt), left, idx)
inp[3] = isreward[t - 1] if t > 0 else 0.0
```

iii. The first recorded trial has no known predecessor. Using raw trial indices retains the true predecessor even if that trial is later dropped for lick error.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral `position` plus active reward-zone identity parsed from the NWB identifier/scene name, with a switch after trial 30 where applicable.

ii. ```python
scene = f['identifier'][()].decode().split('/')[-1]
labels = get_reward_zone_labels(scene, ntrials)
lo, hi = REWARD_ZONE_COORDS[labels[t]]
```

iii. The agent chose the repository's scene logic because it defines zones even on omission/non-entry trials; it reports 0 mismatches against 12,216 observed zone entries.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the active interval, position minus lower edge before it, and position minus upper edge after it.

ii. ```python
return np.where(pos < lo, pos - lo, np.where(pos > hi, pos - hi, 0.0))
```

iii. This implements distance to the nearest point in the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create the seven requested classes with exact zero isolated.

ii. ```python
b[(dist >= -50) & (dist < -10)] = 1
b[(dist >= -10) & (dist < 0)] = 2
b[dist == 0] = 3
b[(dist > 0) & (dist <= 10)] = 4
b[(dist > 10) & (dist <= 50)] = 5
b[dist > 50] = 6
```

iii. The agent says these are exactly the decoder-task boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same frame slice, and distance is calculated elementwise for that slice.

ii. ```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
dist = reward_zone_distance(pos_tr[t], lo, hi)
```

iii. No resampling is needed on the shared imaging-frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavioral `position`.

ii. ```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
```

iii. The stream is already expressed in corridor centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm, floored, clipped to classes 0–4, and cast to int8.

ii. ```python
b = np.floor(pos / (TRACK_LENGTH / N_POSITION_BINS))
return np.clip(b, 0, N_POSITION_BINS - 1).astype(np.int8)
```

iii. Clipping safely absorbs small values outside the nominal 0–450 cm range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90-cm bins are used: boundaries 90, 180, 270, and 360 cm.

ii. ```python
TRACK_LENGTH = 450.0
N_POSITION_BINS = 5
```

iii. This directly follows the decoder specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The position trial slice has the same `[start, stop)` indices and length as the neural trial.

ii. ```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
dff_trials = compute_dff_trials(F, Fneu, starts, stops)
```

iii. Both streams are on the imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavioral `lick` stream.

ii. ```python
lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
```

iii. The notes describe it as a cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero are mapped to 1; all others to 0. Separately, stuck-sensor trials are dropped.

ii. ```python
out[3] = (lick_tr[t] > 0).astype(np.int8)
```

iii. This produces the required binary output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks are sliced with the same trial frame boundaries as neural data.

ii. ```python
lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
```

iii. The source is already frame-aligned.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Zone identity is derived from the scene name in the NWB `identifier`, plus trial index for switch scenes.

ii. ```python
labels = get_reward_zone_labels(scene, ntrials)
```

iii. This mirrors `behavior.get_reward_zones`, and observed zone-entry positions were used only as validation.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed scenes repeat their terminal A/B/C label; switch scenes use the first label for 30 trials and the scene's final label thereafter. A/B/C is mapped to 0/1/2 and broadcast.

ii. ```python
n0 = min(change_trial, ntrials)
return np.array([first] * n0 + [second] * (ntrials - n0))
out[4] = 'ABC'.index(labels[t])
```

iii. The Methods specify switches after 30 trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses mapped `Reward` events and the behavioral `reward_zone` stream.

ii. ```python
rew_tr = [beh['reward'][s:e] for s, e in zip(starts, stops)]
rz_tr = [beh['reward_zone'][s:e] for s, e in zip(starts, stops)]
```

iii. These are the variables used by the repository's trial-type logic.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded only if it contains both any mapped reward event and any positive reward-zone sample; the scalar is broadcast across the trial.

ii. ```python
isreward = np.array([float(np.any(r > 0) and np.any(z > 0)) for r, z in zip(rew_tr, rz_tr)])
out[5] = int(isreward[t])
```

iii. The agent cites `behavior.get_trial_types`; the resulting reward rate is about 84.7%, consistent with ~15% omissions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Structural inconsistencies (trial counts/order, mixed environments, rates) trigger assertions. Nonfinite neural values are warned about and replaced with zero. Reward events are clipped and assigned to the nearest frame. Lick-error trials are removed. There is no neural/behavior length crop or missing-zone inference.

ii. ```python
assert len(starts) == len(stops) and np.all(stops > starts)
if n_nonfinite:
    dff_trials = [np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0) for d in dff_trials]
idx = np.clip(np.searchsorted(ts, rt), 0, len(ts) - 1)
```

iii. The notes report valid synchronized trial streams and explain each defensive check; categorical outputs motivated dropping corrupted lick trials.

## 13-a. What are the most time-consuming steps of the code?

i. Fluorescence I/O, per-trial dF/F filtering, multiprocessing conversion, and writing the 9.5-GB pickle dominate. Per-session timing records read, dF/F, interneuron, and total durations.

ii. ```python
timing['read_fluorescence'] = t2 - t1
timing['dff'] = t3 - t2
pickle.dump(data, fh, protocol=4)
```

iii. The notes report 39 seconds total, including about 11 seconds to write, and identify large NWB arrays as the main workload.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial slicing/filtering and final assembly remain Python loops; much within-trial work is already NumPy-vectorized. The agent specifically replaced a cell-wise correlation loop with vectorized sufficient statistics, but the dF/F trial loop is still necessary for independent baselines.

ii. ```python
for s, e in zip(starts, stops):
    ...
for d, v in zip(dff_trials, speed_trials):
    sxy += d64 @ v
for t in range(ntrials):
    ...
```

iii. The notes argue variable trial lengths and per-trial baselines make full-session vectorization awkward; sessions are parallelized instead.

## 13-c. What processing does the code repeat multiple times?

i. Each behavioral stream is sliced separately over the same start/stop pairs; dF/F trials are traversed again for interneuron statistics and again for filtering/assembly. Optional plotting recomputes an example cell's baseline.

ii. ```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
speed_tr = [beh['speed'][s:e] for s, e in zip(starts, stops)]
lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
```

iii. The agent intentionally streams repeated passes to limit peak memory rather than concatenating the multi-gigabyte signal.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `speed_r` is calculated but not saved; `scanning` is loaded but unused; `reward_zone` is retained only long enough to derive trial outcome; optional diagnostic plotting recomputes traces and is discarded after figures. Timing and rich session metadata are auxiliary rather than decoder inputs.

ii. ```python
['position', 'speed', 'lick', 'reward_zone', 'environment',
 'trial_start', 'teleport', 'scanning']
is_int, speed_r = find_putative_interneurons(...)
```

iii. The notes justify diagnostics and metadata for validation, while memory optimizations avoid larger unnecessary intermediates.
