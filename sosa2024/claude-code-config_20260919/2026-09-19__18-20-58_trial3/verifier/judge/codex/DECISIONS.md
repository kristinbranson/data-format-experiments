# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data/sub-*/*.nwb`, treats each file as one session, and reads each session directly with `h5py`. Within a session it separately loads the behavioral streams and the fluorescence/neuropil imaging streams.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
with h5py.File(path, 'r') as f:
    beh, starts, stops = read_behavior(f)
    ...
    F, Fneu, n_iscell, nplanes = read_fluorescence(f)
```

```python
def read_behavior(f):
    B = f[BEHAVIOR_PATH]
    beh = {k: B[f'{k}/data'][:] for k in
           ['position', 'speed', 'lick', 'reward_zone', 'environment',
            'trial_start', 'teleport', 'scanning']}
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset is a one-level tree of subject folders containing 152 `*_behavior+ophys.nwb` files, so globbing those files captures the full dataset. The notes also say the NWB export contains the already aligned behavioral and imaging arrays needed for conversion.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the NWB `subject_id` field, then uniqued and sorted numerically (`m3`, `m4`, ...). The file layout `sub-*` is used to find candidate files, but subject identity comes from the file metadata.

ii. 
```python
subject = f['general/subject/subject_id'][()].decode()
...
subjects = sorted({r['info']['subject'] for r in results},
                  key=lambda s: int(s[1:]))
subject_idx = np.array([subjects.index(r['info']['subject']) for r in results],
                       dtype=np.int64)
```

iii. The notes describe the subject set as the 11 switch-task mice and explicitly mention a later fix to sort sessions by numeric mouse/day order, indicating the agent treated subject identity as metadata rather than relying only on path order.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is taken from `general/session_id`, and the final session list is sorted by mouse number then session/day number.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
...
session_id = f['general/session_id'][()].decode()
...
results.sort(key=lambda r: (int(r['info']['subject'][1:]),
                            int(r['info']['session_id'])))
```

iii. The notes state that `/app/data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb` is the session-level storage pattern and that `ses-NN` is the experiment day.

## 1-d. How are the data split into trials?

i. Trials are defined as the frame interval from `trial_start` to `teleport`, i.e. `[trial_start, teleport)`. The start indices are frames where `trial_start > 0`, and the stop indices are frames where `teleport > 0`.

ii. 
```python
starts = np.where(beh['trial_start'] > 0)[0]
stops = np.where(beh['teleport'] > 0)[0]
assert len(starts) == len(stops) and np.all(stops > starts), 'bad trial boundaries'
```

```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
speed_tr = [beh['speed'][s:e] for s, e in zip(starts, stops)]
```

iii. The docstring and notes say this matches the behavioral convention in the paper code (`behavior.*`) and represents the on-track lap, excluding the teleport interval.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials flagged as lick-sensor-error trials. A trial is excluded when more than 30% of its frames have cumulative lick count above 2.

ii. 
```python
LICK_CORRECTION_THR = 0.3
...
lick_error = np.array([(l > 2).sum() / len(l) > LICK_CORRECTION_THR for l in lick_tr])
...
for t in range(ntrials):
    if lick_error[t]:
        continue
```

iii. The notes explicitly justify this as the paper's trial curation rule and say the chosen threshold reproduces the paper's 81 bad trials exactly. They also note that the reference code NaNs those licks, but the AI dropped the trials instead because the decoder outputs cannot contain NaNs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the raw fluorescence and neuropil traces, not from the NWB `Deconvolved` series.

ii. 
```python
def read_fluorescence(f):
    ...
    Fs.append(f[f'{OPHYS_PATH}/Fluorescence/{p}/data'][:][:, keep])
    Ns.append(f[f'{OPHYS_PATH}/Neuropil/{p}/data'][:][:, keep])
    F = np.ascontiguousarray(np.concatenate(Fs, axis=1).T)
    Fneu = np.ascontiguousarray(np.concatenate(Ns, axis=1).T)
```

iii. The notes repeatedly say the NWB `Deconvolved` array is suite2p output from raw fluorescence and not the signal the paper analyzes, so the agent intentionally recomputed the neural signal from `Fluorescence` and `Neuropil`.

## 2-b. How is the `neural` data processed?

i. The AI computes per-trial dF/F by subtracting `0.7 * Fneu`, adding back the trial-wise neuropil mean, computing a maximin baseline within each trial, converting to `(F - F0) / |F0|`, and smoothing with a Gaussian (`sigma=2`). It does not deconvolve the dF/F into events; the stored neural matrices are the dF/F traces after interneuron removal.

ii. 
```python
for s, e in zip(starts, stops):
    f = F[:, s:e].astype(np.float64)
    fneu = Fneu[:, s:e].astype(np.float64)
    f = f - NEU_COEF * fneu
    f = f + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)
    flow = nan_gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA, axis=-1)
    flow = ndi.minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
    flow = ndi.maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
    dff = (f - flow) / np.abs(flow)
    dff = nan_gaussian_filter1d(dff, DFF_SMOOTH_SIGMA, axis=-1)
    out.append(dff.astype(np.float32))
```

iii. The notes say this is a reimplementation of the paper's `preprocessing.dff` arithmetic, but that the final neural signal was kept as dF/F instead of events because the per-session deconvolution kernel parameter was not available in the NWB export and a later comparison showed dF/F decoding better.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only `iscell` ROIs, then removes putative interneurons defined by Pearson correlation between dF/F and running speed greater than 0.5.

ii. 
```python
iscell = seg['iscell'][:, 0] > 0
...
is_int, speed_r = find_putative_interneurons(dff_trials, [v.astype(np.float64)
                                                         for v in speed_tr])
keep_cells = ~is_int
dff_trials = [d[keep_cells] for d in dff_trials]
```

```python
return np.nan_to_num(r, nan=0.0) > r_thresh, r
```

iii. The notes say both filters were copied from the paper pipeline: manual/suite2p curation through `iscell`, then `spatial.is_putative_interneuron(..., r_thresh=0.5)`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the start of trial by slicing the same `[trial_start, teleport)` frame window used for behavior. There is no additional offset or resampling.

ii. 
```python
def compute_dff_trials(F, Fneu, starts, stops):
    ...
    for s, e in zip(starts, stops):
        f = F[:, s:e].astype(np.float64)
```

```python
neural.append(dff_trials[t])
inp[0] = time_tr[t]
```

iii. The notes explicitly say the temporal alignment event is trial start and that the same sample window is used for both neural and behavioral streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging-frame resolution. It derives the effective frame rate from the median difference of behavior timestamps, computes the dataset time bin from that, and does no temporal rebinning.

ii. 
```python
frame_rate = float(1.0 / np.median(np.diff(beh['time'])))
...
frame_rates = np.array([r['info']['frame_rate'] for r in results])
bin_ms = 1000.0 / float(np.mean(frame_rates))
```

iii. The notes say the per-plane sampling rate is effectively 15.5078125 Hz in all sessions, including the two-plane mice, and that native-frame resolution was preserved.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the behavior timestamps attached to the position series. The AI stores them in `beh['time']`.

ii. 
```python
beh['time'] = B['position/timestamps'][:]
...
time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
```

iii. The notes say the behavior streams are already on the imaging frame grid, so any behavior timestamp stream would have worked; the AI used the position timestamps as the session time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the timestamp at the first frame of the trial so time starts at 0 within the trial.

ii. 
```python
time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
...
inp[0] = time_tr[t]
```

iii. The notes describe this as trial-start alignment with `off_start = 0.0`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI aligns this input by slicing timestamps with the same frame indices used for the neural data, so both are on the same per-trial frame grid.

ii. 
```python
time_tr = [beh['time'][s:e] - beh['time'][s] for s, e in zip(starts, stops)]
...
neural.append(dff_trials[t])
inp[0] = time_tr[t]
```

iii. The notes say VR behavior was already exported on the imaging frame grid and that no additional temporal alignment was needed beyond using identical trial windows.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii. 
```python
env_tr = [beh['environment'][s:e] for s, e in zip(starts, stops)]
env = np.array([np.unique(e)[0] for e in env_tr])
```

iii. The notes map this field to the paper's environment/morph variable and say it takes values 0 or 1 for ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI requires the environment to be constant within a trial, extracts the unique value for each trial, and then broadcasts that scalar across all frames of the trial in the decoder input matrix.

ii. 
```python
env = np.array([np.unique(e)[0] for e in env_tr])
for e in env_tr:
    assert len(np.unique(e)) == 1, 'environment changes within a trial'
...
inp[1] = env[t]
```

iii. The notes say the environment is constant within trials and switches only at trial boundaries in the few sessions that contain both environments.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial index in the conversion loop after trial boundaries have been identified from `trial_start` and `teleport`.

ii. 
```python
for t in range(ntrials):
    ...
    inp[2] = t
```

iii. The notes say this is the original within-session trial index and that it is intentionally preserved even when some trials are later removed.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond using the 0-based trial index. The value is constant across all frames in the trial.

ii. 
```python
inp = np.empty((4, T), dtype=np.float32)
...
inp[2] = t
```

iii. The notes justify keeping the original ordinal position so it still refers to the true session trial number after lick-error trial removal.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from a per-trial reward outcome `isreward`, which itself is computed from the per-frame reconstructed `Reward` event stream together with whether the trial entered the reward zone.

ii. 
```python
rt = B['Reward/timestamps'][:]
ts = beh['time']
idx = np.clip(np.searchsorted(ts, rt), 0, len(ts) - 1)
...
reward = np.zeros(len(ts))
reward[idx] = 1.0
beh['reward'] = reward
```

```python
isreward = np.array([float(np.any(r > 0) and np.any(z > 0))
                     for r, z in zip(rew_tr, rz_tr)])
```

iii. The notes say the AI followed `behavior.get_trial_types`, where a trial counts as rewarded only if reward delivery and reward-zone occupancy both occurred.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `t`, the AI assigns the previous trial's `isreward` value. The first trial of each session is set to 0, and the value is broadcast across all frames of the current trial.

ii. 
```python
inp[3] = isreward[t - 1] if t > 0 else 0.0
```

iii. The notes explicitly justify setting the first trial to 0 because there is no previous recorded lap in the session.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial position and the active reward-zone identity for that trial. The reward-zone identity is inferred from the session scene string (`identifier`) using the paper's reward-zone schedule logic, not from the raw `reward_zone` occupancy trace itself.

ii. 
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
labels = get_reward_zone_labels(scene, ntrials)
...
lo, hi = REWARD_ZONE_COORDS[labels[t]]
dist = reward_zone_distance(pos_tr[t], lo, hi)
```

iii. The notes say this follows `behavior.get_reward_zones` and is preferable because the active zone is defined even on omission trials or trials without an observed zone entry.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest edge of the active reward zone: negative before the zone, zero inside it, and positive after it.

ii. 
```python
def reward_zone_distance(pos, lo, hi):
    return np.where(pos < lo, pos - lo, np.where(pos > hi, pos - hi, 0.0))
```

```python
lo, hi = REWARD_ZONE_COORDS[labels[t]]
dist = reward_zone_distance(pos_tr[t], lo, hi)
```

iii. The notes describe this as the decoder-task adaptation of the paper's reward-relative position concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses seven bins corresponding to the task specification: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

ii. 
```python
def bin_reward_distance(dist):
    b = np.zeros(dist.shape, dtype=np.int8)
    b[(dist >= -50) & (dist < -10)] = 1
    b[(dist >= -10) & (dist < 0)] = 2
    b[dist == 0] = 3
    b[(dist > 0) & (dist <= 10)] = 4
    b[(dist > 10) & (dist <= 50)] = 5
    b[dist > 50] = 6
    return b
```

iii. The notes say the discretization was taken directly from the decoder task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned frame-by-frame within each trial because it is computed from `pos_tr[t]`, which uses the same `[start, stop)` frame slice as the neural trial.

ii. 
```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
...
dist = reward_zone_distance(pos_tr[t], lo, hi)
...
neural.append(dff_trials[t])
out[0] = bin_reward_distance(dist)
```

iii. The notes say there is no temporal offset between the derived outputs and the neural traces because both come from the same trial-aligned frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii. 
```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
...
out[1] = bin_position(pos_tr[t])
```

iii. The notes say position is already on the imaging frame grid and spans the 450 cm track within each converted trial.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI bins each position sample into one of five equal 90 cm bins across the 450 cm track using floor division and clipping.

ii. 
```python
def bin_position(pos):
    b = np.floor(pos / (TRACK_LENGTH / N_POSITION_BINS))
    return np.clip(b, 0, N_POSITION_BINS - 1).astype(np.int8)
```

iii. The notes say clipping is acceptable because a small number of edge samples fall slightly outside `[0, 450]`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The five categories are 90 cm bins: `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`, with out-of-range values clipped into the end bins.

ii. 
```python
TRACK_LENGTH = 450.0
N_POSITION_BINS = 5
...
def bin_position(pos):
    b = np.floor(pos / (TRACK_LENGTH / N_POSITION_BINS))
    return np.clip(b, 0, N_POSITION_BINS - 1).astype(np.int8)
```

iii. The notes say this discretization comes directly from the decoder task requirement of five equal-sized bins over 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned frame-by-frame within each trial because it is computed from the same per-trial frame slices used to compute and store the neural matrices.

ii. 
```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
...
neural.append(dff_trials[t])
out[1] = bin_position(pos_tr[t])
```

iii. The notes say the behavioral data are already synchronized to the imaging frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii. 
```python
lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
...
out[3] = (lick_tr[t] > 0).astype(np.int8)
```

iii. The notes identify lick as a framewise behavioral stream that is already aligned with imaging frames.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes lick values so any value greater than 0 becomes 1 and 0 stays 0.

ii. 
```python
out[3] = (lick_tr[t] > 0).astype(np.int8)
```

iii. The notes say this follows the paper code's binary treatment of licking for peri-event analyses and matches the decoder task's binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by applying the same `[start, stop)` frame slice to the lick series and neural data.

ii. 
```python
lick_tr = [beh['lick'][s:e] for s, e in zip(starts, stops)]
...
neural.append(dff_trials[t])
out[3] = (lick_tr[t] > 0).astype(np.int8)
```

iii. The notes say there is no extra temporal alignment step because the NWB behavior streams are already on the 2P frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB `identifier` scene string together with the known switch-at-trial-30 task structure.

ii. 
```python
scene = f['identifier'][()].decode().split('/')[-1]
...
labels = get_reward_zone_labels(scene, ntrials)
...
out[4] = 'ABC'.index(labels[t])
```

iii. The notes say this mirrors `behavior.get_reward_zones` and avoids inferring zone identity from noisy or absent zone-entry observations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene name: fixed scenes produce one label for all trials, while switch scenes produce one label for the first 30 trials and another thereafter. It then maps `A/B/C` to `0/1/2`.

ii. 
```python
def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    for lab in 'ABC':
        if scene.endswith('Location' + lab):
            return np.array([lab] * ntrials)
    for first in 'ABC':
        if f'{first}_to' in scene:
            second = scene[-1]
            ...
            n0 = min(change_trial, ntrials)
            return np.array([first] * n0 + [second] * (ntrials - n0))
```

iii. The notes say the agent validated this against observed reward-zone entries and found zero mismatches across the dataset.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the reconstructed per-frame `Reward` event stream plus per-trial reward-zone occupancy.

ii. 
```python
reward = np.zeros(len(ts))
reward[idx] = 1.0
beh['reward'] = reward
...
isreward = np.array([float(np.any(r > 0) and np.any(z > 0))
                     for r, z in zip(rew_tr, rz_tr)])
```

iii. The notes say this follows the paper's `behavior.get_trial_types` definition rather than using raw reward timestamps alone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI computes a binary rewarded/omitted label via `any(reward) and any(reward_zone)` and then broadcasts that trial-level label across every frame in the trial.

ii. 
```python
isreward = np.array([float(np.any(r > 0) and np.any(z > 0))
                     for r, z in zip(rew_tr, rz_tr)])
...
out[5] = int(isreward[t])
```

iii. The notes justify this as the same per-trial outcome definition used in the paper code, with the framewise broadcast done only because the decoder format represents outputs as per-trial time series.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly uses fail-fast checks plus a few targeted fixes. It asserts valid trial boundaries and constant environment within trials, maps reward timestamps to the nearest frame, drops lick-sensor-error trials, and replaces any non-finite dF/F values with zero after warning.

ii. 
```python
idx = np.clip(np.searchsorted(ts, rt), 0, len(ts) - 1)
left = np.clip(idx - 1, 0, len(ts) - 1)
idx = np.where(np.abs(ts[left] - rt) < np.abs(ts[idx] - rt), left, idx)
```

```python
assert len(starts) == len(stops) and np.all(stops > starts), 'bad trial boundaries'
...
assert len(np.unique(e)) == 1, 'environment changes within a trial'
...
if n_nonfinite:
    warnings.warn(f'{os.path.basename(path)}: {n_nonfinite} non-finite dF/F values '
                  f'set to 0')
    dff_trials = [np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                  for d in dff_trials]
```

iii. The notes say the dataset was surveyed first and that these were the actual edge cases the agent decided to handle: lick-sensor failures, occasional non-finite dF/F, and reward events that needed to be mapped back onto the frame grid.

## 13-a. What are the most time-consuming steps of the code?

i. The AI identified reading fluorescence from NWB/HDF5, per-trial dF/F computation, and writing the large pickle as the dominant costs. It also measured these timings per session and in aggregate.

ii. 
```python
timing['read_behavior'] = t1 - t0
timing['read_fluorescence'] = t2 - t1
...
timing['dff'] = t3 - t2
...
timing['interneurons'] = t4 - t3
...
timing['total'] = time.time() - t0
```

```python
with open(args.outfile, 'wb') as fh:
    pickle.dump(data, fh, protocol=4)
print(f'Wrote {args.outfile} '
      f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t:.1f} s')
```

iii. The notes report a full conversion time of about 39 seconds, with about 11 seconds spent writing the 9.5 GB pickle, and explicitly single out fluorescence reading and dF/F as the expensive processing stages.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining trialwise loops are the dF/F loop over trials, the streaming interneuron-correlation loop over trials, and the final assembly loop over trials. The AI notes that the biggest avoidable copies were already removed by earlier optimizations.

ii. 
```python
for s, e in zip(starts, stops):
    ...
    out.append(dff.astype(np.float32))
```

```python
for d, v in zip(dff_trials, speed_trials):
    ...
```

```python
for t in range(ntrials):
    if lick_error[t]:
        continue
    ...
```

iii. The notes' "Code inefficiencies identified" and "Code speedups added" sections say the agent deliberately avoided full-session padded arrays and expensive concatenations, but accepted some per-trial loops because trials have variable lengths.

## 13-c. What processing does the code repeat multiple times?

i. The code traverses per-trial data several times: once to slice behavior into trial lists, again to compute dF/F, again to compute interneuron statistics, and again to assemble the saved tensors. Optional diagnostic plotting also recomputes example-cell preprocessing quantities.

ii. 
```python
pos_tr = [beh['position'][s:e] for s, e in zip(starts, stops)]
speed_tr = [beh['speed'][s:e] for s, e in zip(starts, stops)]
...
dff_trials = compute_dff_trials(F, Fneu, starts, stops)
...
is_int, speed_r = find_putative_interneurons(dff_trials, [v.astype(np.float64)
                                                         for v in speed_tr])
...
for t in range(ntrials):
    ...
```

```python
f = F[craw, s:e].astype(np.float64) - 0.7 * Fneu[craw, s:e].astype(np.float64)
f = f + 0.7 * Fneu[craw, s:e].mean()
flow = ndi.gaussian_filter1d(f, BASELINE_SMOOTH_SIGMA)
```

iii. The notes explicitly compare this against more wasteful alternatives and say the agent reduced, but did not completely eliminate, repeated passes through the per-trial data.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some values that are not used by the saved decoder dataset itself: the full `speed_r` vector returned by interneuron detection, detailed timing/info bookkeeping, summary concatenations, and optional plotting diagnostics. These are used for logging and validation rather than downstream decoding.

ii. 
```python
is_int, speed_r = find_putative_interneurons(dff_trials, [v.astype(np.float64)
                                                         for v in speed_tr])
```

```python
info = dict(
    ...
    reward_rate=float(isreward.mean()), timing=timing,
)
...
if show_processing:
    plot_processing(...)
```

iii. The notes show the agent intentionally kept these diagnostics to validate correctness and runtime, even though they are not consumed by `train_decoder.py` after the pickle is written.
