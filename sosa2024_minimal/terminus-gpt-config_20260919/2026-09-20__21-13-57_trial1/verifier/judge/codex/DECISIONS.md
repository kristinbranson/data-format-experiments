# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing every `/app/data/sub-*/*_behavior+ophys.nwb` path, then iterates over those files in sorted order. Each session is opened directly with `h5py.File`, and the needed HDF5 groups for behavior, deconvolved activity, and segmentation are read from the NWB file during conversion.

ii.
```python
def convert_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        scene, z0, z1, switched = scene_info(f['identifier'][()])
        dg = f['processing/ophys/Deconvolved']
        ...

def main():
    files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*_behavior+ophys.nwb')))
    ...
    for i,p in enumerate(files):
        nt, it, ot, nc, scene, rate, block = convert_session(p)
```

iii. In the trajectory, the AI says the dataset consists of NWB files one level below subject directories and that direct HDF5 access is sufficient because the NWBs already expose frame-aligned behavior and ophys arrays. It chose this path to avoid the overhead of building full `pynwb` objects while still reading every published session.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory names of the NWB files. The code strips the `sub-` prefix from each parent directory and builds `subjects` plus a `subject_idx` mapping for sessions.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*_behavior+ophys.nwb')))
subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
...
subj = os.path.basename(os.path.dirname(p)).replace('sub-','')
data['subject_idx'].append(subj_to_idx[subj])
```

iii. The trajectory repeatedly refers to the NWB directory layout as the authoritative subject split and notes that all files live one level below `sub-*` mouse directories.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The top-level conversion loop appends one session entry to `neural`, `input`, and `output` for each file.

ii.
```python
for i,p in enumerate(files):
    nt, it, ot, nc, scene, rate, block = convert_session(p)
    if len(nt) < 2:
        print('SKIP (<2 trials)', p, flush=True)
        continue
    data['neural'].append(nt); data['input'].append(it); data['output'].append(ot)
```

iii. In the trajectory, the AI describes the full dataset as 152 NWB sessions and tracks progress as one converted session per file, including subject and scene labels for each processed NWB.

## 1-d. How are the data split into trials?

i. Trials are defined from the first index where `trial_start > 0` until the first later sample where `teleport > 0`. The code uses `[s, e)` slices in that interval, ignores any clipped final trial with no later teleport event, and then rebins the slice into fixed-width blocks.

ii.
```python
trial_start = b['trial_start/data'][:]
teleport = b['teleport/data'][:]
starts = np.flatnonzero(trial_start > 0)
...
for s in starts:
    tq = np.flatnonzero(teleport[s:] > 0)
    e = s + int(tq[0]) if len(tq) else len(pos)
    if e <= s:
        continue
    if not len(tq):
        continue
    offsets = np.arange(s, e, block, dtype=int)
```

iii. The trajectory says the AI inspected trial-number transitions and concluded they occur during the inter-trial teleport period, so it aligned trials to the `trial_start` pulse at the 0 cm crossing and ended them at teleport onset to exclude the inter-trial period.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering. It drops trials if they never reach a teleport event, if the end is not after the start, or if the rebinned trial would have fewer than 2 bins. Entire sessions are skipped if fewer than 2 trials survive.

ii.
```python
if e <= s:
    continue
if not len(tq):
    continue
offsets = np.arange(s, e, block, dtype=int)
if len(offsets) < 2:
    continue
...
if len(nt) < 2:
    print('SKIP (<2 trials)', p, flush=True)
    continue
```

iii. In the trajectory, the AI says it did not want to discard long or irregular trials arbitrarily and only removed obviously clipped or too-short trials after block binning so the decoder would still have at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data come from the NWB `processing/ophys/Deconvolved/<plane>/data` arrays, after filtering ROIs with `iscell`. The AI does not derive neural activity from raw fluorescence and neuropil traces.

ii.
```python
dg = f['processing/ophys/Deconvolved']
...
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:,0] == 1
...
for name in plane_names:
    pi = int(name.replace('plane',''))
    dset = dg[name]['data']
    mask = iscell[plane_idx == pi]
    plane_series.append((dset, np.flatnonzero(mask)))
```

iii. The trajectory says the AI decided the NWBs already contain paper-processed OASIS-deconvolved activity and therefore used the supplied deconvolved arrays instead of recomputing the signal from fluorescence.

## 2-b. How is the `neural` data processed?

i. Within each trial, the AI slices the deconvolved traces from all accepted ROIs, concatenates planes, averages consecutive source samples into one common 257.934 ms bin, transposes to neuron-by-time format, and stores the result as `float16`.

ii.
```python
raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
raw = np.concatenate(raw_parts, axis=1)
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
```

iii. In the trajectory, the AI justifies this as a tractability choice: using all sessions and all accepted neurons at native sampling would create a much larger pickle, so it chose block averaging to a common bin size while retaining the supplied deconvolved signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural quality control is limited to Suite2p/manual `iscell` curation from the NWB segmentation table. In multi-plane sessions, the code additionally checks that each plane’s ROI count matches the segmentation-derived mask for that plane.

ii.
```python
seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
iscell = seg['iscell'][:,0] == 1
plane_idx = seg['planeIdx'][:].astype(int) if 'planeIdx' in seg else np.zeros(len(iscell), int)
...
mask = iscell[plane_idx == pi]
if dset.shape[1] != len(mask):
    raise ValueError(f'ROI count mismatch for {name}: {dset.shape[1]} vs {len(mask)}')
plane_series.append((dset, np.flatnonzero(mask)))
```

iii. The trajectory says the AI relied on the NWB `iscell` labels as the accepted-cell curation and patched the code when it discovered multi-plane ROI indexing mismatches. It does not mention any further cell exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from the `trial_start` pulse to the first later teleport sample. No extra offset is applied beyond this trial segmentation and the within-trial block averaging.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
...
for s in starts:
    tq = np.flatnonzero(teleport[s:] > 0)
    e = s + int(tq[0]) if len(tq) else len(pos)
    ...
    raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
```

iii. The trajectory explicitly says the `trial_start` pulse marks the 0 cm crossing and should be the alignment event requested by the instructions, while teleport activity should not be included in a corridor trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 257.934 ms time bin, defined as `4 / 15.5078125` seconds. The code rebins all trials by averaging blocks of source samples into this coarser common bin size.

ii.
```python
TARGET_DT = 4 / 15.5078125
...
dt = float(np.median(np.diff(bt[:min(len(bt), 2000)])))
rate = 1.0 / dt
block = int(round(TARGET_DT / dt))
...
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
```

iii. Early in the trajectory the AI explored using the apparent 15.5 Hz versus 31 Hz metadata, then later audited the timestamps and concluded the shared behavioral timestamps were the authoritative frame spacing. It kept the 257.934 ms target to make every session use the same bin duration and reduce memory usage.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The time-from-start input is derived indirectly from the shared frame interval estimated from `position/timestamps`, together with the number of bins in the trial. The actual per-trial values are generated from bin indices rather than from the raw timestamp vector.

ii.
```python
bt = b['position/timestamps']
dt = float(np.median(np.diff(bt[:min(len(bt), 2000)])))
...
T = len(offsets)
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
```

iii. In the trajectory, the AI says the shared frame timestamps are authoritative and uses them to define one common bin width. Once that width is fixed, it treats time within a trial as a uniform bin index starting at zero.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The code creates a regularly spaced vector `0, TARGET_DT, 2*TARGET_DT, ...` for each trial after rebinnig. It does not subtract the observed first timestamp from each bin.

ii.
```python
T = len(offsets)
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
    np.full(T, ev, np.float32),
    np.full(T, tr, np.float32),
    np.zeros(T, np.float32),
])
```

iii. The trajectory frames this as a consequence of forcing every session onto one common time grid for tractability and cross-session consistency.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input is aligned by construction: it has exactly one value per rebinned neural time bin in the same trial. The code uses the same `offsets`-derived trial length `T` for both neural and input arrays.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
...
T = len(offsets)
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
```

iii. The trajectory states that neural and behavior arrays are aligned by frame index in the NWB, so once the AI rebins all modalities with the same blocks they stay aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the behavior time series `environment/data`.

ii.
```python
env = b['environment/data'][:]
...
ev = int(round(float(np.median(env[s:e]))))
```

iii. In the trajectory, the AI notes that the environment stream is present in the NWB behavior data and should be converted to the requested binary ENV1 versus ENV2 label.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI takes the median environment value over the trial, rounds it, maps positive values to `1` and non-positive values to `0`, and repeats that scalar over all bins in the trial.

ii.
```python
ev = int(round(float(np.median(env[s:e]))))
ev = 1 if ev > 0 else 0
...
np.full(T, ev, np.float32),
```

iii. The trajectory says the raw environment coding looked like a signed value in the NWB, so the AI converted it to a simple binary trial label matching the decoder specification.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the behavior stream `trial number/data`, summarized within each trial by its median value.

ii.
```python
trial_num = b['trial number/data'][:]
...
tr = int(round(float(np.median(trial_num[s:e]))))
...
np.full(T, tr, np.float32),
```

iii. The trajectory says the AI did not use `trial number` to define boundaries because it transitions during teleport, but it still used the within-trial values as the most direct per-trial trial-number label once trials had been segmented by `trial_start`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median trial-number value within the `[trial_start, teleport)` slice is rounded to an integer, repeated across all bins of that trial, and later used to sort trial records chronologically before computing previous-outcome labels.

ii.
```python
tr = int(round(float(np.median(trial_num[s:e]))))
...
inp = np.vstack([
    np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
    np.full(T, ev, np.float32),
    np.full(T, tr, np.float32),
    np.zeros(T, np.float32),
])
...
trial_records.sort(key=lambda x: x[0])
```

iii. In the trajectory, the AI says chronological order matters for previous-outcome assignment, so it sorts by this trial number after trial extraction rather than trusting file iteration order alone.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the reward event timestamps stored in the NWB `Reward` time series.

ii.
```python
rg = b['Reward']
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
```

iii. The trajectory says the AI inspected reward timestamps specifically to determine reward outcome by whether a reward event falls inside a trial interval.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the AI first determines whether the current trial was rewarded by checking whether any reward timestamp falls between the trial start and end times. It stores `(trial number, rewarded, ...)` tuples, sorts them by trial number, then fills the previous-outcome input of each trial with the reward outcome from the preceding chronological trial. The first trial gets `0`.

ii.
```python
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
...
trial_records.append((tr, rewarded, nbin, inp, out))
...
trial_records.sort(key=lambda x: x[0])
prev = 0
for tr, rewarded, nbin, inp, out in trial_records:
    inp[3,:] = prev
    prev = rewarded
```

iii. The trajectory explicitly says “previous outcome is chronological within session (first trial=0)” and that reward should be defined by reward events between trial start and teleport.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from binned position values and from a reward-zone identity inferred from the NWB `identifier` string, not from the `reward_zone` behavior stream. The code parses the identifier to determine starting and ending zone labels and applies a hardcoded switch rule at trial 30.

ii.
```python
scene, z0, z1, switched = scene_info(f['identifier'][()])
...
pos = b['position/data'][:]
trial_num = b['trial number/data'][:]
...
tr = int(round(float(np.median(trial_num[s:e]))))
zone = z1 if (switched and tr >= 30) else z0
lo, hi = ZONE_COORDS[zone]
...
dist_cls = discretize_distance(pbin, lo, hi)
```

iii. The trajectory says the AI investigated the `reward_zone` stream, found values `0..6` that did not directly encode A/B/C, and decided that NWB identifiers preserve the task scene names. It then used the repository’s A/B/C coordinates and the default switch-at-trial-30 convention.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code first averages position within each rebinned block, then computes signed distance to the reward-zone interval: negative before the zone, zero inside it, and positive after it. That continuous value is then discretized.

ii.
```python
def discretize_distance(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
    ...

pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
dist_cls = discretize_distance(pbin, lo, hi)
```

iii. In the trajectory, the AI says it adopted the repository reward-zone coordinates and the task’s requested signed distance bins, using trial-aligned position after rebinning.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The distance is thresholded manually into the seven requested classes: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii.
```python
y[d < -50] = 0
y[(d >= -50) & (d < -10)] = 1
y[(d >= -10) & (d < 0)] = 2
y[d == 0] = 3
y[(d > 0) & (d <= 10)] = 4
y[(d > 10) & (d <= 50)] = 5
y[d > 50] = 6
```

iii. The trajectory says the AI matched the requested decoder discretization after it had fixed the reward-zone coordinates.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance output is aligned by using the same rebinned trial blocks as the neural data. Position is averaged over the same `[s, e)` blocks that generate the rebinned neural activity.

ii.
```python
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
...
dist_cls = discretize_distance(pbin, lo, hi)
```

iii. The trajectory says neural and behavioral streams share frame indexing in the NWB, so the AI rebinned them together to keep them aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the behavior time series `position/data`.

ii.
```python
pos = b['position/data'][:]
...
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
```

iii. The trajectory identifies the `position` stream as the corridor position in centimeters and uses it for both absolute position and distance-to-zone outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI averages position inside each temporal block and then discretizes the block-averaged positions. It does not keep the native per-frame positions.

ii.
```python
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```

iii. The trajectory treats position like the other continuous behavioral variables during its common 257.934 ms rebinning pass.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The position is thresholded with boundaries at `90`, `180`, `270`, and `360` cm, producing five bins across the 450 cm corridor.

ii.
```python
pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
```

iii. The trajectory says the code follows the decoder specification for equal-width position bins over the corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position is aligned with neural data by averaging position over the same fixed blocks used for the neural rebinnig inside each `[trial_start, teleport)` trial slice.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
```

iii. The trajectory says the arrays are already frame-aligned in NWB and were rebinned together on that shared grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the behavior time series `lick/data`.

ii.
```python
lick = b['lick/data'][:]
```

iii. The trajectory identifies the lick stream as a per-frame count-like variable that needs to be turned into the requested binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. For each time bin, the AI marks lick as `1` if any raw lick sample in that block is positive and `0` otherwise.

ii.
```python
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
...
out = np.vstack([
    dist_cls, pos_cls, speed_cls, lbin,
    np.full(T, ZONE_ID[zone], np.int8),
    np.full(T, rewarded, np.int8),
])
```

iii. The trajectory says the raw lick values can exceed one and therefore should be binarized, with any lick inside a rebinned block counting as a lick.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned to neural data by aggregating the raw lick samples over the same trial-local temporal blocks used for neural binning.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

iii. The trajectory says all modalities are aligned by shared frame index, so the same `offsets` vector was used for neural and lick outputs.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB session `identifier` string together with the per-trial `trial number/data`. The code parses the identifier to recover the starting and ending zone letters and uses trial number to choose which zone applies on switch sessions.

ii.
```python
scene, z0, z1, switched = scene_info(f['identifier'][()])
...
trial_num = b['trial number/data'][:]
...
tr = int(round(float(np.median(trial_num[s:e]))))
zone = z1 if (switched and tr >= 30) else z0
```

iii. The trajectory says the AI found that NWB identifiers preserve the scene names and therefore used them instead of the opaque `reward_zone` stream values.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code regex-parses A/B/C letters from the scene name, decides whether the session is fixed or switched, applies the default switch point at trial 30, and maps the resulting A/B/C zone label to integer IDs `0/1/2`.

ii.
```python
def scene_info(identifier):
    ...
    locs = re.findall(r'(?:Location)?([ABC])', scene)
    ...
    z0, z1 = locs[0], (locs[-1] if len(locs) > 1 else locs[0])
    switched = z1 != z0
    return scene, z0, z1, switched
...
zone = z1 if (switched and tr >= 30) else z0
...
np.full(T, ZONE_ID[zone], np.int8),
```

iii. The trajectory says the AI took the switch-at-trial-30 rule from the repository default and task description after deciding that scene metadata were the cleanest source of reward-zone identity.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the NWB `Reward` time series timestamps.

ii.
```python
rg = b['Reward']
reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)
```

iii. The trajectory says the AI explicitly inspected reward event timestamps to determine whether a trial was rewarded or omitted.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI computes the trial’s start and end times from the shared timestamps and sets reward outcome to `1` if any reward event timestamp falls in that interval and `0` otherwise. The label is constant across all bins within the trial.

ii.
```python
t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))
...
np.full(T, rewarded, np.int8),
```

iii. The trajectory states that reward outcome should be based on the presence of a `Reward` event between trial start and teleport.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles irregularities by skipping or failing fast rather than repairing them. It skips clipped final trials with no teleport, skips trials that collapse to fewer than two rebinned bins, skips sessions with fewer than two surviving trials, and raises errors if plane frame rates disagree or if ROI counts do not match the segmentation-derived mask. During development, the AI also patched the converter after finding a multi-plane ROI mismatch and a timestamp-versus-rate inconsistency.

ii.
```python
if not np.allclose(rates, rates[0]):
    raise ValueError(f'Plane frame rates disagree in {path}: {rates}')
...
if dset.shape[1] != len(mask):
    raise ValueError(f'ROI count mismatch for {name}: {dset.shape[1]} vs {len(mask)}')
...
if not len(tq):
    continue
...
if len(offsets) < 2:
    continue
...
if len(nt) < 2:
    print('SKIP (<2 trials)', p, flush=True)
    continue
```

iii. The trajectory shows two concrete examples: the AI stopped on a multi-plane ROI mismatch, inspected the failing file, patched the plane handling, and reran; later it audited a mismatch between stored ophys rates and frame timestamps, patched the bin-size logic to use timestamps, and regenerated the output.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are per-session HDF5 I/O, extracting and concatenating accepted deconvolved traces for every trial, computing rebinned means for neural and behavioral signals, and finally serializing the large pickle output.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    for s in starts:
        ...
        raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
        raw = np.concatenate(raw_parts, axis=1)
        nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
        pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
        sbin = binned_mean(speed[s:e], np.arange(0, e-s, block), block)
...
with open(OUT, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly frames dataset size and I/O cost as a core constraint, including explicit memory estimates that motivated the temporal rebinning choice and repeated discussion of the 1.00 to 1.15 GiB pickle size.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-trial loop over `starts`, the list-comprehension implementation of `binned_mean`, and the lick aggregation loop that applies `np.any` block by block. The code also rebuilds the same block-start arrays several times per trial.

ii.
```python
def binned_mean(x, starts, block):
    return np.stack([np.asarray(x[s:min(s+block, len(x))]).mean(axis=0)
                     for s in starts], axis=0)
...
for s in starts:
    ...
    nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
    pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
    sbin = binned_mean(speed[s:e], np.arange(0, e-s, block), block)
    lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
```

iii. The trajectory emphasizes tractability and simplicity over perfect optimization. The final code keeps the loops explicit, likely because variable-length trials and direct HDF5 slicing are easier to express that way.

## 13-c. What processing does the code repeat multiple times?

i. Within each trial, the code recomputes the block starts repeatedly and separately rebins neural activity, position, and speed. It also creates a temporary `trial_records` list and then sorts it before the final append pass, so trial data are traversed twice after extraction.

ii.
```python
offsets = np.arange(s, e, block, dtype=int)
...
nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
sbin = binned_mean(speed[s:e], np.arange(0, e-s, block), block)
...
trial_records.append((tr, rewarded, nbin, inp, out))
...
trial_records.sort(key=lambda x: x[0])
for tr, rewarded, nbin, inp, out in trial_records:
    ...
```

iii. The trajectory does not describe a separate survey or preprocessing cache. Its main repeated work is the simple per-trial/per-modality binning structure plus the extra sort pass used to guarantee chronological previous-outcome assignment.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs only a small amount of clearly non-decoder work. It parses and stores scene metadata for every session, records `source_frame_rate_hz`, `frames_per_bin`, `n_trials`, and `n_neurons` in `session_info`, and keeps temporary `(trial number, rewarded, neural, input, output)` tuples solely to sort by trial number before appending. None of that metadata are used directly by the downstream decoder.

ii.
```python
trial_records.append((tr, rewarded, nbin, inp, out))
...
trial_records.sort(key=lambda x: x[0])
...
data['metadata']['session_info'].append({
    'file': os.path.relpath(p, DATA_ROOT), 'scene':scene,
    'source_frame_rate_hz':rate, 'frames_per_bin':block,
    'n_trials':len(nt), 'n_neurons':nc})
```

iii. The trajectory presents these choices as bookkeeping and safety measures rather than as required decoder inputs. The main substantive processing is retained in the final dataset; the extra work is mostly metadata capture and ordering safeguards.
