# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `data/sub-*/*.nwb`, then processes each NWB file one at a time with `NWBHDF5IO`. All subject/session/trial content is discovered from those files rather than from any precomputed manifest.

ii. ```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

...

with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. In `CONVERSION_NOTES.md`, the AI says the released dataset is organized as NWB files under per-subject directories and that there are 152 session files across 11 subjects. It treats one NWB file as one session and uses a single-pass NWB reader for conversion.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by NWB file provenance: the code uses `nwb.subject.subject_id` when present, otherwise the parent `sub-*` directory name. The final `subjects` list is the sorted set of subjects across processed sessions.

ii. ```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)

...

subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes say the data are organized under per-subject `sub-*` directories and that there are 11 subjects in the full release, so the AI treated subject identity as file-level metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The code stores one converted session object per file and keeps only sessions with at least two retained trials.

ii. ```python
def process_file(fpath):
    with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
        nwb = io.read()
        sess_id = nwb.session_id
        ...
        return {
            'subject': subj,
            'session_id': sess_id,
            ...
        }

...

for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```

iii. In the notes, the AI explicitly states that each session is a single `*_behavior+ophys.nwb` file.

## 1-d. How are the data split into trials?

i. Trials are segmented from the `trial_start` behavioral time series only. The AI uses each positive `trial_start` sample as a trial onset and closes the trial at the next `trial_start`, or at the end of the recording for the last trial. It does not use `teleport` to end trials.

ii. ```python
def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds

...

bounds = trial_bounds_from_trial_start(trial_start)
for ti, (s, e) in enumerate(bounds):
    ...
```

iii. `CONVERSION_NOTES.md` says trial structure is recovered from behavior time series and lists a key decision to use `trial_start` as onset and the next `trial_start` as the main delimiter.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only with lightweight heuristics: skip trials shorter than 2 samples, skip windows whose `trial number` is entirely negative, and skip windows whose positions are all sentinel-like (`<= -100`). Sessions are kept if at least two trials survive. There is no `< 50` timepoint filter.

ii. ```python
for ti, (s, e) in enumerate(bounds):
    if e - s < 2:
        continue
    if np.nanmax(trial_num[s:e]) < 0:
        continue
    if np.all(position[s:e] <= -100):
        continue

...

if len(sess['neural']) >= 2:
    sessions.append(sess)
```

iii. The notes justify baseline exclusion with negative trial/environment codes and sentinel positions, and separately justify keeping sessions with at least two trials after segmentation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the NWB `processing['ophys']['Deconvolved'].roi_response_series['plane0']` array.

ii. ```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The notes state that the reference text uses deconvolved activity as the neural signal and that the script was built around that interface.

## 2-b. How is the `neural` data processed?

i. The AI performs very little processing: it reads the deconvolved matrix, casts it to `float32`, then slices it per trial and transposes each trial from `(time, neurons)` to `(neurons, time)`. It does not apply the reference `iscell` filter or any smoothing/rebinning.

ii. ```python
neural = np.asarray(deconv.data[:], dtype=np.float32)

...

trial_neural = neural[s:e, :].T.astype(np.float32)
sess_neural.append(trial_neural)
```

iii. The notes describe the intended neural signal as already-processed deconvolved activity on a shared native time grid, so the AI chose a minimal transformation pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The converted neural data are not filtered by cell-quality metadata. The code keeps every column in `plane0`.

ii. ```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes emphasize subject/session/trial handling and deconvolved traces, but they do not record an `iscell`-based neuron-curation step. The implementation reflects that omission.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data are aligned by defining each trial window from a `trial_start` event and slicing the neural matrix with the same `[s:e]` indices. No separate alignment transform is applied.

ii. ```python
bounds = trial_bounds_from_trial_start(trial_start)

...

trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes say the decoder should be aligned to `trial_start`, and the script uses `trial_start` as the temporal anchor for every trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native deconvolved sampling rate. The time bin size is recorded as `1000 / rate` ms, and no temporal rebinning is applied.

ii. ```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))

...

'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. The notes say behavior and deconvolved traces appear synchronized on a shared native grid at about 15.5 Hz, so the AI preserved that resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the deconvolved time base: if neural timestamps exist they are used; otherwise the code synthesizes timestamps from sample index and `deconv.rate`. In this dataset, that means `np.arange(T) / rate`.

ii. ```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)

...

rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The notes say the native shared sampling grid should be used. Because the deconvolved series lacks explicit timestamps, the AI backed out time from the stored rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp in the sliced window is subtracted so that every trial starts at 0 seconds.

ii. ```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The notes state that all trials should be aligned to `trial_start`, so the AI used relative time from the start of each segmented window.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the same `[s:e]` indices are applied to the neural matrix and the timestamp array, and the per-trial relative-time vector is built from that identical slice.

ii. ```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes repeatedly state that behavior and deconvolved traces share a common native grid, so the AI did not add any interpolation step.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the behavioral `environment` time series.

ii. ```python
env = np.asarray(behavior['environment'][0]).ravel()
...
env_valid = env[s:e][env[s:e] >= 0]
```

iii. The notes identify `environment` as the NWB source variable and say valid task samples use codes `0` and `1`, with `-1` reserved for baseline.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code removes negative values within each trial, takes the median of the remaining samples, rounds it to an integer label, and broadcasts that label across the whole trial. If no valid sample exists, it falls back to 0.

ii. ```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
...
np.full(e - s, env_label, dtype=np.float32)
```

iii. The notes justify baseline exclusion and say valid task samples should use environment codes `0/1` directly, which is why the code discards negative values first.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the behavioral `trial number` time series, not from the loop index. The code takes the within-trial median of nonnegative `trial number` values.

ii. ```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The notes say the dataset contains a `trial number` time series with baseline `-1` outside task epochs, so the AI used that field once invalid values were excluded.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Within each trial, negative samples are discarded, the median remaining trial number is computed, and that scalar is broadcast across all timepoints in the trial. If no valid sample exists, the loop index `ti` is used.

ii. ```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
...
np.full(e - s, tr_label, dtype=np.float32)
```

iii. The notes frame `trial number` as a per-trial context variable, so the AI turned it into a constant-within-trial input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the behavioral `Reward` event timestamps. For each trial, the code determines whether the current trial contained any reward event and stores that binary outcome for later use on the next trial.

ii. ```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
```

iii. The notes explicitly say reward outcome should be recovered from `Reward` events and used to distinguish rewarded and omission trials.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code computes the current trial’s reward outcome first, appends it to `reward_outcomes`, then sets the previous-trial input to the prior entry in that list. The first trial gets 0. The result is broadcast across time.

ii. ```python
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)

...

prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
...
np.full(e - s, prev_rew, dtype=np.float32)
```

iii. The notes say previous outcome should be a per-trial binary context variable with a safe default on the first trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from `position`, `reward_zone`, and sometimes `Reward` timestamps. The code first infers a coarse A/B/C reward-zone label for the trial, then measures each position sample relative to a fixed center for that coarse zone.

ii. ```python
position = np.asarray(behavior['position'][0]).ravel()
reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
...
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)

zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
```

iii. The notes justify reward-zone mapping by physical position rather than raw reward-zone code, and say reward delivery can be used as a fallback clue when reward-zone samples are missing.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code infers a trial-level reward-zone center, computes signed distance from each position sample to that center, and then bins the resulting scalar distances with the custom `distance_bin` function. It does not compute distance to the nearest reward-zone edge.

ii. ```python
def distance_bin(d):
    if d < -50:
        return 0
    if d < -10:
        return 1
    if d < 0:
        return 2
    if d == 0:
        return 3
    if d <= 10:
        return 4
    if d <= 50:
        return 5
    return 6

...

zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. The notes say the AI collapsed reward-zone identity to three physical corridor locations and then derived reward-relative outputs from those inferred locations.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code uses seven categories matching the requested cut points: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `> 50`.

ii. ```python
def distance_bin(d):
    if d < -50:
        return 0
    if d < -10:
        return 1
    if d < 0:
        return 2
    if d == 0:
        return 3
    if d <= 10:
        return 4
    if d <= 50:
        return 5
    return 6
```

iii. The notes say the target decoder outputs should use the specification in the instructions, so the bin cutoffs were hard-coded to that scheme.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data are aligned by slicing both with the same trial indices `[s:e]`, then computing distance on that slice.

ii. ```python
d = position[s:e] - zc
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes say behavior and deconvolved traces live on the same native grid, so the AI aligned outputs by shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the behavioral `position` time series.

ii. ```python
position = np.asarray(behavior['position'][0]).ravel()
...
out_pos = pos_bins(position[s:e])
```

iii. The notes identify `position` as the direct source for corridor location outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code clips position values into `[0, 450)` and discretizes them into 5 equal bins over that range using `np.linspace(0, 450, 6)` and `np.digitize`.

ii. ```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The notes describe absolute position as a 5-bin corridor variable and treat the physical corridor span as roughly 0 to 450 cm.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into five equal-width bins with edges at `0, 90, 180, 270, 360, 450` after clipping.

ii. ```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The notes say the task requires 5 equal-sized bins, so the AI encoded a simple uniform partition of the corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by using the same trial slice `[s:e]` as the neural data before discretization.

ii. ```python
out_pos = pos_bins(position[s:e])
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes justify shared-index alignment on the common native sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the behavioral `lick` time series.

ii. ```python
lick = np.asarray(behavior['lick'][0]).ravel()
...
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. The notes list `lick` among the decoder-relevant behavioral variables available in NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick values are binarized: any value greater than 0 becomes 1, otherwise 0.

ii. ```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. The notes say lick should be a binary decoder target and that the raw time series contains non-binary values.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same per-trial slice `[s:e]` used for the neural data.

ii. ```python
out_lick = (lick[s:e] > 0).astype(np.int64)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes rely on the shared native time grid, so no separate alignment step was used.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived mainly from `reward_zone` and `position`, with a fallback to `Reward` timestamps when the reward-zone trace is absent within a trial.

ii. ```python
reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
position = np.asarray(behavior['position'][0]).ravel()
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
...
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The notes explicitly say reward-zone identity should be inferred from the physical position of active reward-zone samples, with reward delivery used as a fallback when needed.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Within each trial, the AI takes the modal positive `reward_zone` code if one exists, maps that code to a session-wide median position, and collapses that position to coarse labels A/B/C using thresholds `<150`, `<260`, else `C`. If no positive reward-zone samples exist, it falls back to reward position or trial median position.

ii. ```python
def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        pos = pos[np.isfinite(pos)]
        if len(pos):
            centers[int(code)] = float(np.median(pos))
    return centers

def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0
    if pos < 260:
        return 1
    return 2

...

if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ridx = min(ridx, len(position) - 1)
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The notes justify this with a physical-position interpretation of reward-zone identity and a desire to collapse raw codes into the three corridor reward locations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the behavioral `Reward` event timestamps.

ii. ```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. The notes say rewarded versus omission trials should be recovered from `Reward` events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward timestamp falls between the trial’s start and end times and then broadcasts that binary result across all timepoints in the trial.

ii. ```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))

...

out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. The notes explicitly describe reward outcome as a per-trial binary label recovered from reward events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles a few edge cases with simple fallbacks: invalid baseline trials are excluded using negative `trial number` or sentinel positions; missing valid environment or trial-number samples fall back to `0` or the loop index; missing reward-zone samples fall back to reward position, trial median position, or a default 225 cm center; missing imaging-plane annotations fall back to `'unknown'`. There is no neural/behavior length-cropping step or short-trial filter like the reference code uses.

ii. ```python
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue

...

env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)

...

elif rew:
    ...
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)

...

except Exception:
    return 'unknown'
```

iii. The notes justify excluding baseline-coded samples and using fallback physical-position heuristics for missing reward-zone information.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading each NWB file, materializing large deconvolved/behavior arrays, iterating over every trial in each session, and writing the final pickle. The code processes the full dataset in a single pass, so file I/O dominates.

ii. ```python
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
    ...

for ti, (s, e) in enumerate(bounds):
    ...

with open(args.outpickle, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes explicitly mention that repeated NWB reads are slow and estimate conversion time per session, which points to NWB I/O as the main bottleneck.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop is still the main serial structure. Within each trial, the list comprehensions used for `distance_bin` and `speed_bin` are also vectorizable. `reward_zone_centers` iterates over codes and recomputes masks, which could likewise be vectorized or cached more tightly.

ii. ```python
for code in sorted(c for c in np.unique(reward_zone) if c > 0):
    pos = position[reward_zone == code]
    ...

for ti, (s, e) in enumerate(bounds):
    ...
    out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
    out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. The notes say the script uses vectorized slicing but still processes sessions trial-by-trial, so these are the obvious remaining scalar loops.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly slices the same arrays per trial and repeatedly applies scalar binning logic for distance and speed. It also recomputes reward-zone center information session-by-session rather than persisting any intermediate survey product.

ii. ```python
zone_centers = reward_zone_centers(position, reward_zone)

for ti, (s, e) in enumerate(bounds):
    ...
    d = position[s:e] - zc
    out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
    out_pos = pos_bins(position[s:e])
    out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. The notes explicitly describe the implementation as a single-pass per-session conversion, so the repeated work is mostly within-session slicing and binning rather than a duplicated whole-dataset survey pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed or loaded but not used downstream: `contiguous_segments` is dead code, `teleport`/`autoreward`/`scanning` are loaded but unused, `trial_zone_labels` is accumulated but never returned, and `session_id`/the initial zero `brain_region_idx` do not affect the final saved tensors.

ii. ```python
def contiguous_segments(mask):
    ...

for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])

...

trial_zone_labels = []
...
trial_zone_labels.append(zone_label)

...

region_idx = np.zeros(n_neurons, dtype=np.int64)
return {
    'session_id': sess_id,
    ...
    'brain_region_idx': region_idx,
}
```

iii. The notes do not justify these items; they appear to be leftovers from exploration or placeholders for more detailed processing that was never used in the final output.
