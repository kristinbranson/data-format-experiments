# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by globbing `data/sub-*/*.nwb`, sorts them, and processes each file as one session. In `--sample` mode it only keeps the first two files from that sorted list.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

...

files = get_files(sample=sample)
...
for i, f in enumerate(files, 1):
    sess = process_file(f)
```

iii. In `CONVERSION_NOTES.md`, the AI documented that the released dataset is organized as NWB files under `data/sub-*` and that there are 152 session files across 11 subjects. The trajectory repeatedly states that each `*_behavior+ophys.nwb` file should be treated as one session.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `nwb.subject.subject_id`, with a fallback to the parent directory name. After all sessions are processed, the AI builds a sorted list of unique subject IDs from the processed sessions.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)

...

subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}

...

'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes say the data are organized per subject directory and that 11 subject IDs are present. The AI chose to trust the NWB subject metadata once it had confirmed the folder structure.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A session is included in the final output if, after trial extraction, it still contains at least two trials.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

...

for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```

iii. In `CONVERSION_NOTES.md`, the AI states that each session is a single `*_behavior+ophys.nwb` file and that the released NWB dataset contains 152 such session files.

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` pulses only. Each trial starts at one positive sample in `trial_start` and ends at the next `trial_start`; the final trial runs to the end of the recording. Although `teleport` is loaded, it is not used to end trials.

ii.
```python
def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds

...

trial_start = np.asarray(behavior['trial_start'][0]).ravel()
...
bounds = trial_bounds_from_trial_start(trial_start)
```

iii. The trajectory shows the AI knew `teleport` also marked trial structure, but step 105 says the exact trial-end rule was still uncertain and that the initial implementation would use “`trial_start` and next `trial_start`.”

## 1-e. How are trials filtered based on quality controls?

i. The AI skips candidate trials if they are shorter than 2 samples, if all `trial number` values in the segment are negative, or if all positions are at or below `-100`. At the session level, sessions with fewer than two kept trials are dropped.

ii.
```python
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

iii. In the notes, the AI justified excluding baseline samples because `trial number` and `environment` use `-1` outside the task and some position values are sentinel-like. I did not find an explicit justification for the 2-sample minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data are taken from the deconvolved ophys traces in `processing['ophys']['Deconvolved'].roi_response_series['plane0']`.

ii.
```python
bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']

neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The notes repeatedly state that the paper’s methods use deconvolved activity and that inspection of the NWB files showed the relevant signal under `Deconvolved/plane0`.

## 2-b. How is the `neural` data processed?

i. The AI does almost no additional neural preprocessing. It reads the stored deconvolved traces, casts them to `float32`, and later slices each trial and transposes from `(time, neurons)` to `(neurons, time)`. It does not concatenate multiple planes, normalize, smooth, or otherwise transform the activity.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']

neural = np.asarray(deconv.data[:], dtype=np.float32)

...

trial_neural = neural[s:e, :].T.astype(np.float32)
sess_neural.append(trial_neural)
```

iii. `CONVERSION_NOTES.md` says the deconvolved activity should be used directly on the native synchronized sampling grid. I did not find a separate justification for omitting the ROI-level filtering used in the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply neuron-level quality control. It keeps every column in `plane0` and does not inspect the ROI `iscell` flag.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']

neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. I did not find an explicit justification for skipping `iscell` filtering. The notes focus on the signal choice (`Deconvolved`) rather than cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to `trial_start` by using the trial bounds derived from `trial_start` and then setting time zero at the first sample of each trial.

ii.
```python
bounds = trial_bounds_from_trial_start(trial_start)

...

rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. In the planning notes, the AI explicitly says trials should be aligned to `trial_start` because that is what the task instructions request.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native deconvolved sampling grid. The time bin size is inferred from the deconvolved series rate, and no temporal rebinning is applied.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))

...

'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. `CONVERSION_NOTES.md` says that the behavior and deconvolved signals appear to share a native ~15.5 Hz grid and that the converter should keep that sampling rather than rebinding.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The time input is derived from the deconvolved timestamps if they exist; otherwise the AI synthesizes timestamps from the deconvolved sampling rate using `np.arange(...) / deconv.rate`.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)

...

rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The AI’s notes say the behavior and neural streams appear synchronized on the same grid, so it treated the neural time base as sufficient for alignment.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp in that trial from all timestamps in the trial.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
inp = np.vstack([
    rel_time,
    np.full(e - s, env_label, dtype=np.float32),
    np.full(e - s, tr_label, dtype=np.float32),
    np.full(e - s, prev_rew, dtype=np.float32),
])
```

iii. The notes and trajectory repeatedly describe the decoder inputs as trial-aligned and starting at zero at `trial_start`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI uses the same `[s:e]` sample indices for both the time input and the neural matrix, so the time input is aligned sample-for-sample with the extracted neural trial.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)

...

trial_neural = neural[s:e, :].T.astype(np.float32)
sess_neural.append(trial_neural)
sess_input.append(inp.astype(np.float32))
```

iii. The notes state that the streams are synchronized on a shared grid and that trial slicing with the same indices is sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. `Environment type` comes from the behavior time series `environment`.

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])

...

env = np.asarray(behavior['environment'][0]).ravel()
```

iii. In the trajectory, the AI says it inspected multiple sessions and concluded that valid task values are `0` and `1`, with `-1` used for baseline.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI removes negative values, takes the median of the remaining `environment` samples, rounds it to an integer, and broadcasts that single label across all timepoints in the trial.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0

...

np.full(e - s, env_label, dtype=np.float32),
```

iii. The trajectory says the AI observed that most sessions are constant 0 or constant 1 during valid task periods, with one mixed transition session, so it treated environment as a per-trial label after excluding baseline `-1`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI derives trial number from the behavior time series `trial number`, not from the loop index.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()

...

tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The mapping table in `CONVERSION_NOTES.md` says `trial number` should come from the stored within-session trial-number variable, with invalid baseline samples excluded.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, the AI takes the median of the nonnegative `trial number` samples in that trial. If there are no nonnegative samples, it falls back to the loop counter `ti`. It then broadcasts that scalar over the trial duration.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)

...

np.full(e - s, tr_label, dtype=np.float32),
```

iii. The justification in the notes is indirect: the AI treated `trial number` as the within-session trial label and baseline `-1` values as invalid.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The previous-trial-outcome input is derived from the `Reward` event timestamps. The code first computes whether each extracted trial contains any reward event.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
```

iii. The trajectory explicitly says reward outcome should be recovered from whether a `Reward` event falls within the trial window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI marks the current trial rewarded if any reward timestamp falls between `t0` and `t1`, stores that outcome in `reward_outcomes`, and sets `previous_trial_outcome` to the previous kept trial’s outcome. The first kept trial gets 0.

ii.
```python
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)

...

prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0

...

np.full(e - s, prev_rew, dtype=np.float32),
```

iii. The notes say the variable should be binary and that the first trial can safely default to 0 because there is no previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives distance-to-reward-zone from `position` and an inferred per-trial reward-zone label. That label comes primarily from the `reward_zone` time series, with fallbacks to the reward-delivery position (`Reward` timestamps) or to the median trial position when the zone signal is absent.

ii.
```python
reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
position = np.asarray(behavior['position'][0]).ravel()
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ridx = min(ridx, len(position) - 1)
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
```

iii. In trajectory step 104, the AI says raw `reward_zone` codes 1-6 appear to collapse to three physical locations around ~80-100, ~200, and ~320-330 cm, so it should infer reward-zone identity from physical position rather than directly from the raw code values.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After inferring A/B/C, the AI maps that label to a fixed center point (`90`, `205`, or `325` cm) and computes signed distance as `position - center`. It does not compute distance to the nearest edge of the reward-zone interval.

ii.
```python
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)

zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
```

iii. The notes and sample-validation comments acknowledge that this center-based choice may be why the exact zero-distance bin is sparse or absent.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI thresholds the signed distance into seven bins using the task-specified breakpoints: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `> 50`.

ii.
```python
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

out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. The notes say this part was meant to follow the decoder-task bin specification directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI slices `position` and the neural traces with the same trial bounds `[s:e]`, so distance-to-reward-zone is aligned sample-for-sample with the neural data within each extracted trial.

ii.
```python
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)

...

trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes say the neural and behavior streams share a common time grid, so indexing them by the same samples is enough.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the behavior time series `position`.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()

...

out_pos = pos_bins(position[s:e])
```

iii. The AI treated `position` as the direct VR corridor coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the per-trial position slice and passes it to `pos_bins`, which clips values to `[0, 450)` before binning.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)

...

out_pos = pos_bins(position[s:e])
```

iii. I did not find a fuller written justification beyond the general note that the corridor is about 450 cm long and position should be discretized into five equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses five equal-width bins spanning 0 to 450 cm via `np.linspace(0, 450, 6)`, which gives boundaries at 90, 180, 270, and 360 cm after clipping.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The notes say the AI interpreted “5 equal-sized bins” literally as equal-width bins over the corridor span.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The AI aligns absolute position with neural data by taking the same trial slice `[s:e]` from the position time series and the deconvolved matrix.

ii.
```python
out_pos = pos_bins(position[s:e])

...

trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes say the behavioral streams are synchronized with the neural stream, so the shared indices provide alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output is derived from the behavior time series `lick`.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()

...

out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. The AI’s notes describe `lick` as a task-relevant behavioral stream available at the same sampling grid as the other behavioral variables.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes lick values: any value greater than 0 becomes `1`, otherwise `0`.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. The task instructions ask for lick as a binary output, and the notes mention that the raw lick values can exceed 1.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned with neural data by slicing both streams with the same trial indices `[s:e]`.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)

...

trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes say the neural and behavior streams were treated as synchronized samples on a common grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the `reward_zone` and `position` behavior streams, with `Reward` timestamps used as a fallback when no nonzero `reward_zone` samples appear in a trial.

ii.
```python
reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
position = np.asarray(behavior['position'][0]).ravel()
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
...
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ridx = min(ridx, len(position) - 1)
    center = float(position[ridx])
```

iii. The trajectory says the raw `reward_zone` code values were not directly interpretable as A/B/C, so the AI decided to infer reward-zone identity from the physical location along the corridor.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI first builds a global median position for each nonzero raw `reward_zone` code. Within a trial, it takes the most frequent nonzero code and looks up that median position. If the trial has no active zone samples, it falls back to reward-delivery position or median trial position. It then collapses the resulting position into A/B/C using thresholds `<150`, `<260`, and `>=260`, and broadcasts the label across the trial.

ii.
```python
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
    ...
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)

...

out_rz = np.full(e - s, zone_label, dtype=np.int64)
```

iii. The justification comes from trajectory step 104 and the Step 5 notes: the AI believed raw codes 1-6 corresponded to two environment-specific encodings of three physical reward locations, so it mapped those locations to A/B/C by ascending corridor position.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward` event timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

...

rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. The notes say reward/omission status should be recovered from whether any `Reward` event occurs in the trial window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each extracted trial, the AI checks whether any reward timestamp falls between the trial’s first and last sample times. The result is converted to `0` or `1` and then broadcast across all timepoints in the trial.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))

...

out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. This is the same logic described in the notes for identifying rewarded versus omission trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses heuristics and fallbacks rather than explicit validation. It synthesizes timestamps from the deconvolved rate when timestamps are absent, falls back to the parent folder name if `subject_id` is missing, returns `'unknown'` if region inference fails, skips baseline-like or degenerate trials, and falls back to reward-time position or median trial position when the reward-zone signal is missing. It does not check for mismatched neural/behavior lengths or timestamp misalignment.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)

...

timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)

...

except Exception:
    return 'unknown'

...

if e - s < 2:
    continue
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue

...

elif rew:
    ...
else:
    center = np.nanmedian(position[s:e])
```

iii. The notes justify excluding baseline `-1` values and handling sparse `reward_zone` signals, but the trajectory and notes do not show the stronger consistency checks used in the human reference code.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive parts are reading each NWB file and materializing the full deconvolved and behavioral arrays inside `process_file`, then looping over all extracted trials for all 152 sessions and finally writing the large pickle.

ii.
```python
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
    ...
    neural = np.asarray(deconv.data[:], dtype=np.float32)
    ...
    for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
        behavior[k] = _ts_data(bts[k])

...

for i, f in enumerate(files, 1):
    sess = process_file(f)

...

with open(args.outpickle, 'wb') as f:
    pickle.dump(data, f)
```

iii. In Step 6, the AI explicitly noted that repeated NWB reads across the full dataset could be slow, although it also described the script as a single-pass per-session conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over `bounds` is still scalar Python work, and two per-timepoint binning steps are implemented with Python list comprehensions (`distance_bin` and `speed_bin`). Those are the clearest vectorization opportunities in the AI code.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    ...
    out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
    ...
    out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. I did not find an explicit discussion of these vectorization opportunities in the notes; this is mainly evident from the implementation itself.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code repeats several slice-level computations trial by trial: checking reward timestamps against the current window, recomputing medians for environment and trial number, inferring reward-zone labels, and running the binning logic separately for each trial. It also loads `teleport`, `autoreward`, and `scanning` for every session even though they are not used downstream.

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])

...

for ti, (s, e) in enumerate(bounds):
    ...
    rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
    ...
    env_valid = env[s:e][env[s:e] >= 0]
    ...
    tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
```

iii. The AI did not explicitly call this out in the notes. The repeated work is visible from the per-trial structure of the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI code includes several unused pieces: `contiguous_segments` is defined but never called, `trial_zone_labels` is accumulated but never used, `teleport`, `autoreward`, and `scanning` are loaded but not used in conversion, and `session_id` is stored in the intermediate per-session dict but dropped from the final dataset.

ii.
```python
def contiguous_segments(mask):
    idx = np.flatnonzero(mask)
    ...

...

for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])

...

trial_zone_labels = []
...
trial_zone_labels.append(zone_label)

...

return {
    'subject': subj,
    'session_id': sess_id,
    'region': region,
    ...
}
```

iii. I did not find an explicit justification for these extra computations or unused values in the notes or trajectory.
