# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file under `data/sub-*/*.nwb`, treats each file as one session, and processes them one by one with `pynwb.NWBHDF5IO`.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

def process_file(fpath):
    with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
        nwb = io.read()
```

iii. `CONVERSION_NOTES.md` Step 2 says the data are organized as per-subject NWB files and Step 6 says the script performs “single-pass per-session processing.” The trajectory also states the plan was to “load NWB across sessions” rather than reconstruct the paper’s intermediate session objects.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB file parent directory / subject metadata. The final dataset uses `nwb.subject.subject_id` when present, otherwise the directory name.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. Step 2 of `CONVERSION_NOTES.md` says the dataset is “organized as NWB files under per-subject directories `data/sub-*`,” so the agent treated directory/NWB subject identity as the subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session.

ii.
```python
files = sorted(Path('data').glob('sub-*/*.nwb'))
...
for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```

iii. `CONVERSION_NOTES.md` Step 2 says “Each session is a single `*_behavior+ophys.nwb` file,” and Step 5 explicitly maps one file to one session.

## 1-d. How are the data split into trials?

i. Trials are defined from each positive `trial_start` sample to the next positive `trial_start` sample, or to the end of the recording for the last trial. The code does not use `teleport` to end trials.

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
bounds = trial_bounds_from_trial_start(trial_start)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the planned trial boundary was “each `trial_start` as onset and the next `trial_start` ... as trial end,” and the trajectory repeats that the initial script would implement “trial segmentation using `trial_start` and next `trial_start`.”

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps a trial unless it has fewer than 2 samples, has only negative `trial number`, or has positions entirely at sentinel values `<= -100`. Sessions are kept if at least two trials remain.

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

iii. `CONVERSION_NOTES.md` Step 5 lists “baseline exclusion” for samples with negative trial/environment values or sentinel positions, and Step 5/6 say sessions should have at least two trials for decoder evaluation. I did not find a note justifying the very small `e - s < 2` threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from the NWB `Deconvolved` signal and only reads `plane0`.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 5 say the methods “use deconvolved activity” and that the planned mapping was `processing['ophys']['Deconvolved'].roi_response_series['plane0'] -> neural`. The notes do not mention reconstructing the paper’s deconvolution from fluorescence.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: the agent reads the deconvolved matrix, optionally uses stored timestamps or derives them from the stored rate, and slices it into per-trial matrices transposed to `(neurons, time)`.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Step 6 says the script “loads deconvolved ophys activity” and “uses vectorized slicing on synchronized time series.” No additional neural preprocessing justification appears in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply any neuron-level quality-control filtering. It uses all columns from `Deconvolved/plane0`.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
...
n_neurons = neural.shape[1]
```

iii. The notes describe dataset-level and trial-level checks, but I found no explicit neuron-filtering rationale in `CONVERSION_NOTES.md`. The omission is consistent with the final script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by slicing each trial from the `trial_start`-defined bounds and making time 0 the first sample in that slice.

ii.
```python
bounds = trial_bounds_from_trial_start(trial_start)
...
for ti, (s, e) in enumerate(bounds):
    ...
    trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Step 5 states “Temporal alignment: Align trials to `trial_start`, matching the decoder task requirement,” and the trajectory repeats that alignment choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native `Deconvolved` sample rate with no rebinned resampling. The metadata time bin is `1000 / rate` ms from the `plane0` series.

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
...
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. `CONVERSION_NOTES.md` Step 5 says “Use the native shared sampling grid (~15.5 Hz),” and Step 6 says the script loads deconvolved activity without extra resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the deconvolved neural timestamps (`deconv.timestamps`) or an inferred time vector from `deconv.rate`.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
...
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. Step 5 says the agent would use the “native shared sampling grid” because behavior and deconvolved traces appeared synchronized. That is the only explicit justification I found.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp in the slice is subtracted so that the first sample is 0 s.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
inp = np.vstack([
    rel_time,
    ...
])
```

iii. This follows the Step 5 alignment plan: “Align trials to `trial_start`.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same sample indices `s:e` are used for time and neural data, so the time vector is on the same per-trial grid as the neural matrix.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 4 says behavior and ophys share timestamps/sample intervals, and Step 5 says the script would use the “native shared sampling grid.”

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior time series `environment`.

ii.
```python
env = np.asarray(behavior['environment'][0]).ravel()
```

iii. Step 5 maps `environment -> input[1] environment_type`, and the notes say task-valid codes are `0/1` with baseline `-1`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, negative values are dropped, the median valid environment code is rounded to an integer, and that scalar is broadcast across all time points in the trial.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
...
np.full(e - s, env_label, dtype=np.float32),
```

iii. Step 5 says environment codes `0/1` can be “used directly” after excluding baseline `-1`. The trajectory additionally notes one mixed session and treats the trial label as effectively constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavior time series `trial number`, not from the per-trial loop index.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. Step 5 maps `trial number -> input[2] trial_number` and says “Use within-session trial index.” The final code operationalizes that with the median of the stored `trial number` values inside each trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Negative values are ignored, the median remaining `trial number` is taken as the trial label, and that scalar is broadcast across time within the trial; if no valid values exist, the loop index `ti` is used.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
...
np.full(e - s, tr_label, dtype=np.float32),
```

iii. I did not find a more detailed justification in the notes beyond the plan to use the within-session trial index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward-event timestamps in the behavior `Reward` series.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. Step 4 and Step 5 both state that reward outcome should be recovered from whether a `Reward` event occurred within a trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The current trial’s reward outcome is computed as any reward timestamp between `t0` and `t1`. For the decoder input, the previous retained trial’s reward outcome is used, with the first retained trial set to 0 and the result broadcast across the trial.

ii.
```python
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
...
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
...
np.full(e - s, prev_rew, dtype=np.float32),
```

iii. Step 5 says “previous trial reward outcome from `Reward` events” and “First trial may use 0.” The trajectory also describes “previous reward” as one of the planned decoder inputs.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus a per-trial inferred reward-zone label based mainly on `reward_zone`, with fallback to reward-event position or trial median position.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ...
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
```

iii. Step 5 says reward-zone codes `1-6` should be collapsed to A/B/C “by the physical position of active reward-zone samples or reward delivery, not by raw code alone.”

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent first maps the inferred zone to one of three hard-coded centers (`90`, `205`, `325` cm). It then computes signed distance from position to the zone center, not to the nearest edge of a reward-zone interval.

ii.
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. The notes repeatedly frame the reward-zone problem in terms of physical zone location clusters around ~80, ~200, and ~320 cm. I did not find a note acknowledging the difference between zone center and zone edges.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is thresholded with a custom piecewise function implementing the bins `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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
```

iii. Step 5 says distance-to-zone should be “discretized categorical 0-6 per task spec,” so the agent encoded those thresholds directly in Python instead of using `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[s:e]` using the same per-trial sample indices used for neural slices.

ii.
```python
d = position[s:e] - zc
...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Step 5 says all variables should be built on the shared sampling grid after trial slicing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior time series `position`.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
...
out_pos = pos_bins(position[s:e])
```

iii. Step 5 maps `position -> output[1] absolute_position_bin`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, 450)` inside `pos_bins`, then digitized into five equal bins.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. Step 5 says absolute position should be “5 equal bins across corridor.” The clipping behavior is implicit in the final helper function; I did not find a specific written justification for it.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm corridor is split into 5 equal-width bins via `np.linspace(lo, hi, 6)` and `np.digitize`.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
```

iii. This follows the Step 5 mapping and the task instructions for 5 equal bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position bins are computed from the same per-trial `s:e` slice used for the neural matrix.

ii.
```python
out_pos = pos_bins(position[s:e])
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Step 5’s general plan was to use the synchronized trial slices for both behavior and neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior time series `lick`.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
```

iii. Step 5 maps `lick -> output[3] lick`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick signal is binarized: values greater than 0 become `1`, otherwise `0`.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Step 5 says “Convert lick values >0 to 1 for decoder output.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same trial indices `s:e` as the neural data.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Step 5’s alignment plan uses one shared time grid across variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from `reward_zone` and `position`, with reward-event position as a fallback when `reward_zone` is absent within a trial.

ii.
```python
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
    ...
    center = float(position[ridx])
```

iii. Step 5 says the raw reward-zone codes `1-6` should be collapsed to A/B/C by physical position rather than raw code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. A per-trial reward-zone center is inferred, then `collapse_zone_position_to_abc` maps that physical location to A/B/C using thresholds `<150`, `<260`, or higher. The resulting label is broadcast across the trial.

ii.
```python
def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0
    if pos < 260:
        return 1
    return 2
...
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
out_rz = np.full(e - s, zone_label, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 says reward-zone identity should be inferred by corridor position because the raw codes appear to collapse onto three physical locations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward-event timestamps in the behavior `Reward` series.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. Step 4 and Step 5 both state that trial reward outcome should be determined from whether a `Reward` event falls within the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward timestamp falls between the trial start and end timestamps, converts that to `0/1`, and broadcasts it across the trial.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
...
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. Step 5 says “1 if reward event occurs within trial,” which is exactly how `rew` is computed.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles a few edge cases by skipping clearly invalid trials and using fallbacks for missing reward-zone information. Specifically, it skips trials that are too short or contain only invalid trial-number/sentinel-position values; if reward-zone samples are missing, it falls back to reward-event position or the trial’s median position.

ii.
```python
if e - s < 2:
    continue
if np.nanmax(trial_num[s:e]) < 0:
    continue
if np.all(position[s:e] <= -100):
    continue
...
if len(rz_nz):
    ...
elif rew:
    ...
else:
    center = np.nanmedian(position[s:e])
```

iii. Step 5’s “Baseline exclusion” note justifies excluding negative trial/environment periods and sentinel positions. I did not find explicit notes about handling mismatched neural/behavior lengths or missing timestamps.

## 13-a. What are the most time-consuming steps of the code?

i. The main expensive steps in the implemented code are repeated NWB file I/O, reading full deconvolved/behavior arrays for each session, per-trial Python loops, and saving the large pickle.

ii.
```python
for i, f in enumerate(files, 1):
    sess = process_file(f)
...
neural = np.asarray(deconv.data[:], dtype=np.float32)
for ti, (s, e) in enumerate(bounds):
    ...
with open(args.outpickle, 'wb') as f:
    pickle.dump(data, f)
```

iii. `CONVERSION_NOTES.md` Step 6 says “Repeated full NWB reads may be slow” and estimates conversion cost per session. The full conversion log shows every session being read and processed sequentially.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_file` and the list-comprehension binning for distance and speed could have been vectorized more aggressively. The outer loop over session files is also strictly sequential.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    ...
    out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
    out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. Step 6 claims “vectorized slicing” was used, but the final code still leaves several Python loops in place. I did not find a more detailed efficiency discussion for these specific loops.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats similar per-trial slicing for neural, inputs, and outputs inside one loop. It also repeatedly computes small trial-local summaries such as environment median, trial-number median, and reward-zone inference separately for every trial.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    ...
    env_valid = env[s:e][env[s:e] >= 0]
    tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
    ...
    trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The notes emphasize single-pass session processing, so there is less repeated whole-dataset work than in the reference pipeline. I did not find an explicit self-critique of repeated trial-local computation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script defines `contiguous_segments` but never uses it, appends `trial_zone_labels` without returning it, reads some behavior streams (`teleport`, `autoreward`, `scanning`) without using them downstream, and stores `session_id`/`region_idx` values that are not otherwise used in the final assembly logic.

ii.
```python
def contiguous_segments(mask):
    ...

for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])

trial_zone_labels = []
...
trial_zone_labels.append(zone_label)
```

iii. I did not find a written justification for these extras in the notes. They appear to be leftovers from exploration or from earlier planned logic.
