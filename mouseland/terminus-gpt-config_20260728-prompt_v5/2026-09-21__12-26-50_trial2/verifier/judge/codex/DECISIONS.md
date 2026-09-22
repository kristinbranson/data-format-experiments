# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans every behavior file in `/app/data/beh`, loads each `.npy` object dict, and keeps entries whose keys match a session-like regex. It then defines usable sessions as those behavior keys that also have a matching spike file in `/app/data/spk`. It does not use `Imaging_Exp_info.npy` as a master index, and it does not load retinotopy when building the dataset.

ii. 
```python
def load_behavior_index():
    beh = {}
    for f in sorted(BEH_DIR.glob('*.npy')):
        obj = np.load(f, allow_pickle=True).item()
        for k, v in obj.items():
            if is_session_key(k):
                beh[k] = {'file': f.name, 'data': v}
    return beh
```
```python
beh = load_behavior_index()
common_sessions = [sid for sid in sorted(beh.keys()) if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
```

iii. The notes say behavior dicts contain non-session keys, so the agent justified regex filtering of keys as a cleanup step. In trajectory step 377 it also justified indexing behavior sessions across files and then trimming neural and behavior to a common length. There is no explicit justification for ignoring `Imaging_Exp_info.npy` or retinotopy here.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the prefix before the first underscore in each session id, and the final `subjects` list is the sorted set of those prefixes among converted sessions.

ii. 
```python
def get_subject(session_id):
    return session_id.split('_')[0]
```
```python
subjects = sorted({get_subject(sid) for sid in common_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes argue that naive parsing of behavior keys was contaminated by non-session keys, so the AI relied on cleaned session ids and then used the session-id prefix as subject identity. No more elaborate justification was given.

## 1-c. How are the data split into sessions?

i. Sessions are defined by regex-matching behavior keys and then intersected with spike filenames. In practice, a session is any key already formatted like `MOUSE_YYYY_MM_DD_BLOCK` for which `<session_id>_neural_data.npy` exists.

ii. 
```python
SESSION_RE = re.compile(r'^[A-Z0-9]+_\d{4}_\d{2}_\d{2}_[0-9]+(?:_swap[0-9]+)?$')
```
```python
if is_session_key(k):
    beh[k] = {'file': f.name, 'data': v}
```
```python
common_sessions = [sid for sid in sorted(beh.keys()) if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
```

iii. The notes justify this as a way to ignore metadata-like keys in the behavior files. In Step 5 they explicitly planned to treat only “session-ID-like keys” as sessions. There is no explicit justification for not using the reference master index or for not deduplicating repeated recordings via `Imaging_Exp_info.npy`.

## 1-d. How are the data split into trials?

i. Trials are recovered from the frame-level trial index `ft_trInd`. For each unique non-NaN trial id after trimming to the common time axis, the script takes every frame whose `ft_trInd` equals that trial id. It does not additionally restrict frames to `ft_CorrSpc`.

ii. 
```python
valid = ~np.isnan(ft_tr)
tr_ids = np.unique(ft_tr[valid].astype(int))
```
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
    if idx.size < 2:
        continue
```

iii. Trajectory step 377 says the agent found `ft_trInd` and planned to “segment trials using ft_trInd or StartFr/EndFr.” The notes also present trial segmentation by `ft_trInd` as a straightforward common-grid alignment strategy. No explicit justification was given for dropping the `ft_CorrSpc` restriction used by the reference.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is skipped only if it has fewer than two frames after trimming to the shared time axis. A whole session is skipped if fewer than two trials survive. There is no trial-length outlier filter.

ii. 
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
    if idx.size < 2:
        continue
```
```python
if len(neural_trials) < 2:
    return None
```

iii. The trajectory and notes do not give a substantive trial-QC rationale beyond needing at least two trials for decoder training. No justification was recorded for omitting the reference trial-length exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` list in each session’s `<session_id>_neural_data.npy` file. The three matrices are concatenated along the neuron axis.

ii. 
```python
def load_neural_session(session_id):
    f = SPK_DIR / f'{session_id}_neural_data.npy'
    if not f.exists():
        return None
    mats = np.load(f, allow_pickle=True).item()['spks']
    return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. The notes say the neural files contain a dict with key `spks` and that each session has three large matrices sharing a common time axis. The recorded rationale was to use the deconvolved traces directly and concatenate the three blocks.

## 2-b. How is the `neural` data processed?

i. The three `spks` matrices are concatenated, cast to `float32`, trimmed to the shortest shared length with behavior frame arrays, and then sliced into per-trial matrices using the frames selected from `ft_trInd`.

ii. 
```python
neural = load_neural_session(session_id)
...
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
neural = neural[:, :T]
```
```python
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The notes justify using deconvolved calcium activity directly because the methods said analyses used deconvolved traces. Trajectory step 377 justifies trimming neural and behavior arrays to a common length. The AI did not record any additional neural preprocessing rationale.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not meaningfully filtered. All neurons in the concatenated `spks` matrices are kept. The script never consults retinotopy or removes non-visual-area neurons.

ii. 
```python
def load_neural_session(session_id):
    ...
    mats = np.load(f, allow_pickle=True).item()['spks']
    return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. The notes say additional neuron curation still needed to be determined, but the final code never implements it. Later trajectory comments acknowledge that brain regions were placeholders, which implies the agent knew the neural curation/annotation was incomplete.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code treats the first retained frame of each `ft_trInd` trial segment as time zero for that trial and extracts the neural columns for exactly those frames. It labels the alignment event as “trial start / corridor entry” in metadata, but the actual code does not use `StartFr` or `ft_CorrSpc` to enforce corridor-entry alignment.

ii. 
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
    ...
    t0 = ft[idx[0]]
    times = (ft[idx] - t0) * 24 * 3600
```
```python
'temporal_alignment_event': 'trial start / corridor entry',
```

iii. The notes and trajectory repeatedly say all streams should be aligned to trial start / corridor entry. The implemented rationale seems to be that `ft_trInd` provides an already aligned per-frame trial segmentation, but the code does not add a corridor-space check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the original frame resolution. The metadata time-bin size is estimated from the median inter-frame interval of the first retained session’s `ft` array. No temporal rebinning is applied.

ii. 
```python
'time_bin_size': float(np.median(np.diff(np.asarray(beh[common_sessions[0]]['data']['ft'])))) * 24 * 3600 * 1000 if common_sessions else None,
```

iii. The notes say the frame-level behavior arrays are already aligned to neural timepoints and present this as a reason not to resample. No separate temporal rebinning rationale was recorded.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the per-trial absolute cue times `SoundTime`, the per-trial absolute start times `Trial_start_time`, and the frame timestamps `ft`.

ii. 
```python
sound_times = np.asarray(b['SoundTime'])
trial_start_times = np.asarray(b['Trial_start_time'])
```
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
...
cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
```

iii. Trajectory step 377 explicitly highlights `SoundTime` and `Trial_start_time` as available trial-level arrays and uses them as the basis for time-varying inputs. The notes frame this as using explicit cue timing rather than inferring cue time from frame numbers.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code computes elapsed time since the first retained frame, computes cue time relative to `Trial_start_time`, and then subtracts the per-frame elapsed time from that cue offset. If the trial index exceeds the available cue array length, the whole input stays `NaN`.

ii. 
```python
time_to_cue = np.full(idx.size, np.nan, dtype=np.float32)
if tr < len(sound_times):
    cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
    time_to_cue = (cue_rel - times).astype(np.float32)
```

iii. The recorded rationale was pragmatic: use the explicit trial-level cue timestamp and a continuous countdown variable aligned to the frame times. There is no explicit justification for not interpolating cue onset from `SoundFr`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same frame indices `idx` used for the per-trial neural slice, so the cue-timing vector has one value per neural time bin in that trial.

ii. 
```python
times = (ft[idx] - t0) * 24 * 3600
...
neural_trials.append(neural[:, idx].astype(np.float32))
input_trials.append(inp)
```

iii. The notes repeatedly say all streams are aligned on the neural/behavior frame grid after trimming to common length. This is the main justification given.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from a true raw-data day counter. Instead, it is heuristically derived from the behavior filename string, using keywords like `before_learning`, `after_learning`, `trainN`, and `testN`.

ii. 
```python
def get_day_value(session_id, beh_file):
    name = beh_file.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 0.0
    if 'after_learning' in name or 'after_grating' in name:
        return 1.0
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
```

iii. The only explicit justification is later in trajectory step 400, where the agent itself calls day-of-training “a heuristic derived from filenames.” The notes had earlier considered session ordering by filenames/condition files, but the final implementation never computes a per-mouse session order.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The filename is lowercased and mapped to a float: `before_* -> 0.0`, `after_* -> 1.0`, `trainN -> N`, `testN -> N`, else `0.0`. That scalar is then broadcast across all time bins of the trial.

ii. 
```python
day_val = np.full(idx.size, get_day_value(session_id, beh_entry['file']), dtype=np.float32)
```

iii. The notes do not provide a strong defense of this processing. The only recorded justification is that it was a quick heuristic based on experimental-condition filenames.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the frame timestamps `ft` and the first retained frame of each `ft_trInd` trial segment. Although `Trial_start_time` is loaded for cue timing, it is not used to compute this input.

ii. 
```python
ft = np.asarray(b['ft'])
ft_tr = np.asarray(b['ft_trInd'])
```
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
time_since = times.astype(np.float32)
```

iii. The notes and trajectory justify alignment on the shared frame grid and trial segmentation by `ft_trInd`. There is no explicit justification for using the first retained frame instead of `StartFr` or `Trial_start_time` here.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the script subtracts the timestamp of the first retained frame from every frame timestamp in that trial and converts the difference from days to seconds.

ii. 
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
time_since = times.astype(np.float32)
```

iii. The trajectory shows the agent debugging a previous time-unit bug and then keeping this relative-time calculation. The main justification recorded is that it produces sensible ranges and is aligned to the trial frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built from the same trial frame indices `idx` used to slice the neural matrix, so it has one value per retained neural frame.

ii. 
```python
times = (ft[idx] - t0) * 24 * 3600
...
neural_trials.append(neural[:, idx].astype(np.float32))
input_trials.append(inp)
```

iii. The notes state that all time-varying inputs/outputs should be constructed on the same trimmed frame grid as the neural data. That is the recorded justification.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` array.

ii. 
```python
is_rew = np.asarray(b['isRew']).astype(int)
```
```python
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
```

iii. The notes planned to use rewarded-corridor information as a binary per-trial input, and `isRew` is the field the final code uses. No further justification was needed or recorded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is almost no processing: the trial’s `isRew` flag is cast to `int`, converted to float, and broadcast across all time bins of that trial.

ii. 
```python
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
```

iii. The notes describe reward availability as a binary per-trial variable. The implementation follows that plan directly, with no extra transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is primarily derived from the per-trial `WallName` array. If the trial index is out of range for `WallName`, it falls back to the frame-level `ft_WallID` at the first frame of the trial.

ii. 
```python
wall_names = np.asarray(b['WallName']).astype(str)
ft_wall = np.asarray(b['ft_WallID']).astype(str)
```
```python
trial_wall = wall_names[tr] if tr < len(wall_names) else ft_wall[idx[0]]
```

iii. The notes identify `WallName` and `stim_id` as candidate sources and emphasize preserving the actual labels observed in the data rather than assuming a fixed canonical set. The final code chooses `WallName` with an `ft_WallID` fallback.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI builds a single global category map from all unique `WallName` strings observed across included sessions, sorts those strings, maps each label to an integer id, and then broadcasts the chosen trial label across all frames of that trial. It does not collapse variants like `circle1` and `circle2` to a shared base texture.

ii. 
```python
def session_category_map(all_wall_names):
    cats = sorted(str(x) for x in np.unique(all_wall_names))
    return {c: i for i, c in enumerate(cats)}, cats
```
```python
cat_map, cat_names = session_category_map(np.array(all_wall))
...
stim_cat = np.full(idx.size, global_cat_map.get(str(trial_wall), 0), dtype=np.int64)
```

iii. The notes justify this as preserving “actual per-session categories rather than assume all global categories appear in every session.” The final implementation only partially matches that rationale: it preserves raw wall labels, but with one global map over all included sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the frame-index array `LickFr`.

ii. 
```python
lick = np.zeros(T, dtype=np.int64)
lick_fr = np.asarray(b.get('LickFr', []))
if lick_fr.size:
    lick_fr = lick_fr.astype(int)
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
    lick[lick_fr] = 1
```

iii. The notes planned to convert lick timestamps into binary time bins. The final code uses the frame-index version of lick timing directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The lick frames are truncated to integers, clipped to the valid frame range, and converted into a binary per-frame vector where any frame containing at least one lick is `1`.

ii. 
```python
lick_fr = lick_fr.astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
lick[lick_fr] = 1
```
```python
lick_out = lick[idx].astype(np.int64)
```

iii. The notes explicitly say licking should become a binary time-varying series. No stronger justification was recorded beyond that.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is indexed with the same per-trial frame indices `idx` used for the neural matrix, so the output has one lick label per retained neural time bin.

ii. 
```python
lick_out = lick[idx].astype(np.int64)
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The notes and trajectory repeatedly justify using the shared frame grid after trimming all streams to common length. That is the alignment rationale here.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level position array `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(b['ft_Pos'])
...
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. The notes planned to use corridor position and discretize it into four equal bins. The final code uses `ft_Pos` directly.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code floor-divides `ft_Pos` by `1.0`, converts to integer, and clips the result to `[0, 3]`. Because `ft_Pos` is not rescaled before binning, this effectively thresholds on raw `ft_Pos` units rather than on four 1 m bins.

ii. 
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. The notes say the intended rule was four equal-length 1 m bins. There is no recorded justification for the final `// 1.0` implementation.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are the integer boundaries induced by `ft_Pos // 1.0`, followed by clipping to category ids `0, 1, 2, 3`. So values from `0` to `<1` map to bin `0`, `1` to `<2` to bin `1`, `2` to `<3` to bin `2`, and everything `>=3` maps to bin `3`.

ii. 
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. No explicit justification for these thresholds appears in the notes or trajectory. The code simply implements this categorical clipping.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position labels are taken on the same per-trial frame indices `idx` as the neural slices, so the position sequence is framewise aligned to the neural data.

ii. 
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The notes justify using frame-level variables aligned to the neural frames after trimming to common length. That is the relevant justification here.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level running-speed array `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(b['ft_RunSpeed'])
```
```python
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. The notes planned to use running speed and discretize it into quartile-like bins. The final code uses `ft_RunSpeed` directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before session conversion, the code pools all finite `ft_RunSpeed` values from all included sessions, computes the 25th/50th/75th percentiles with `np.quantile`, and then digitizes each trial’s framewise speeds against those global thresholds.

ii. 
```python
def discretize_speed(all_speeds):
    qs = np.quantile(all_speeds, [0.25, 0.5, 0.75])
    return qs
```
```python
for sid in common_sessions:
    ...
    all_speed.extend(np.asarray(b['ft_RunSpeed'])[np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
...
speed_qs = discretize_speed(np.asarray(all_speed))
```
```python
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. The notes explicitly planned to use “global quartiles over included samples,” so the final implementation follows that plan. The trajectory also reports the computed `speed_quantiles` as a sanity check.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The category thresholds are the three global numeric quartiles returned by `np.quantile(all_speeds, [0.25, 0.5, 0.75])`. `np.digitize` then maps each speed value into bins `0` through `3`.

ii. 
```python
qs = np.quantile(all_speeds, [0.25, 0.5, 0.75])
```
```python
def apply_speed_bins(x, qs):
    return np.digitize(x, qs, right=False).astype(np.int64)
```

iii. The notes justify quartile binning as matching the decoder-task requirement. They do not discuss ties or per-session versus global thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running-speed values are sampled on the same per-trial frame indices `idx` used for the neural matrix, then discretized on that aligned vector.

ii. 
```python
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. As with other time-varying variables, the notes justify alignment by keeping everything on the common frame grid after trimming.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles length mismatches by trimming neural and frame-level behavior streams to a shared minimum length `T`. It drops lick frames outside `[0, T)`, ignores `NaN` trial ids when enumerating trials, and skips sessions that lack a spike file or end up with fewer than two surviving trials. It does not add more specialized missing-data logic.

ii. 
```python
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
neural = neural[:, :T]
ft = ft[:T]
ft_tr = ft_tr[:T]
ft_pos = ft_pos[:T]
ft_speed = ft_speed[:T]
ft_wall = ft_wall[:T]
```
```python
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
```
```python
valid = ~np.isnan(ft_tr)
tr_ids = np.unique(ft_tr[valid].astype(int))
```

iii. Trajectory step 377 explicitly calls out the near-match but small offset between neural and behavior frame counts and justifies trimming to common length to handle that mismatch carefully. Beyond that, there is little explicit discussion of missing data.

## 12-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify full-session neural slicing and handling very large neuron-by-time matrices as the main cost. Reading and concatenating the full `spks` matrices for each session is the obvious hotspot in the code.

ii. 
```python
def load_neural_session(session_id):
    ...
    mats = np.load(f, allow_pickle=True).item()['spks']
    return np.concatenate(mats, axis=0).astype(np.float32)
```
```python
for sid in common_sessions:
    ...
    converted = convert_session(sid, beh[sid], speed_qs, cat_map, max_trials=(20 if args.sample else None))
```

iii. In Step 6 of the notes, the AI explicitly wrote that “Full-session neural slicing is expensive due to very large neuron-by-time matrices.” That is the recorded justification.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest repeated loop is the per-trial scan `np.where(ft_tr.astype(float) == float(tr))`, which rescans the trial-index vector for every trial. The session loop that repeatedly extends `all_speed` and `all_wall` is also straightforward but less central.

ii. 
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
```
```python
for sid in common_sessions:
    b = beh[sid]['data']
    all_wall.extend([str(x) for x in np.asarray(b['WallName']).astype(str)])
    all_speed.extend(np.asarray(b['ft_RunSpeed'])[np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
```

iii. The notes mention code inefficiency but do not name this exact loop. The conclusion comes from the final code structure rather than an explicit justification entry.

## 12-c. What processing does the code repeat multiple times?

i. Inside the per-trial loop, the code repeatedly casts `ft_tr` to float and rescans it to find the current trial frames. It also recomputes `get_day_value(session_id, beh_entry['file'])` for every trial even though that value is session-constant.

ii. 
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
```
```python
day_val = np.full(idx.size, get_day_value(session_id, beh_entry['file']), dtype=np.float32)
```

iii. There is no explicit justification for this repeated work in the notes or trajectory. It appears to be an implementation convenience rather than a documented decision.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores placeholder brain-region assignments (`unknown_block0/1/2`) instead of real retinotopy-based labels, loads `ft_WallID` mainly as a fallback that is usually unused, and constructs global wall-label categories even though downstream decoding only needs integer class ids. It also computes sample-selection scores used only for `--sample`.

ii. 
```python
ft_wall = np.asarray(b['ft_WallID']).astype(str)
```
```python
'brain_regions': ['unknown_block0', 'unknown_block1', 'unknown_block2'],
...
data['brain_region_idx'].append(np.concatenate([
    np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 0, dtype=int),
    np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 1, dtype=int),
    np.full(nneu - 2 * (19408 if '2022_07_12_1' in sid else nneu // 3), 2, dtype=int),
]))
```
```python
if args.sample:
    scored = []
    for sid in common_sessions:
        ...
        scored.append((rew_n > 0, lick_n > 0, rew_n, lick_n, sid))
```

iii. The trajectory explicitly acknowledges that brain regions are placeholders. Otherwise the notes do not defend these extra computations; they seem to have been added for convenience or to satisfy the target schema.
