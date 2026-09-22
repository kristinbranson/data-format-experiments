# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans all `.npy` files in the `beh/` directory and loads each as a dictionary. It identifies sessions by regex-matching keys against a session-ID pattern (`^[A-Z0-9]+_\d{4}_\d{2}_\d{2}_[0-9]+(?:_swap[0-9]+)?$`). It then checks which of these behavior sessions have a corresponding neural file in `spk/`. It does NOT use the master index file `Imaging_Exp_info.npy`.

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
common_sessions = [sid for sid in sorted(beh.keys()) if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
```

iii. The AI chose to identify sessions by regex pattern matching on behavior dictionary keys. The CONVERSION_NOTES.md acknowledges that some behavior dict keys are non-session labels (e.g., `naive`, `sup`) and notes the regex filter is needed. However, it does not use the canonical `Imaging_Exp_info.npy` master index that the reference code uses.

## 1-b. How are the data split into subjects?

i. The subject (mouse) name is extracted from the first component of the session ID by splitting on underscores.

ii.
```python
def get_subject(session_id):
    return session_id.split('_')[0]
```

iii. No explicit justification. This is a straightforward approach that works because session IDs begin with the mouse name.

## 1-c. How are the data split into sessions?

i. A session is identified as any behavior dictionary key matching the session-ID regex pattern that also has a corresponding neural data file in `spk/`. Unlike the reference, duplicate sessions appearing under multiple experiment types are not explicitly de-duplicated via a "seen" set; instead, the regex + existence check on neural files implicitly creates the session list. The AI also includes swap sessions (session IDs with `_swap` suffix) in the regex.

ii.
```python
SESSION_RE = re.compile(r'^[A-Z0-9]+_\d{4}_\d{2}_\d{2}_[0-9]+(?:_swap[0-9]+)?$')
common_sessions = [sid for sid in sorted(beh.keys()) if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
```

iii. The AI notes 76 common sessions were found (conversion log shows `n_common_sessions 76`), and one was skipped (TX109_2023_03_27_1), resulting in 75 sessions. The reference solution gets 89 sessions after de-duplication.

## 1-d. How are the data split into trials?

i. Trials are identified from `ft_trInd` — the AI finds unique trial IDs in the frame-level trial-index array and extracts frame indices where `ft_trInd == trial_id`. Crucially, the AI does NOT filter by `ft_CorrSpc` (the corridor-space flag), meaning frames outside the texture corridor (grey space, inter-trial intervals) are included in each trial.

ii.
```python
ft_tr = np.asarray(b['ft_trInd'])
...
valid = ~np.isnan(ft_tr)
tr_ids = np.unique(ft_tr[valid].astype(int))
...
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
    if idx.size < 2:
        continue
```

iii. The CONVERSION_NOTES.md step 5 states "segments trials via `ft_trInd`" but does not mention filtering by `ft_CorrSpc`. No justification is given for including non-corridor frames.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 surviving trials return None. There is no filtering of excessively long trials (no percentile-based cutoff).

ii.
```python
if idx.size < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. No explicit justification for the lack of outlier trial filtering. The resulting data includes very long trials (max T = 5621 frames, ~30 minutes), which the reference removes using a 99th-percentile threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`spk/<session_id>_neural_data.npy`), which contains a list of neuron-by-time matrices per imaging plane. These are concatenated along the neuron axis.

ii.
```python
def load_neural_session(session_id):
    f = SPK_DIR / f'{session_id}_neural_data.npy'
    mats = np.load(f, allow_pickle=True).item()['spks']
    return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. The AI correctly identifies the deconvolved fluorescence traces in `spks`.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are concatenated across imaging planes and cast to float32. For each trial, the columns (frames) corresponding to that trial are extracted. No further processing (e.g., normalization, smoothing) is applied.

ii.
```python
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The CONVERSION_NOTES state "Use Suite2p-derived deconvolved fluorescence traces directly, because methods state analyses are based on deconvolved fluorescence traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. The AI does NOT use the retinotopy data (`iarea` from `retinotopy/` files) to filter neurons by visual area. All neurons from all imaging planes are included. Brain region assignments are fabricated by dividing neurons into three roughly equal blocks.

ii.
```python
data['brain_region_idx'].append(np.concatenate([
    np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 0, dtype=int),
    np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 1, dtype=int),
    np.full(nneu - 2 * (19408 if '2022_07_12_1' in sid else nneu // 3), 2, dtype=int),
]))
```
```python
'brain_regions': ['unknown_block0', 'unknown_block1', 'unknown_block2'],
```

iii. The AI's CONVERSION_NOTES.md step 1 notes "No explicit neuron quality curation function has yet been identified." The AI failed to discover and use the retinotopy data for neuron filtering and brain region assignment.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by selecting frames where `ft_trInd` equals the trial index. However, since `ft_CorrSpc` is not used for filtering, the alignment includes frames before corridor entry and after corridor exit (grey space).

ii.
```python
idx = np.where(ft_tr.astype(float) == float(tr))[0]
...
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. CONVERSION_NOTES mentions "temporal alignment event: trial start / corridor entry" but the code does not ensure frames are restricted to the corridor space.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frames are used directly. The time bin size is computed from the median inter-frame interval of the first session's `ft` timestamps, converted to milliseconds.

ii.
```python
'time_bin_size': float(np.median(np.diff(np.asarray(beh[common_sessions[0]]['data']['ft'])))) * 24 * 3600 * 1000 if common_sessions else None,
```

iii. This approach computes the bin size from the data rather than using the known imaging rate of 3.17 Hz. The result should be approximately ~315 ms, consistent with the reference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute time of the sound cue for each trial) and `Trial_start_time` (the absolute start time of each trial), both in MATLAB datenum format, and `ft` (frame timestamps).

ii.
```python
sound_times = np.asarray(b['SoundTime'])
trial_start_times = np.asarray(b['Trial_start_time'])
...
cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
time_to_cue = (cue_rel - times).astype(np.float32)
```

iii. The AI uses absolute timestamps (`SoundTime`, `Trial_start_time`) rather than frame numbers (`SoundFr`). The reference uses `SoundFr` interpolated onto the frame time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue time relative to trial start is computed as `(SoundTime[trial] - Trial_start_time[trial]) * 24 * 3600` (converting from MATLAB datenums to seconds). The input is then `cue_relative_time - time_since_trial_start`, giving positive values before the cue and negative after, which matches the "time TO cue" semantics.

ii.
```python
cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
time_to_cue = (cue_rel - times).astype(np.float32)
```

iii. No explicit justification given for using `SoundTime`/`Trial_start_time` vs `SoundFr`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are derived from the same frame indices (`idx`), so the time-to-cue values correspond to the same frames as the neural data.

ii.
```python
times = (ft[idx] - t0) * 24 * 3600
...
time_to_cue = (cue_rel - times).astype(np.float32)
```

iii. The alignment is correct by construction since both use the same `idx` frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the behavior file name (e.g., `Beh_sup_train2.npy`, `Beh_unsup_test1.npy`, `Beh_before_learning.npy`). The AI parses the filename to extract the training day.

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
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0
```

iii. No explicit justification. The AI treats the "day" as the experiment-type label rather than the actual ordinal recording day per mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The behavior filename is parsed with regex to extract a numeric day value. `before_learning`/`before_grating` maps to 0, `after_learning`/`after_grating` maps to 1, `trainN` maps to N, `testN` maps to N, and anything else defaults to 0. This value is broadcast across all time bins of a trial.

ii.
```python
day_val = np.full(idx.size, get_day_value(session_id, beh_entry['file']), dtype=np.float32)
```

iii. The reference instead counts the ordinal recording day per mouse (0 for first session, 1 for second, etc.), which gives values up to 7. The AI's approach gives a maximum of 2 (from `train2` or `test2`), and conflates different experiment types.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` (frame timestamps in MATLAB datenum format). The time of the first frame of the trial is used as the reference point.

ii.
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
...
time_since = times.astype(np.float32)
```

iii. No explicit justification.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `(ft[frame] - ft[first_frame_of_trial]) * 24 * 3600`, converting from MATLAB datenums to seconds. This always starts at 0 for the first frame. The reference instead uses `StartFr` (the corridor entry frame) interpolated onto the time axis, which may start slightly before or after 0 depending on when the first imaging frame falls relative to corridor entry.

ii.
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
time_since = times.astype(np.float32)
```

iii. No justification for using the first frame time vs the `StartFr` timestamp.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Both use the same frame indices (`idx`), so they are aligned by construction.

ii.
```python
times = (ft[idx] - t0) * 24 * 3600
time_since = times.astype(np.float32)
```

iii. Aligned by using the same frame index array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether a trial is in the rewarded corridor.

ii.
```python
is_rew = np.asarray(b['isRew']).astype(int)
...
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
```

iii. Consistent with the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` value for the trial is broadcast across all frames of that trial. If the trial index exceeds the `isRew` array length, it defaults to 0.

ii.
```python
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
```

iii. Matches the reference approach. The bounds check is a safety measure not in the reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture for each trial.

ii.
```python
wall_names = np.asarray(b['WallName']).astype(str)
...
trial_wall = wall_names[tr] if tr < len(wall_names) else ft_wall[idx[0]]
stim_cat = np.full(idx.size, global_cat_map.get(str(trial_wall), 0), dtype=np.int64)
```

iii. Consistent with the reference in terms of source variable.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses each individual wall name (e.g., `circle1`, `circle2`, `leaf1`, `leaf2`, `rock1`, `wood1`, etc.) as a separate category. This produces 13 distinct categories. The reference groups these into 4 base texture categories (circle, leaf, rock, wood) using a mapping table.

ii.
```python
def session_category_map(all_wall_names):
    cats = sorted(str(x) for x in np.unique(all_wall_names))
    return {c: i for i, c in enumerate(cats)}, cats
```
```python
'output_values': [cat_names, ...],  # cat_names = ['circle1', 'circle2', 'circle3', 'leaf1', ...]
```

iii. The AI's CONVERSION_NOTES step 5 says "Use per-trial visual category from `WallName` or `stim_id`, preserving actual session-specific categories rather than imposing a global fixed subset." This contradicts the task instructions which say "Visual stimulus category. e.g. circle, leaf, etc." — implying the 4 base texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers of lick events.

ii.
```python
lick_fr = np.asarray(b.get('LickFr', []))
if lick_fr.size:
    lick_fr = lick_fr.astype(int)
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
    lick[lick_fr] = 1
```

iii. Consistent with the reference approach.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array of zeros is created for all frames. Lick frame indices (from `LickFr`) that fall within the valid range are set to 1. Per trial, the lick values at the trial's frame indices are extracted.

ii.
```python
lick = np.zeros(T, dtype=np.int64)
lick_fr = np.asarray(b.get('LickFr', []))
if lick_fr.size:
    lick_fr = lick_fr.astype(int)
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
    lick[lick_fr] = 1
...
lick_out = lick[idx].astype(np.int64)
```

iii. Consistent with the reference.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is in imaging frame indices, and the lick array is indexed with the same frame indices as the neural data, so alignment is by construction.

ii.
```python
lick_out = lick[idx].astype(np.int64)
```

iii. Correctly aligned via shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame.

ii.
```python
ft_pos = np.asarray(b['ft_Pos'])
...
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. Consistent source variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position is divided by 1.0 (floor division) and clipped to the range [0, 3]. This is INCORRECT because `ft_Pos` is in decimeters (0 to ~40 for the 4m texture corridor). Dividing by 1.0 and clipping to 3 means almost all frames map to bin 3 (position >= 3 decimeters, which is 0.3m), not the intended 1m bins.

ii.
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. No justification for the divisor of 1.0. The verification output confirms the severe imbalance: position bin4 contains ~92.4% of all data, confirming the binning is wrong. The reference uses `ft_Pos // 10` (dividing by 10 decimeters = 1 meter).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 1.0, clipped to [0, 3]. This produces 4 bins but with wrong boundaries due to the unit error.

ii.
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. The output values are labeled `['bin1', 'bin2', 'bin3', 'bin4']` without specifying the actual meter ranges. The reference labels them `['0-1 m', '1-2 m', '2-3 m', '3-4 m']`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same frame indices (`idx`) as the neural data.

ii.
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. Aligned by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(b['ft_RunSpeed'])
...
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. Consistent source variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global quantiles (25th, 50th, 75th percentiles) are computed across all included sessions' `ft_RunSpeed` values. `np.digitize` is used to bin each speed value into 4 bins. This differs from the reference, which computes rank-based quartiles per session on kept frames only.

ii.
```python
def discretize_speed(all_speeds):
    qs = np.quantile(all_speeds, [0.25, 0.5, 0.75])
    return qs

def apply_speed_bins(x, qs):
    return np.digitize(x, qs, right=False).astype(np.int64)
```
```python
all_speed = []
for sid in common_sessions:
    b = beh[sid]['data']
    all_speed.extend(np.asarray(b['ft_RunSpeed'])[np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
speed_qs = discretize_speed(np.asarray(all_speed))
```

iii. The AI computes global quantiles across all sessions and all frames (including frames outside kept trials). The reference computes per-session rank-based quartiles on kept frames only, using a rank-ordering method that handles ties at zero speed. The result is unequal bin sizes: the verification output shows `q1: 9.3%, q2: 40.7%, q3: 25.0%, q4: 24.9%`, far from the expected 25% each.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with three global quantile boundaries. Output values labeled `['q1', 'q2', 'q3', 'q4']`.

ii.
```python
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. The reference uses per-session rank-based quartiles, guaranteeing each bin contains exactly 25% of the data within each session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same frame indices (`idx`) as the neural data.

ii.
```python
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. Aligned by construction.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: behavior arrays are clipped to the minimum of neural and behavior lengths; lick frames outside the valid range are filtered; trials with fewer than 2 frames are skipped; sessions that fail conversion or have fewer than 2 trials are skipped. One session (TX109_2023_03_27_1) was skipped without investigation.

ii.
```python
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
neural = neural[:, :T]
...
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
...
if idx.size < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The AI's CONVERSION_NOTES mention that some sessions lacked `BefCueFr`/`AftCueFr` fields, which were initially required but then removed. One skipped session was noted but not investigated.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the large neural data files from disk. Each session's spike file contains three large neuron-by-time matrices that must be loaded and concatenated.

ii.
```python
mats = np.load(f, allow_pickle=True).item()['spks']
return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. The I/O-bound nature of loading large `.npy` files is the primary bottleneck, consistent with the reference analysis.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that extracts frame indices via `np.where(ft_tr == tr)` for each trial could be replaced by a single groupby-like operation. The global speed quantile computation loops through all sessions to build a list.

ii.
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
```

iii. The trial-level loop performs a linear scan of the entire frame array for each trial. This could be replaced by `np.unique(ft_tr, return_inverse=True)` or `np.searchsorted`.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data for each session is loaded once (since `load_behavior_index` loads all behavior files upfront), but the global quantile/category computation iterates over all sessions a second time to collect speed and wall-name data before the main conversion loop processes them again.

ii.
```python
# First pass: collect global stats
for sid in common_sessions:
    b = beh[sid]['data']
    all_wall.extend(...)
    all_speed.extend(...)

# Second pass: convert sessions
for sid in common_sessions:
    converted = convert_session(sid, beh[sid], ...)
```

iii. Two passes over behavior data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes ALL neurons (no retinotopy-based filtering), including neurons outside the four visual areas. These extra neurons add computational cost to the decoder without contributing signal. Additionally, frames outside the corridor space (`ft_CorrSpc = False`) are included, adding noise to the data.

ii.
```python
return np.concatenate(mats, axis=0).astype(np.float32)  # no filtering
```

iii. Including non-visual-area neurons and non-corridor frames increases data size and adds noise that is not useful for the decoder.
