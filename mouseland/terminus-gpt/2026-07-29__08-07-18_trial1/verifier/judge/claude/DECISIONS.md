# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavior data from all `.npy` files in `data/beh/`, each containing a dictionary keyed by session IDs. It loads neural data from all `*_neural_data.npy` files found recursively under `data/`. Sessions are matched by finding common keys between behavior and neural data (76 sessions out of 100 behavior sessions and 89 neural files).

ii.
```python
def load_behavior_sessions(data_dir: Path):
    sessions = {}
    session_source = {}
    for p in sorted((data_dir / 'beh').glob('*.npy')):
        obj = np.load(p, allow_pickle=True).item()
        if not isinstance(obj, dict):
            continue
        for sess, dat in obj.items():
            if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
                if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
                    sessions[sess] = dat
                    session_source[sess] = p.name
    return sessions, session_source

def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files
```

iii. The agent iterates over all behavior `.npy` files, filters entries by session ID regex pattern, and handles duplicate sessions by preferring richer entries (more keys, larger `ft_trInd`). Neural files are discovered via recursive glob. Common session IDs between behavior and neural data are used for conversion. This was documented as resolving a bug where poorer duplicates from `example_bef_and_aft_learning_behavior.npy` overwrote richer entries.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from session IDs by taking the first underscore-delimited token (e.g., `TX109` from `TX109_2023_03_27_1`). A sorted list of unique subjects is built.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
# ...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. This is a straightforward extraction from the session naming convention which embeds subject ID, date, and session number.

## 1-c. How are the data split into sessions?

i. Each unique session ID (e.g., `TX109_2023_03_27_1`) with both behavior and neural data constitutes one session. Sessions with fewer than 2 valid trials after processing are excluded.

ii.
```python
common = sorted(set(beh_sessions) & set(neural_files))
# ...
for sess in common:
    neural_trials, input_trials, output_trials = build_session(...)
    if len(neural_trials) < 2:
        continue
```

iii. The agent matched sessions between behavior and neural files by session ID. The minimum-2-trials filter is from the instructions requiring at least two trials per session for decoder evaluation.

## 1-d. How are the data split into trials?

i. Trials are segmented using the frame-level `ft_trInd` array from behavior data, which maps each imaging frame to a trial index. Frames with valid (finite, non-negative) `ft_trInd` values are grouped by trial number.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
# ...
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)

valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:] for i, tr in enumerate(uniq_tr)}
```

iii. The agent correctly uses `ft_trInd` for trial segmentation, matching the reference code pattern `ft_trInd = beh['ft_trInd'][:nfr]`. The agent truncates to `min(spk.shape[1], len(ft_tr), ...)` to handle length mismatches.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on: (1) having at least 2 frames (`idx.size < 2`), and (2) having valid metadata indices (trial index within bounds of `wall`, `is_rew`, `sound`, `tstart` arrays). No filtering based on movement (`ft_move`), corridor space (`ft_CorrSpc`), or neuron quality (d-prime) is applied.

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```

iii. The agent applied minimal quality filtering. The reference code uses `VRmove = beh['ft_move'][:nfr] > 0` and `corr_fr = beh['ft_CorrSpc'][:nfr] & VRmove` to restrict analysis to corridor frames while moving. The agent does not apply these filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in the neural data `.npy` files (`data/spk/*_neural_data.npy`). The `spks` field contains deconvolved fluorescence traces from Suite2p processing.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])

def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
```

iii. The agent loads `spks` and concatenates the list elements (three imaging planes) along axis 0, producing a `(n_neurons, n_frames)` matrix. This matches the reference code at `utils.py:341`: `spk = np.concatenate([nspk for nspk in np.load(...).item()['spks']], 0)`.

## 2-b. How is the `neural` data processed?

i. Neural data is loaded, concatenated across planes, cast to float32, and sliced per trial using frame indices from `ft_trInd`. No further processing (z-scoring, position interpolation, deconvolution) is applied.

ii.
```python
spk = concat_spks(nobj['spks'])
# ...
spk = spk[:, :n_frames]
# ...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The reference code applies additional processing including position interpolation (`get_interpPos_spk` into 60 position bins) and z-scoring (`stats.zscore(spk_norm, axis=1)`), but these are specific to the figure analyses rather than general data loading. The agent uses raw frame-level neural data without interpolation or normalization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. All neurons from all three imaging planes are included. The reference code uses d-prime based selection (`dprime()`) and corridor-responsiveness filtering (`corr_neu`), but the agent does not implement either.

ii.
```python
spk = concat_spks(nobj['spks'])  # all neurons included
# ...
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. The agent's CONVERSION_NOTES.md mentions Suite2p cell classification but does not implement any neuron filtering. The brain region is set to `'unknown'` for all neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by extracting frames belonging to each trial via `ft_trInd`. The first frame of each trial's segment is effectively the trial start.

ii.
```python
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:] ...}
# ...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The agent uses `ft_trInd` to extract per-trial frame segments, which naturally aligns to corridor entry since `Trial_start_time` is documented as "time when animal enters each corridor."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate, estimated per session from trial durations and frame counts. No temporal rebinning is applied. The `time_bin_size` in metadata is set to `None`.

ii.
```python
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
# ...
data['metadata'] = { 'time_bin_size': None, ... }
```

iii. The agent estimates `dt_frame` from MATLAB datenum differences (converted to seconds via `* 86400.0`) divided by frame count per trial. This is used for computing time-varying inputs/outputs but no rebinning of neural data is applied. Setting `time_bin_size` to `None` rather than the computed value is a metadata gap.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundTime` (per-trial sound cue timestamp) and `Trial_start_time` (per-trial corridor entry timestamp), both in MATLAB datenum format.

ii.
```python
sound = np.asarray(beh['SoundTime'], dtype=float)
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
# ...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
```

iii. The agent correctly identifies `SoundTime` and `Trial_start_time` as the source variables.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `sound_time_in_seconds - time_since_trial_start` for each frame, where `sound_time_in_seconds = (SoundTime - Trial_start_time) * 86400`. This produces a time-varying signal that decreases as the trial progresses, becoming zero at the sound cue and negative afterward.

ii.
```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. This is a reasonable computation — it represents the signed time remaining until the sound cue at each frame. When `SoundTime` is NaN (no sound cue), the entire trial gets NaN values.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-sound signal is computed at each frame within the trial, using `time_since = np.arange(idx.size) * dt_frame` as the time axis. This is naturally aligned with the neural data since both use the same frame indices from `ft_trInd`.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32)
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. The alignment is implicit — the input array has the same number of time points as the neural array for each trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is inferred from the **behavior source filename** (e.g., `Beh_sup_train1_before_learning.npy`), not from the data itself.

ii.
```python
def infer_day(source_name: str):
    name = source_name.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 1.0
    if 'after_learning' in name or 'after_grating' in name:
        return 5.0
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0
```

iii. The agent uses heuristic filename parsing. "before_learning" maps to day 1, "after_learning" to day 5, and test/train numbers are extracted directly.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Day of training is a scalar per trial, broadcast to match the number of time points. The value is derived from the behavior filename heuristic described above.

ii.
```python
day = infer_day(beh_source)
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. The methods state training lasted exactly 5 days (day 1: passive, days 2-5: active). The agent maps "before_learning" to 1.0 and "after_learning" to 5.0, which is a rough approximation. Test sessions get their test number, and sessions that don't match any pattern get 0.0.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The agent does **not** include "Environment type" as a decoder input. The `input_names` are `['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']`.

ii.
```python
data['input_names'] = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
```

iii. The instructions list four decoder inputs: Time to sound cue, Day of training, Time since trial start, and Reward availability. "Environment type" is not listed in the instructions as a decoder input, so the agent did not include it. However, environment type (supervised vs. unsupervised vs. naive) could be a relevant contextual variable that the reference paper discusses.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable — Environment type is not included as a decoder input by the agent.

ii. N/A

iii. The instructions do not list environment type as a decoder input, so its omission is consistent with the task specification.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the frame index within each trial and the estimated frame period `dt_frame`.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. `dt_frame` is estimated from `(Trial_end_time - Trial_start_time) * 86400 / n_frames_per_trial`, using the median across trials in each session.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A simple linear ramp from 0 to `(n_frames - 1) * dt_frame` seconds. The frame period `dt_frame` is estimated per session as the median of `trial_duration_seconds / n_frames_in_trial`.

ii.
```python
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
# ...
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. The agent computes a per-session median frame period rather than using per-trial actual timestamps from `VRposTime`. This is an approximation that assumes uniform frame rate.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The time-since-trial-start array has length equal to the number of neural frames in that trial, so alignment is implicit via shared indexing.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. Same frame indices used for both neural and input data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the `isRew` array in the behavior data, which is a per-trial boolean/integer indicating whether the corridor is rewarded.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
# ...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. `isRew` is directly available in the behavior data and matches the instruction: "1 if in rewarded corridor, 0 if not."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is broadcast to all frames within the trial as a constant signal.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. Straightforward — the reward availability is constant within a trial (determined by corridor type).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from the `WallName` array in the behavior data, which stores the stimulus name (e.g., `'leaf1'`, `'circle1'`) for each trial.

ii.
```python
wall = np.asarray(beh['WallName'])
# ...
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
# ...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),  # placeholder, filled later
    lick, pos_bin, speed_bin,
])))
# ...
out[0, :] = wall_to_id[wall_name]
```

iii. The agent collects all unique wall names across sessions, creates a sorted mapping to integer IDs, and assigns the per-trial wall name as a constant per-trial output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name for each trial is mapped to an integer category index. The category is constant across all time points within a trial. The output values list contains the sorted unique wall names.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
# output_values[0] = wall_names = ['circle1', 'circle2', 'leaf1', 'leaf2', 'rock1', 'wood1']
```

iii. The agent uses `WallName` directly rather than `stim_id` or `get_cat_id()` from the reference code. The result is similar but bypasses the reference's category mapping logic.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickTrind` (trial index per lick event) and `LickTime` (timestamp per lick event), plus `Trial_start_time` for temporal alignment.

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
```

iii. The agent uses event-based lick data to construct a binary time series.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick events are converted to a binary frame-level time series. For each lick event, the relative time from trial start is computed, converted to a frame index, and the corresponding frame is marked as 1 (licking).

ii.
```python
lick_series = np.zeros(n_frames, dtype=np.int64)
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
if lick_tr.size and lick_time.size:
    lick_tr_i = lick_tr.astype(int)
    for tr in np.unique(lick_tr_i):
        idx = trial_frame_idx.get(int(tr))
        if idx is None or idx.size == 0 or tr >= len(tstart):
            continue
        rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            continue
        bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
        lick_series[idx[np.unique(bins)]] = 1
```

iii. The agent converts lick event timestamps to relative time from trial start, then maps to frame indices using `dt_frame`. Multiple licks in the same frame are collapsed to a single 1. This approach constructs a binary time series as required by the instructions.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary series is indexed by the same frame indices as the neural data for each trial.

ii.
```python
lick = lick_series[idx]  # idx = trial_frame_idx[tr]
```

iii. Since `lick_series` is computed at the full-session frame level and then sliced using the same trial frame indices as neural data, alignment is maintained.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_PosCum` (frame-level cumulative position) and `Corridor_Length`.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
```

iii. The agent uses the frame-aligned cumulative position, matching the reference code's use of `ft_AcumPos = beh['ft_PosCum'][:nfr]`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Cumulative position is taken modulo corridor length to get within-corridor position, then discretized into 4 equal-length bins.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. The agent applies `np.mod` to get within-corridor position, then divides into 4 bins of 1 m each (for a 4 m corridor). This matches the instruction: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided into 4 bins of equal length: `bin = floor(pos / corridor_len * 4)`, clipped to range [0, 3].

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. Four bins cover [0, 1m), [1m, 2m), [2m, 3m), [3m, 4m], which matches "4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted at the same frame indices as neural data for each trial.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. `idx` is the same frame index array used for neural data, ensuring alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_PosCum` (frame-level cumulative position) via finite differences and the estimated frame period `dt_frame`.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Speed is computed from position differences, not from a direct speed variable in the data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is computed as `diff(within_corridor_position) / dt_frame` per frame. The first frame uses `prepend=pos[0]`, effectively giving speed 0 for the first frame.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Note: Speed is computed from `np.mod(ft_pos, corridor_len)` (within-corridor position), not from cumulative position. This means speed at corridor boundaries (where position wraps from ~4m to ~0m) will produce large negative artifacts. Computing speed from cumulative position before the modulo operation would avoid this issue.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized into 4 bins based on quartiles computed across all trials within a session. `np.digitize` with quartile boundaries [Q25, Q50, Q75] produces bins 0-3.

ii.
```python
all_speeds.append(speed)
# ...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0])
# ...
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The instructions specify "Running speed discretized into 4 bins, each corresponding to 25% of the data." The agent computes quartiles per session rather than globally across all sessions. This means each session has its own speed distribution, so the 25% split applies within each session rather than across the entire dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed at the same frame indices as neural data for each trial.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Same frame alignment as position and neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases:
- Non-finite `ft_trInd` values are marked as invalid (`ft_tr_int = -1`)
- Neural/behavior length mismatches are handled by truncating to the minimum length
- Missing `LickTrind`/`LickTime` default to empty arrays
- Missing `Corridor_Length` defaults to 4.0
- Non-finite `SoundTime` or `Trial_start_time` produces NaN for time-to-sound
- Non-finite lick relative times are filtered out
- Trials with fewer than 2 frames are skipped
- Trials with out-of-bounds metadata indices are skipped
- Duplicate behavior sessions are resolved by preferring the richer entry

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
# ...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
# ...
lick_tr = np.asarray(beh.get('LickTrind', []))
# ...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
# ...
rel = rel[np.isfinite(rel)]
```

iii. The agent documents the duplicate-session fix in CONVERSION_NOTES.md Step 10. Other edge cases are handled implicitly in code.

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural data files is the most time-consuming step. Each neural file contains ~50k-90k neurons x ~20k-30k frames of float32 data. The full conversion took 2177 seconds (~36 minutes) for 76 sessions.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. The agent's CONVERSION_NOTES.md reports ~41 seconds per session for sample conversion, with estimated full conversion of ~52 minutes. Actual full conversion ran in ~36 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick processing loop iterates over unique lick trial indices to compute relative lick times and frame bins. This could potentially be vectorized.

ii.
```python
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    if idx is None or idx.size == 0 or tr >= len(tstart):
        continue
    rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
    # ...
```

iii. Additionally, the `frame_periods` estimation loop iterates over all trials to compute per-trial frame counts using `np.sum(ft_tr_int == tr)`, which could be replaced by `np.bincount`.

## 12-c. What processing does the code repeat multiple times?

i. The code iterates over trials twice: once in the main `for tr, idx in trial_frame_idx.items()` loop to compute all per-trial data and accumulate speeds, and then again in `for tr_spk, inp, wall_name, lick, pos_bin, speed in cache` to apply speed binning. This two-pass approach is intentional (to compute global quartiles before binning).

ii.
```python
cache = []
for tr, idx in trial_frame_idx.items():
    # ... compute all trial data, append to cache and all_speeds
cache.append((tr_spk, inp, str(wall[tr]), lick, pos_bin, speed))
all_speeds.append(speed)
# ...
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False)
```

iii. The two-pass approach is reasonable for computing data-dependent quantiles but does require storing all trial data in memory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent processes the `ft_move` and `ft_WallID` arrays but only uses `ft_WallID` for truncation (not for filtering or selection). `ft_move` is loaded and truncated but never used for filtering. The reference code uses `ft_move > 0` as a movement mask, but the agent does not apply it.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
# ft_move and ft_wall are truncated but ft_move is never used for filtering
# ft_wall is never used after truncation
```

iii. Loading `ft_move` and `ft_wall` adds minor I/O overhead but they are not used in the processing pipeline. The `--show-processing` flag is accepted but not implemented.
