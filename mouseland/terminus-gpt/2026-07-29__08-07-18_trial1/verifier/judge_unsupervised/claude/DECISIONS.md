# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavioral data from all `.npy` files in `data/beh/`, each containing a dictionary mapping session IDs to per-session behavioral dictionaries. It also loads neural data from `data/spk/*_neural_data.npy` files. Sessions are matched by session ID between behavior and neural files, yielding 76 common sessions.

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

# In main():
common = sorted(set(beh_sessions) & set(neural_files))
```

iii. The AI documented that behavior files are organized as dictionaries mapping session IDs to per-session data, and neural files are named by session ID. It filters behavior keys using a regex to select session-like IDs and handles duplicate sessions across files by preferring the richer entry (based on key count and `ft_trInd` size). The approach matches the reference code's pattern of loading `.npy` files and matching sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subject IDs are extracted from session names by splitting on underscore and taking the first element (e.g., `TX83_2022_08_17_1` -> `TX83`). A sorted list of unique subjects is created and each session is mapped to its subject index.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
# ...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
```

iii. The AI observed that session naming convention embeds subject ID as the first component before the date. This matches the reference code and data structure.

## 1-c. How are the data split into sessions?

i. Each session corresponds to one behavior dictionary entry matched to one neural data file. The session is the natural unit of organization. Sessions are processed independently in a loop over `common` session IDs.

ii.
```python
for sess in common:
    neural_trials, input_trials, output_trials = build_session(beh_sessions[sess], beh_source[sess], neural_files[sess])
    if len(neural_trials) < 2:
        continue
```

iii. The AI found 76 sessions with both behavior and neural data. Sessions with fewer than 2 valid trials are excluded (the minimum for decoder validation).

## 1-d. How are the data split into trials?

i. Trials are identified using the `ft_trInd` field which assigns each imaging frame to a trial index. The AI finds unique trial indices in the valid (finite) `ft_trInd` values and creates a mapping from trial index to frame indices. Each trial is a set of contiguous frames.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
# ...
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)
# ...
valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:] for i, tr in enumerate(uniq_tr)}
```

iii. The AI noted that `ft_trInd` is a frame-time aligned trial index field. This is consistent with the reference code which uses `ft_trInd` to segment frames into trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) requiring at least 2 frames (`idx.size < 2`), (2) the trial index must be within the bounds of the behavioral arrays (`tr >= len(wall)`, etc.), and (3) sessions must have at least 2 valid trials. No filtering based on corridor space (`ft_CorrSpc`) or movement (`ft_move > 0`) is applied.

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```

iii. The AI documented minimal trial filtering. The reference code uses `VRmove = beh['ft_move'] > 0` and `corr_fr = beh['ft_CorrSpc'] & VRmove` to filter frames to only corridor space and moving frames. The AI does not apply these filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in `*_neural_data.npy` files. `spks` is a list of arrays (typically 3 elements, corresponding to imaging planes), each of shape `(n_neurons_per_plane, n_timepoints)`.

ii.
```python
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])

def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
```

iii. The AI confirmed from the reference code (utils.py line 341) that `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)`. This concatenation along axis 0 indicates the list elements are neuron blocks (planes), not trials.

## 2-b. How is the `neural` data processed?

i. The neural data (deconvolved spikes from Suite2p) is concatenated across planes and then sliced by trial frame indices. No additional processing is applied -- no z-scoring, no neuropil correction beyond what Suite2p already did, no spatial binning, and no normalization.

ii.
```python
spk = concat_spks(nobj['spks'])
# ...
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
# ...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. The AI treated `spks` as already-deconvolved neural activity traces from Suite2p, consistent with the paper stating "All our analyses were based on deconvolved fluorescence traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons from all planes are included. The reference code filters neurons that respond more in corridor than gray space (`corr_neu = spk[:, stim1_fr].mean(1) > spk[:, grey_fr].mean(1)`), but the AI does not implement this.

ii.
```python
# No neuron filtering code -- all neurons from concat_spks are included
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. The AI noted from reference code that neuron curation rules existed but did not implement them, stating "Suite2p cell classification was used; exact downstream neuron inclusion criteria not yet identified."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). The `ft_trInd` field indexes frames belonging to each trial, and the first frame of each trial is treated as time 0. No explicit time offset or pre-trial window is included.

ii.
```python
# Trial frames are extracted directly from ft_trInd mapping
tr_spk = spk[:, idx].astype(np.float32, copy=False)
# time_since = np.arange(idx.size) * dt_frame starts at 0
```

iii. The AI documented `temporal_alignment_event: 'corridor entry / trial start'` and `off_start: 0.0` in metadata, consistent with the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate, approximately 315 ms (~3.2 Hz). No temporal rebinning is applied. The AI sets `time_bin_size: None` in metadata.

ii.
```python
# dt_frame computed per session from trial duration / frame count:
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1

# metadata:
'time_bin_size': None,
```

iii. The AI noted that the imaging frame rate was not explicitly stated in the methods excerpt. It computes the frame period empirically. Setting `time_bin_size` to `None` rather than the computed value (~315 ms) is an oversight in documentation, though the actual data uses the native frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundTime` (per-trial sound cue timestamp), `Trial_start_time` (per-trial start timestamp), and the computed frame timing (`dt_frame`).

ii.
```python
sound = np.asarray(beh['SoundTime'], dtype=float)
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
# ...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
```

iii. The AI uses absolute timestamps from the behavior data, converting from MATLAB serial date number format (multiply by 86400 to get seconds).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound cue time is converted to seconds relative to trial start, then subtracted from the `time_since_trial_start` series to produce a time-varying signal that counts down to (and up from) the sound cue.

ii.
```python
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. This produces a decreasing time-varying signal that is positive before the sound cue and negative after it. If sound time is not available, NaN is used.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are frame-aligned: `time_since_trial_start` is computed as `arange(n_frames) * dt_frame`, and `time_to_sound = sound_sec - time_since`. The result has the same number of time points as the neural data for each trial.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32)
```

iii. Naturally aligned through shared frame indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the behavior source file name (e.g., `Beh_unsup_train1_before_learning.npy`), NOT from any explicit training day field in the data.

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

iii. The AI inferred training day from file naming conventions. Files named `before_learning` or `before_grating` are assigned day 1, and `after_learning` or `after_grating` are assigned day 5 (based on the paper stating training lasts exactly 5 days). The `test` and `train` file suffixes with numbers provide day indices. This is a heuristic approach.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day is inferred once per session from the source filename and broadcast as a constant value across all time points of every trial in that session.

ii.
```python
day = infer_day(beh_source)
# ...
day_arr = np.full(idx.size, day, dtype=np.float32)
```

iii. Day of training is a per-trial constant, replicated across time bins to match the temporal dimension of other inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from frame indices within each trial and the computed `dt_frame` (frame period).

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. No raw data variable is directly used; it is reconstructed from the frame count and estimated frame duration.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A linear ramp starting at 0, incrementing by `dt_frame` for each frame in the trial. `dt_frame` is estimated per-session as the median of (trial_duration / n_frames_per_trial).

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. Simple linear time series construction.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Inherently aligned -- both have the same number of time points per trial, indexed by the same frame indices.

ii.
```python
# idx is the same set of frame indices used for both neural and time_since
tr_spk = spk[:, idx]
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. Frame-level alignment through shared indexing.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew`, a per-trial boolean array indicating whether each trial is in a rewarded corridor.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
# ...
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. `isRew` directly encodes whether the trial is in a rewarded corridor (1) or not (0).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial binary value is broadcast to all frames within the trial.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
```

iii. Simple per-trial constant replicated across time points.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `WallName`, a per-trial string array containing wall/texture names (e.g., `'circle1'`, `'leaf2'`, `'rock1'`).

ii.
```python
wall = np.asarray(beh['WallName'])
# ...
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
# ...
cache.append((tr_spk, inp, str(wall[tr]), ...))
# Later:
out[0, :] = wall_to_id[wall_name]
```

iii. Wall names are collected from all sessions, sorted alphabetically, and mapped to integer indices. Each trial's visual stimulus is the wall name for that trial.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. String wall names are mapped to integer category indices via a global dictionary. The index is broadcast as a constant across all time points of the trial.

ii.
```python
# Stored as -1 initially in cache, then filled:
np.full(tr_spk.shape[1], -1, dtype=np.int64),  # placeholder
# Then:
out[0, :] = wall_to_id[wall_name]  # constant per trial
```

iii. Per-trial categorical variable replicated across time steps.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickTrind` (trial index of each lick event) and `LickTime` (absolute timestamp of each lick event), combined with `Trial_start_time` and the computed `dt_frame`.

ii.
```python
lick_tr = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
```

iii. The AI reconstructs lick events from timestamps rather than using `LickPos` (lick position in corridor). The reference code uses `LickPos` and `LickTrind` for position-based lick rasters.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick timestamps are converted to frame indices within each trial: (1) compute lick time relative to trial start in seconds, (2) divide by `dt_frame` to get frame index, (3) set the corresponding frame in a binary series to 1.

ii.
```python
lick_series = np.zeros(n_frames, dtype=np.int64)
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

iii. The AI converts absolute lick timestamps to relative frame indices within each trial, creating a binary time-varying output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick events are mapped to the same frame indices used for neural data, ensuring temporal alignment. The `lick_series` array is indexed by the same `idx` used for neural data slicing.

ii.
```python
lick = lick_series[idx]  # idx is the same frame indices used for tr_spk
```

iii. Frame-level alignment through shared indexing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_PosCum` (cumulative position in VR units, frame-aligned) and `Corridor_Length` (corridor length in VR units, typically 60).

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
# ...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. The AI uses `ft_PosCum` modded by `Corridor_Length` to get within-corridor position, which is equivalent to `ft_Pos` (verified in the data).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Cumulative position is converted to within-corridor position via modulo, then discretized into 4 equal spatial bins.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. The position is divided into 4 bins spanning [0, corridor_len/4), [corridor_len/4, corridor_len/2), etc.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The corridor is divided into 4 equal-length bins. Position is normalized by corridor length, multiplied by 4, and truncated to integer to get bin index 0-3.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. This creates 4 equal-length spatial bins. The instructions say "4 equal-length, 1-m-long spatial bins", which assumes a 4m corridor. In VR units the corridor is 60 units, so each bin is 15 VR units. The AI's approach divides the corridor into 4 equal parts, which is correct.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position data comes from `ft_PosCum`, which is already frame-aligned. The same frame indices (`idx`) are used for both neural and position data.

ii.
```python
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. Frame-level alignment through shared `ft_` indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is computed from the position data (`ft_PosCum` modded by corridor length) using finite differences. The raw data contains `ft_RunSpeed` but the AI does NOT use it.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. The AI computes speed as the derivative of within-corridor position, not using the precomputed `ft_RunSpeed` field.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is computed via `np.diff` on the within-corridor position (already modded by corridor length), divided by `dt_frame`. Speed is then discretized into 4 quartile-based bins.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
# ...
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75])
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. Speed quantiles are computed per session across all trials, then each frame's speed is binned. The instructions say "4 bins, each corresponding to 25% of the data", which matches the quartile approach.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized using the 25th, 50th, and 75th percentiles of per-session speed distribution, producing 4 bins with approximately equal numbers of frames.

ii.
```python
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0])
speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. Quartile-based binning ensures each bin has ~25% of the data, matching the instructions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from position at each frame, using the same frame indices as neural data.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Frame-level alignment through shared indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Non-finite `ft_trInd` values are marked as invalid (frame index set to -1)
- Mismatched lengths between neural and behavioral arrays are handled by truncating to the minimum length
- Missing `LickTrind`/`LickTime` arrays default to empty
- Non-finite `SoundTime` results in NaN for `time_to_sound`
- Missing `Corridor_Length` defaults to 4.0
- Trials with fewer than 2 frames are skipped
- Sessions with fewer than 2 valid trials are skipped
- Duplicate sessions across behavior files are resolved by preferring the richer entry

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
# ...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
# ...
time_to_sound = ... if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
```

iii. The AI handles missing data defensively with defaults and NaN propagation.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating neural data (`concat_spks`) and iterating over frame indices to build per-trial arrays. The full conversion took ~2177 seconds (~36 minutes) for 76 sessions.

ii.
```python
# In CONVERSION_NOTES.md:
# "Corrected full conversion completed successfully with 76 sessions in 2177.04 s (~36.3 min)"
# Per-session time ~41 seconds, dominated by neural data loading
```

iii. The AI identified neural data loading as the bottleneck but did not implement parallel processing. Each neural file is loaded sequentially.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick reconstruction loop iterates over unique lick trial indices to map timestamps to frame bins. This could potentially be vectorized. The trial-by-trial processing loop could also be partially vectorized for input/output construction.

ii.
```python
# Lick loop:
for tr in np.unique(lick_tr_i):
    idx = trial_frame_idx.get(int(tr))
    # ...
    bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
    lick_series[idx[np.unique(bins)]] = 1
```

iii. The AI noted it "Precomputed trial_frame_idx mapping once per session, reducing repeated frame-index scans" as an optimization. The lick loop could be vectorized but handles relatively few events.

## 12-c. What processing does the code repeat multiple times?

i. The code does a two-pass approach over trials: first collecting trial data and speeds into a cache, then a second pass to compute speed bins and finalize outputs. This is necessary because speed binning requires knowing the full speed distribution. No unnecessary repeated processing is evident.

ii.
```python
# First pass: collect data and speeds
for tr, idx in trial_frame_idx.items():
    # ...
    cache.append((tr_spk, inp, str(wall[tr]), lick, pos_bin, speed))
    all_speeds.append(speed)

# Second pass: compute speed bins
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75])
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. The two-pass approach is justified by the need to compute speed quantiles before binning.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `ft_move` values for all frames but never uses them for filtering (the reference code uses `VRmove = ft_move > 0` to filter non-moving frames). The AI also loads behavioral fields it doesn't use, like `ft_WallID`. The speed is computed from position differences rather than using the precomputed `ft_RunSpeed` field.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
# These are loaded and truncated but never used for filtering
```

iii. Loading unused fields adds minor overhead but is not a major inefficiency.
