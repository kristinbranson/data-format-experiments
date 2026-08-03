# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories under `data/`: `beh/` for behavior, `spk/` for deconvolved neural traces, and `retinotopy/` for brain region assignments. It first reads `Imaging_Exp_info.npy` as the master session index, then iterates through experiment types to build a unique session list. For each session, it loads neural data (`load_spk`), retinotopy (`load_retino`), and behavior (`Beh_<exp_type>.npy`). Behavior files are loaded per-session rather than being shared across sessions in the same file.

ii.
```python
info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
```

iii. The AI identified the same data sources and master index as the reference. The behavior file is re-loaded for each session rather than being cached across sessions sharing the same behavior file, but functionally loads the same data.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from `entry['mname']` in the session index. Subjects are collected as they appear during processing. `subject_idx` maps each session to its subject.

ii.
```python
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The AI builds the subject list incrementally as sessions are processed, rather than pre-computing a sorted unique set. The subjects list may not be sorted, depending on processing order.

## 1-c. How are the data split into sessions?

i. A session is one unique combination of mouse name, date, and block (`mname_datexp_blk`). The AI deduplicates by keeping only the first occurrence of each session_id across experiment types, yielding 89 sessions.

ii.
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. The AI correctly identifies sessions by the same triple (mname, datexp, blk) and deduplicates, matching the reference approach.

## 1-d. How are the data split into trials?

i. The AI iterates over `range(ntrials)` for each session. For each trial, it selects frames where `ft_trInd == trial_idx` AND `ft_move > 0` (VR-moving frames). Trials with fewer than 2 valid frames are skipped. Trials are variable length (no fixed frame count or padding).

ii.
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    if len(valid_frame_indices) < 2:
        continue
```

iii. The AI chose to use `ft_move > 0` as the frame filter based on the paper's statement that "only considered timepoints during running for analysis". This differs from the reference which uses `ft_CorrSpc` (corridor space) to select frames within the textured portion of the corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 valid frames (after applying `ft_move > 0` filter) are dropped. Sessions with fewer than 2 valid trials are also dropped.

ii.
```python
if len(valid_frame_indices) < 2:
    continue
# ...
if valid_count < 2:
    return None
```

iii. No additional quality filtering beyond the minimum frame/trial count. The reference similarly applies no quality filter beyond dropping empty trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `<session_id>_neural_data.npy`, which contains deconvolved calcium traces per imaging plane, concatenated across planes. Brain region assignments come from `iarea` in the retinotopy files.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed (no additional dF/F or deconvolution). For each trial, the neural data is extracted at the valid frame indices (`ft_move > 0` frames for that trial) and stored as float32. Trials are variable length with no padding.

ii.
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The reference pads/truncates all trials to a fixed length of 32 frames and stores as float16. The AI uses variable-length trials and float32, which uses more memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by brain region: neurons with `iarea == -1` or `iarea == 7` are excluded. The remaining neurons are kept regardless of their specific visual area assignment.

ii.
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
```

iii. The reference filters neurons by keeping only those in specific visual areas: V1 (iarea=8), mHV (iarea=0,1,2,9), lHV (iarea=5,6), aHV (iarea=3,4). The AI's filter `(iarea != -1) & (iarea != 7)` is broader — it keeps neurons in any area except -1 and 7, potentially including areas not in the reference's four visual regions. The CONVERSION_NOTES state the filter "matches reference code" but the actual implementation differs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by selecting frames where `ft_trInd == trial_idx` and `ft_move > 0`. The first valid frame corresponds to the first VR-moving frame of the trial. There is no fixed-length windowing; trials are variable length.

ii.
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The reference aligns to corridor entry using `ft_CorrSpc` and pads/truncates to N_FRAMES=32. The AI aligns to VR-moving frames within each trial, which may include frames from before corridor entry or after corridor exit, and produces variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The imaging frame rate of 3.17 Hz is the temporal resolution, giving ~315.5 ms per bin. The time_bin_size in metadata is set to `1000.0 / 3.17`.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. Matches the reference approach — no rebinning, using the native imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame index of the sound cue per trial) and `ft` (the timestamp of each frame in MATLAB datenum format).

ii.
```python
s_fr = sound_fr[trial_idx]
# ...
sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. Same source variables as the reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound frame is fractional, so the AI interpolates linearly between adjacent frame times to get the sound event time. Then for each frame in the trial, it computes `frame_time - sound_time` in seconds. This gives negative values before the cue and positive after. NaN sound frames are replaced with zeros.

ii.
```python
s_fr_int = int(np.floor(s_fr))
s_fr_frac = s_fr - s_fr_int
sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The reference computes `cue[trial] - time` (positive before cue, negative after). The AI computes `frame_time - sound_time` (negative before cue, positive after). The sign convention is reversed. The variable name "time_to_sound_cue" suggests time remaining until the cue, which should be positive before and negative after (matching the reference).

## 3-c. How is `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices used for the neural data (the `valid_frame_indices` from `ft_move > 0` filtering).

ii.
```python
ft_trial = ft[valid_frame_indices]
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. Aligned by using the same frame indices as the neural data, consistent in approach.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field of each session entry, which contains the date string (e.g., "2022_07_12"). The AI parses this into a datetime object.

ii.
```python
def compute_day_of_training(sessions):
    # ...
    dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))) for idx, s in msessions]
    dates.sort(key=lambda x: x[1])
    first_date = dates[0][1]
    for idx, date in dates:
        day_of_training[idx] = (date - first_date).days
```

iii. The AI parses actual calendar dates from the session identifiers.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the number of calendar days since the first recording session for each mouse. For example, if a mouse's first session was on day 0, a session 5 calendar days later would have day_of_training=5. This is broadcast across all time bins in a trial.

ii.
```python
first_date = dates[0][1]
for idx, date in dates:
    day_of_training[idx] = (date - first_date).days
```

iii. The reference counts session ordinals (0, 1, 2, ...) — the first session is 0, second is 1, etc., regardless of how many calendar days apart they are. The AI uses actual calendar day differences, which could produce values like 0, 5, 12 instead of 0, 1, 2. This is a different interpretation.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` (frame timestamps in MATLAB datenum format). The first valid frame's timestamp serves as the trial start reference.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The reference uses `StartFr` (the corridor entry frame) interpolated onto the time axis as the reference point. The AI uses the first VR-moving frame's timestamp instead.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes the elapsed time in seconds from the first valid (VR-moving) frame of the trial. The first frame always has time_since_start = 0.

ii.
```python
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The reference computes `time - start[trial]` where `start` is interpolated from `StartFr`, which represents corridor entry. The AI's approach always starts at 0, whereas the reference's value starts near 0 but not exactly 0 (since StartFr is between frames). There may also be large time gaps when the mouse stops running (non-contiguous frames due to `ft_move` filter).

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same frame indices used for the neural data.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Aligned by using the same frame indices as the neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. Same source variable as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` value is cast to 1.0 (rewarded) or 0.0 (not rewarded) and broadcast across all time bins in a trial.

ii.
```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. Same approach as the reference. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the walls of the corridor for each trial.

ii.
```python
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each unique `WallName` to a unique integer index. All 15 unique stimulus names (e.g., circle1, circle2, leaf1, leaf1_swap1, etc.) are treated as separate categories. The value is per-trial, broadcast across all time bins.

ii.
```python
all_stim_names = get_all_stim_names(all_sessions)  # 15 unique names
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
# ...
output_trial[0] = stim_idx
```

iii. The reference groups the 15 names into 4 base texture categories (circle, leaf, rock, wood) via a lookup table. The AI keeps all 15 as separate categories, which is inconsistent with the instructions that say "Visual stimulus category. e.g. circle1, leaf2, etc." and with the reference paper which treats them as 4 texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame indices of each lick event in the session.

ii.
```python
lick_binary = build_lick_per_frame(n_frames, beh.get('LickFr', np.array([])))
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices are rounded to the nearest integer, then a binary array is created where 1 indicates at least one lick at that frame. The per-trial licking is extracted at the valid frame indices.

ii.
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary
```

iii. The reference truncates lick frames with `.astype(int)` (floor), while the AI rounds with `np.round()`. This is a minor difference. The AI also handles the case where `LickFr` might not exist with a default empty array.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is indexed at the same `valid_frame_indices` used for the neural data.

ii.
```python
lick_trial = lick_binary[valid_frame_indices]
```

iii. Aligned by using the same frame indices as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. Same source variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position is divided by 15 decimeters (1.5 m) and floored, then clipped to range [0, 3], giving 4 bins over the full 6 m corridor (0-1.5m, 1.5-3m, 3-4.5m, 4.5-6m).

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The reference divides by 10 decimeters (1 m), giving 4 bins of 1 m over the 4 m textured corridor (0-1m, 1-2m, 2-3m, 3-4m). The instructions say "4 equal-length, 1-m-long spatial bins", and the corridor texture is 4 m, so the reference's 1 m bins are correct. The AI chose 1.5 m bins over the full 6 m corridor (including grey space), which does not match the instructions.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided by 15 dm and floored, giving 4 bins: 0-1.5m, 1.5-3m, 3-4.5m, 4.5-6m.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The reference uses `ft_Pos // 10` clipped to [0, 3], giving 4 bins of 1 m each over the 4 m textured corridor. The AI uses 15 dm bins over 6 m including grey space.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted at the same `valid_frame_indices` used for the neural data.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. Aligned by using the same frame indices as the neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. Same source variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartiles (25th, 50th, 75th percentiles) from all sessions using `ft_move > 0` frames, then uses `np.digitize` to bin each frame's speed. Only a random subsample of 5000 speeds per session is used for quartile computation.

ii.
```python
def compute_global_speed_quartiles(sessions):
    # ...
    vr_move = beh['ft_move'] > 0
    speeds = beh['ft_RunSpeed'][vr_move]
    if len(speeds) > 5000:
        speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
    all_speeds.append(speeds)
    # ...
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. The reference computes per-session rank-based quartiles using all kept frames of that session, ensuring exactly 25% of frames in each bin per session. The AI uses global percentile-based quartiles, which means the distribution across bins won't be exactly 25% in each session. Also, the reference only uses frames that belong to valid trials (via `kept`), while the AI uses all `ft_move > 0` frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with globally-computed quartile boundaries [25th, 50th, 75th percentiles].

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The reference uses rank-based assignment (`rank * 4 // N`) per session to ensure equal bin counts. The AI uses value-based thresholds from global percentiles, which may not produce equal bin counts per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted at the same `valid_frame_indices` used for the neural data.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. Aligned by using the same frame indices as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior arrays are truncated to the neural frame count (`n_frames_neural`). Lick frames outside the neural range are dropped. Missing `LickFr` defaults to an empty array. NaN `SoundFr` values produce zero time-to-sound arrays. Trials with fewer than 2 valid frames are skipped.

ii.
```python
ft = beh['ft'][:n_frames_neural]
# ...
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
# ...
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
```

iii. The AI handles edge cases similarly to the reference (truncating behavior to neural frame count, dropping out-of-range licks). The NaN handling for SoundFr is an additional safeguard not present in the reference.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (multi-GB each, 89 total). The AI's initial implementation also loaded neural data for speed quartile computation, which was later optimized. The full conversion took 73.1 minutes.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```

iii. Same bottleneck as the reference — I/O for large neural data files dominates.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over all trials sequentially, extracting frames and computing inputs/outputs one trial at a time. This could potentially be vectorized using grouped operations on `ft_trInd`.

ii.
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
    # ...
```

iii. The reference has the same pattern. The per-trial loop is not the bottleneck compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The AI loads behavior files multiple times: once for `get_all_stim_names`, once for `compute_global_speed_quartiles`, and once per session during processing. The reference loads each behavior file only once by grouping sessions by behavior file.

ii.
```python
# In get_all_stim_names:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), ...)
# In compute_global_speed_quartiles:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), ...)
# In process_session:
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), ...)
```

iii. Behavior files are loaded 3 times each across the pipeline.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes global speed quartiles by loading behavior data from all 89 sessions with subsampling, which is a pre-processing step not needed by the reference approach (which computes quartiles per-session). The AI also creates temporary cache files for each session and then re-loads them for assembly, adding unnecessary I/O.

ii.
```python
speed_quartiles = compute_global_speed_quartiles(all_sessions)
# ...
temp_fn = f'cache/session_{i:03d}.pkl'
with open(temp_fn, 'wb') as f:
    pickle.dump({...}, f, protocol=4)
```

iii. The global speed quartile computation and temp file caching are architectural choices that add overhead not present in the reference.
