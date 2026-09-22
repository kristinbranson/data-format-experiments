# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as a master index of all recordings grouped by experiment type. For each experiment type, it loads the corresponding `Beh_<exp_type>.npy` behavior file. It builds a "canonical" behavior dictionary keyed by base session key (mouse_date_blk), deduplicating sessions that appear under multiple experiment types. It also lists spike files from the `spk/` directory. For each session, the spike file `<session_key>_neural_data.npy` and retinotopy file `<mouse>_<date>_trans.npz` are loaded during conversion.

ii.
```python
exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()
# ...
for exp_type, records in exp_info.items():
    behavior_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
    beh = np.load(behavior_path, allow_pickle=True).item()
```
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
```

iii. The agent stated: "Use every imaging session listed in `Imaging_Exp_info.npy`, including `swap1`/`swap2` sessions that need the `stimtype` suffix in the behavior key." The agent also cross-references spike files from the `spk/` directory to determine which sessions to process.

## 1-b. How are the data split into subjects (mice)?

i. The mouse name is parsed from the session key using a regex `^(?P<mouse>.+?)_(?P<date>\d{4}_\d{2}_\d{2})_(?P<blk>\d+)$`. Subjects are the sorted unique mouse names across all session keys.

ii.
```python
subjects = sorted({parse_base_key(key)[0] for key in session_keys})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.array([subject_to_idx[parse_base_key(key)[0]] for key in session_keys], dtype=np.int64)
```

iii. The agent parses mouse names from session keys and builds a sorted unique list. This is consistent with the reference approach.

## 1-c. How are the data split into sessions?

i. A session is identified by a base key: `mouse_date_blk`. The AI deduplicates recordings that appear under multiple experiment types in `Imaging_Exp_info.npy` by keeping only the first occurrence per base key. It also cross-references against spike files in the `spk/` directory, and also enumerates sessions from the spk directory using `get_spk_session_keys()`.

ii.
```python
def load_canonical_behavior_sessions():
    # ...
    for exp_type, records in exp_info.items():
        for record in records:
            full_key = full_key_from_record(record)
            base_key = f"{record['mname']}_{record['datexp']}_{record['blk']}"
            session = beh[full_key]
            if base_key not in canonical:
                canonical[base_key] = { ... }
                continue
            # verify duplicates match
```

iii. The agent stated it deduplicates recordings, keeping one canonical behavior entry per base session key.

## 1-d. How are the data split into trials?

i. Trials are identified by `ft_trInd` (frame-to-trial index). For each trial, the AI selects frames where `ft_trInd == trial` AND the frame passes the `retained_mask` filter (`ft_CorrSpc & (ft_move > 0)`). This means only frames inside the corridor where the mouse was running are kept, which differs from the reference that keeps all corridor frames regardless of movement.

ii.
```python
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
    if len(frame_idx) == 0:
        continue
```

iii. The agent stated: "Match the paper's curation by restricting each trial to running frames (`ft_move > 0`) inside the corridor (`ft_CorrSpc`), which removes stationary reward-collection periods the authors excluded."

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has zero retained frames after applying the `ft_CorrSpc & (ft_move > 0)` mask. Sessions with fewer than 2 valid trials are also dropped. There is no outlier-based trial length filtering.

ii.
```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
if len(frame_idx) == 0:
    continue
# ...
if len(neural_trials) < 2:
    return None
```

iii. The agent did not implement a trial length percentile cutoff like the reference (99th percentile). Instead it relies solely on the running + corridor mask to implicitly shorten or remove problematic trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike files (`spk/<session_key>_neural_data.npy`), which is a list of per-plane neuron-by-frame matrices. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
```
```python
retino = np.load(retino_path, allow_pickle=True)
return build_region_index(retino["iarea"])
```

iii. The agent uses the same raw data source as the reference.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed (no dF/F, no normalization). Each trial's neural data is the selected neurons' traces at the retained frame indices, cast to float16.

ii.
```python
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```

iii. The agent stated it uses Suite2p deconvolved fluorescence traces directly, matching the paper's statement that "all our analyses were based on deconvolved fluorescence traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies TWO filters beyond the visual area restriction: (1) Only neurons in V1, mHV, lHV, or aHV are kept. (2) A cap of 128 neurons per region per session is applied, selecting neurons by highest variance during corridor running, with a preference for "responsive" neurons (corridor mean > gray mean). This is a significant additional filter not in the reference.

ii.
```python
selected_global = select_visual_neurons(
    spk_planes=spk_planes,
    region_idx_full=region_idx_full,
    corridor_mask=retained_mask,
    gray_mask=gray_running_mask,
    max_per_region=max_neurons_per_region,
)
```
```python
MAX_NEURONS_PER_REGION = 128
```

iii. The agent stated: "For tractability it keeps retinotopically assigned visual-cortex neurons only and caps each session at 128 neurons per region (V1, mHV, lHV, aHV), so 512 neurons/session." The reference keeps ALL neurons in the four visual areas with no cap.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). However, due to the `ft_move > 0` filter, the first frame of a trial may not be the actual corridor entry frame — it is the first frame where the mouse was both in the corridor AND running. This introduces a variable lag relative to actual corridor entry.

ii.
```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
# retained_mask = ft_corr & (ft_move > 0)
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```

iii. The agent acknowledged: "I'm quantifying how far the first kept sample tends to lag corridor entry after the running-only filter." But still proceeded with the running filter.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The imaging frame rate (~3.17 Hz) defines the temporal resolution. The AI computes `TIME_BIN_MS` as `1000.0 * 24.0 * 3600.0 * 0.31469352543354034` which equals ~315 ms, matching the reference.

ii.
```python
TIME_BIN_MS = 1000.0 * 24.0 * 3600.0 * 0.31469352543354034
```

iii. The agent uses the imaging frame period as the time bin, same as the reference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue per trial) and `ft` (the timestamp of each imaging frame). This differs from the reference which uses `SoundFr` (the frame number of the sound cue).

ii.
```python
sound_time = np.asarray(behavior["SoundTime"], dtype=np.float64)
ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The agent uses `SoundTime` timestamps directly rather than `SoundFr` frame numbers with interpolation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundTime[trial] - ft_time[frame_idx]) * 24 * 3600` in seconds. This gives the time remaining until the sound cue (positive before, negative after), matching the "time TO sound cue" semantics.

ii.
```python
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The conversion from MATLAB datenum to seconds is correct (multiply by 86400).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed at the same retained frame indices used for the neural data, ensuring alignment.

ii.
```python
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. All streams use the same `frame_idx` array, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date portion of the session key. The AI parses dates and computes the number of calendar days since each mouse's first imaging session.

ii.
```python
def compute_training_day_by_session(session_keys):
    # ...
    first_day[mouse] = min(date_from_base_key(key) for key in keys)
    training_day[session_key] = float((date_from_base_key(session_key) - first_day[mouse]).days)
```

iii. The agent stated: "Calendar days since the first imaging session for that mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes actual calendar days between the session date and the mouse's first session date. This differs from the reference which counts recording sessions (0, 1, 2, ...) rather than calendar days (which could be 0, 7, 14, ...).

ii.
```python
training_day[session_key] = float((date_from_base_key(session_key) - first_day[mouse]).days)
```

iii. The agent explicitly stated this is "Calendar days since the first imaging session for that mouse" in the metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (per-trial start timestamp) and `ft` (per-frame timestamps). The reference uses `StartFr` (the corridor entry frame number) instead.

ii.
```python
trial_start_time = np.asarray(behavior["Trial_start_time"], dtype=np.float64)
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The agent uses `Trial_start_time` rather than `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is `(ft_time[frame_idx] - Trial_start_time[trial]) * 24 * 3600` in seconds. Positive values indicate time after trial start.

ii.
```python
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
```

iii. Simple time difference in seconds using MATLAB datenum conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at the same `frame_idx` array used for neural data.

ii.
```python
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
```

iii. Same frame indices ensure alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
is_rew = np.asarray(behavior["isRew"], dtype=bool)
reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to a float (0.0 or 1.0) and broadcast across all frames of the trial.

ii.
```python
reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)
```

iii. Straightforward conversion matching the reference approach.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which gives the wall texture name for each trial.

ii.
```python
wall_names = np.asarray(behavior["WallName"])
stim_category = STIM_CATEGORY_TO_IDX[canonical_stimulus_category(wall_names[trial])]
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are stripped of swap suffixes and trailing digits via regex to get the base category (circle, leaf, rock, wood), then mapped to a category index. The value is broadcast across all frames of the trial.

ii.
```python
def canonical_stimulus_category(wall_name):
    category = re.sub(r"_swap[12]$", "", str(wall_name))
    category = re.sub(r"\d+$", "", category)
    return category
```

iii. The agent uses regex instead of a hardcoded lookup table, but achieves the same four-category mapping.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame indices), `LickTrind` (trial index of each lick), and `LickPos` (position of each lick). The reference only uses `LickFr`.

ii.
```python
lick_trial_index = np.asarray(behavior["LickTrind"], dtype=np.float64)
lick_frame_idx = np.asarray(behavior["LickFr"], dtype=np.float64)
lick_pos = np.asarray(behavior["LickPos"], dtype=np.float64)
```

iii. The AI uses additional variables (`LickTrind` and `LickPos`) to filter licks by trial and by position within the texture.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI filters licks to those belonging to the current trial (`LickTrind == trial`) and within the texture corridor (`LickPos >= 0 and < texture_length`). It then uses a nearest-frame matching approach to assign licks to retained (running) frames, only marking a frame as "lick" if a lick occurred within 0.5 frames of a retained frame. This is substantially more complex than the reference, which simply marks any frame containing a lick as 1.

ii.
```python
def nearest_retained_licks(retained_frame_idx, lick_frame_idx, lick_pos, texture_length):
    # ... filters licks by position validity
    # ... uses searchsorted to find nearest retained frame
    lick_binary[np.unique(nearest[nearest_dist <= 0.5])] = 1
    return lick_binary
```

iii. The agent's approach is complex because it needs to handle the fact that not all frames are retained (due to `ft_move > 0` filter), so licks may fall on non-retained frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary vector is computed over the retained frame indices and has the same length as the neural data for that trial.

ii.
```python
licking = nearest_retained_licks(
    retained_frame_idx=frame_idx,
    lick_frame_idx=lick_frame_idx[trial_lick_mask],
    lick_pos=lick_pos[trial_lick_mask],
    texture_length=texture_length,
)
```

iii. Aligned via the retained frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame.

ii.
```python
ft_pos = np.asarray(behavior["ft_Pos"][:n_frames], dtype=np.float32)
position_out = make_position_bins(ft_pos[frame_idx], texture_length)
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to `[0, texture_length)` and then binned into 4 equal bins using `np.digitize` with edges at `[0, 10, 20, 30, 40]` (for texture_length=40 dm). This gives 4 bins of 10 dm = 1 m each.

ii.
```python
def make_position_bins(position, texture_length):
    pos = np.clip(np.asarray(position, dtype=np.float32), 0.0, float(texture_length) - 1e-6)
    bin_edges = np.linspace(0.0, float(texture_length), 5)
    return np.clip(np.digitize(pos, bin_edges[1:-1], right=False), 0, 3).astype(np.int8)
```

iii. Uses `np.digitize` instead of integer division, but produces the same 4 x 1m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1m bins: 0-1m, 1-2m, 2-3m, 3-4m. Same as reference.

ii.
```python
bin_edges = np.linspace(0.0, float(texture_length), 5)  # [0, 10, 20, 30, 40]
```

iii. Matches the instruction to discretize into 4 equal-length 1m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled at the same retained frame indices as the neural data.

ii.
```python
position_out = make_position_bins(ft_pos[frame_idx], texture_length)
```

iii. Aligned via shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(behavior["ft_RunSpeed"][:n_frames], dtype=np.float32)
speed_out = make_speed_bins(ft_speed[frame_idx], speed_edges)
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed edges from the 25th, 50th, and 75th percentiles of running speeds across ALL sessions, using only running corridor frames (`ft_CorrSpc & ft_move > 0`). It then uses `np.digitize` to bin per-trial speeds. This differs from the reference which uses per-session rank-based quartiles.

ii.
```python
def compute_speed_edges(session_keys, canonical_behavior):
    all_speeds = []
    for session_key in session_keys:
        sess = canonical_behavior[session_key]["behavior"]
        mask = sess["ft_CorrSpc"] & (sess["ft_move"] > 0)
        if np.any(mask):
            all_speeds.append(np.asarray(sess["ft_RunSpeed"][mask], dtype=np.float32))
    speed_values = np.concatenate(all_speeds)
    q25, q50, q75 = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return np.array([q25, q50, q75], dtype=np.float32)
```

iii. The agent uses global quantile-based edges applied uniformly, while the reference uses per-session rank-based splits.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins using global percentile edges (25th, 50th, 75th percentile of all running speeds). Each frame's speed is digitized into one of 4 bins.

ii.
```python
def make_speed_bins(speed, speed_edges):
    return np.clip(np.digitize(np.asarray(speed, dtype=np.float32), speed_edges, right=False), 0, 3).astype(np.int8)
```

iii. The global edges mean individual sessions won't have exactly 25% in each bin, unlike the reference's per-session approach.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sampled at the same retained frame indices as the neural data.

ii.
```python
speed_out = make_speed_bins(ft_speed[frame_idx], speed_edges)
```

iii. Aligned via shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior arrays are truncated to the number of neural frames (`n_frames = spk_planes[0].shape[1]`). Trials with zero retained frames are skipped. Sessions with fewer than 2 valid trials are skipped. Licks are validated for finite values and position within the texture corridor before being assigned to frames.

ii.
```python
n_frames = spk_planes[0].shape[1]
ft_trind = np.asarray(behavior["ft_trInd"][:n_frames])
# ...
if len(frame_idx) == 0:
    continue
```
```python
valid = np.isfinite(lick_frame_idx) & np.isfinite(lick_pos) & (lick_pos >= 0.0) & (lick_pos < texture_length)
```

iii. The AI handles edge cases with validation checks on lick data and frame counts.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files, which are large numpy arrays. The AI loads each spike file once per session, and the full run processes all 89 sessions.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
```

iii. Same bottleneck as the reference — I/O of neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `select_visual_neurons` function loops over planes and regions with nested Python loops. The `nearest_retained_licks` function is called per-trial. The trial loop itself processes one trial at a time.

ii.
```python
for plane_idx, plane in enumerate(spk_planes):
    # ...
    for region in range(len(VISUAL_CORTEX_REGIONS)):
        # ...
```

iii. The neuron selection logic involves multiple nested loops that could potentially be vectorized.

## 12-c. What processing does the code repeat multiple times?

i. The `compute_speed_edges` function iterates over all sessions' behavior data to compute global speed quantiles, then the main loop iterates over all sessions again for conversion. The behavior files are loaded twice: once in `load_canonical_behavior_sessions()` and kept in memory.

ii.
```python
def compute_speed_edges(session_keys, canonical_behavior):
    all_speeds = []
    for session_key in session_keys:
        # reads behavior again
```

iii. The speed edge computation iterates over behavior separately from the main conversion loop.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `select_visual_neurons` function computes corridor responsiveness (corridor_mean > gray_mean) and variance scores for neuron selection, which is additional processing not required by the task. The `behavior_sessions_match` function validates duplicate sessions, which is a correctness check but not needed for the conversion itself.

ii.
```python
def select_visual_neurons(spk_planes, region_idx_full, corridor_mask, gray_mask, max_per_region):
    # computes corridor_mean, corridor_var, gray_mean for responsiveness scoring
```

iii. The neuron curation logic (responsiveness + variance ranking + cap) adds significant computation not present in the reference.
