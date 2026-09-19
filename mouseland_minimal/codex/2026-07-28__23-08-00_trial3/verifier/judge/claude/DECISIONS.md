# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from three subdirectories under `data/`: `beh/` for behavior (one file per experiment type, prefixed `Beh_`), `spk/` for deconvolved calcium traces (one file per session), and `retinotopy/` for visual area assignments. `Imaging_Exp_info.npy` is loaded first as the authoritative session list. All behavior files are loaded into a canonical lookup dictionary keyed by session base ID. For each session, the spike file and retinotopy file are loaded during conversion.

ii. Loading the master index and behavior:
```python
exp_info = np.load(path, allow_pickle=True).item()
# ...
canonical[base] = (filename, raw_key, beh)
```

Loading spikes and retinotopy per session:
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. The agent verified that duplicate behavior entries across experiment types had matching core arrays (comparing `ntrials`, `WallName`, `ft_trInd`, etc.), and canonicalized them to avoid double-processing. The agent noted: "ordinary duplicate group assignments are literally the same behavior object reused across figures."

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in each entry of `Imaging_Exp_info.npy`. Subjects are collected in the order they appear, and `subject_idx` maps each session to its subject.

ii.
```python
subjects = []
subject_to_idx = {}
for session in sessions:
    if session["mname"] not in subject_to_idx:
        subject_to_idx[session["mname"]] = len(subjects)
        subjects.append(session["mname"])
subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The agent used the `mname` field directly from the experiment info. No additional logic was needed.

## 1-c. How are the data split into sessions?

i. A session is identified by the tuple `(mname, datexp, blk)`. Duplicate entries across experiment types are deduplicated, yielding 89 unique sessions.

ii.
```python
def collect_unique_sessions(exp_info):
    sessions = []
    seen = set()
    for entries in exp_info.values():
        for item in entries:
            key = (item["mname"], item["datexp"], item["blk"])
            if key in seen:
                continue
            seen.add(key)
            sessions.append({...})
    return sessions
```

iii. The agent verified that recordings appearing under multiple experiment types are the same data, and kept only one canonical entry per session.

## 1-d. How are the data split into trials?

i. Trials are identified by `ft_trInd` and filtered to frames satisfying both `ft_CorrSpc == True` (inside the textured corridor) AND `ft_move > 0` (mouse is running). Only frames meeting both conditions are retained. Trials with no surviving frames are dropped.

ii.
```python
def trial_frame_indices(beh, nfr):
    ft_trind = np.asarray(beh["ft_trInd"][:nfr])
    is_corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    is_move = np.asarray(beh["ft_move"][:nfr], dtype=np.float64) > 0
    ntrials = int(beh["ntrials"])
    out = []
    for tr in range(ntrials):
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
    return out
```

iii. The agent stated: "restrict each trial's retained frames to `ft_move>0` to match the paper's 'running-only' analyses." This adds a running-only filter not present in the reference code.

## 1-e. How are trials filtered based on quality controls?

i. The only filtering is that trials with zero retained frames (after the `ft_CorrSpc` and `ft_move > 0` filters) are implicitly skipped, since `chunk_indices` returns an empty list and the trial loop continues. There is no explicit trial length or quality threshold.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
```

iii. The agent did not apply a trial length percentile filter as the reference does. The `ft_move > 0` filter already removes stationary frames, which partially addresses the concern about long stationary trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the per-session neural data files (one array per imaging plane, concatenated), and `iarea` from the retinotopy files for region assignment.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. Same source data as the reference.

## 2-b. How is the `neural` data processed?

i. Neurons are subselected: only 128 neurons per session are retained, chosen proportionally across V1/mHV/lHV/aHV by variance over retained frames. The selected neurons' traces are then averaged within bins of 3 consecutive retained frames.

ii.
```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
# ...
neural_bin = spk_sel[:, chunk].mean(axis=1)
# ...
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The agent explained neuron subselection as needed "for decoder tractability" due to dataset size constraints. The agent reasoned that "a literal frame-by-frame, all-neuron, all-session conversion would be far too large to pickle and far too large for the provided decoder to train."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) neurons must be in one of the four visual cortex areas (V1, mHV, lHV, aHV), and (2) only the top 128 neurons by variance are retained per session, proportionally allocated across regions.

ii.
```python
def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0   # V1
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    out[(iarea == 5) | (iarea == 6)] = 2  # lHV
    out[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return out

selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
```

iii. The area mapping matches the reference. The additional variance-based subselection is the agent's own decision for tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Each trial's neural data starts at the first retained frame of that trial (frames where `ft_CorrSpc==True` and `ft_move>0`). The retained frames are then grouped into bins of 3 consecutive frames.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    # ...
    for chunk in chunks:
        neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The alignment event matches the instructions (corridor entry). However, frames where the mouse is not running are excluded, so the first bin may not exactly correspond to the moment of corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes, temporal rebinning is applied. Every 3 consecutive retained imaging frames are averaged into one decoder time bin. The reported `time_bin_size` is computed as the median inter-frame interval times 3 (approximately 945 ms).

ii.
```python
frame_dt_ms = compute_frame_dt_ms(canonical_lookup, sessions)
# ...
"time_bin_size": float(frame_dt_ms * args.frames_per_bin),
```

```python
def chunk_indices(indices, chunk_size):
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
```

iii. The agent chose 3-frame binning to reduce dataset size. The reference keeps native resolution (1 frame per bin, ~315 ms).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of each imaging frame), both in MATLAB datenum format (days).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
# ...
chunk_ft = ft[chunk]
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The agent uses `SoundTime` directly rather than `SoundFr` (the frame number of the sound). Both encode the same event but in different units. The reference uses `SoundFr` interpolated onto the frame time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference `SoundTime[trial] - ft[frame]` is computed in day units, then multiplied by `SECONDS_PER_DAY` (86400) to get seconds. Within each 3-frame bin, the values are averaged.

ii.
```python
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. Positive before the sound, negative after, consistent with the "time TO sound cue" semantics.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time values are computed from the same frame indices (`chunk`) used for the neural data of each bin.

ii.
```python
chunk_ft = ft[chunk]
# same chunk used for neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. Alignment is maintained by using the same frame indices for all data streams.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field of each session, which contains the date string in `YYYY_MM_DD` format.

ii.
```python
def compute_training_days(sessions):
    by_subject = defaultdict(list)
    for idx, session in enumerate(sessions):
        by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
    offsets = {}
    for subject, entries in by_subject.items():
        first_date = min(date for _, date in entries)
        for idx, date in entries:
            offsets[idx] = float((date - first_date).days)
    return offsets
```

iii. The agent used the date string from `datexp` to compute calendar day offsets.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, sessions are grouped by date. The training day is the number of calendar days since the subject's first imaging session. This is a per-trial scalar broadcast across all time bins.

ii.
```python
first_date = min(date for _, date in entries)
offsets[idx] = float((date - first_date).days)
# ...
day_value = np.float32(training_days[session_idx])
# used as: day_value in each bin's input array
```

iii. The agent stated: "Calendar days since the first imaging session for that mouse." The reference counts session ordinals (0, 1, 2, ...) rather than calendar days. These differ when there are gaps between recording days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the timestamp of the trial start) and `ft` (the timestamps of imaging frames), both in MATLAB datenum format.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
# ...
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The agent uses `Trial_start_time` rather than `StartFr` (the frame number of corridor entry). The reference uses `StartFr` interpolated onto frame times.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The difference `ft[frame] - Trial_start_time[trial]` is computed, converted to seconds, and averaged within each 3-frame bin.

ii.
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. Positive after trial start, matching "time SINCE trial start" semantics.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Same frame indices (`chunk`) are used for both neural and timing data.

ii.
```python
chunk_ft = ft[chunk]
```

iii. Alignment maintained via shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial flag indicating rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
# ...
float(is_rew[tr])
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and included as a per-trial scalar in each bin's input vector.

ii.
```python
float(is_rew[tr])
```

iii. No additional processing needed. Matches the reference approach.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the name of the wall texture for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"])
# ...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The raw `WallName` strings (e.g., `circle1`, `leaf1_swap2`, `wood5`) are used directly as 15 individual categories, rather than being grouped into 4 base texture categories (circle, leaf, rock, wood) as the reference does.

ii.
```python
stim_names = output_stimulus_names(canonical_lookup, sessions)  # 15 unique names, sorted
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
# ...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The agent noted: "Per-trial visual category taken directly from beh['WallName'] for each trial." The reference groups the 15 wall names into 4 texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers of individual lick events.

ii.
```python
def build_lick_frame_mask(beh, nfr):
    lick_mask = np.zeros(nfr, dtype=bool)
    lick_fr = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
    lick_mask[lick_idx] = True
    return lick_mask
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A boolean mask is created over all frames, marking frames where a lick occurred. Within each 3-frame decoder bin, licking is 1 if any frame in the bin had a lick, 0 otherwise.

ii.
```python
int(lick_mask[chunk].any())
```

iii. The reference similarly creates a binary per-frame lick flag; the AI additionally aggregates across the 3-frame bin using `any()`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick mask is indexed by the same frame indices used for the neural data bins.

ii.
```python
lick_mask[chunk]  # same chunk used for neural_bin
```

iii. Alignment via shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
# ...
mean_pos = float(ft_pos[chunk].mean())
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values within each 3-frame bin are averaged, then the mean is divided by 10 (converting decimeters to meters) and floored to get a bin index 0-3.

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))

mean_pos = float(ft_pos[chunk].mean())
position_to_bin(mean_pos)
```

iii. The reference bins per-frame (`ft_Pos // 10` then clips), whereas the AI first averages position across frames in a bin then discretizes.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m. Position in decimeters is divided by 10, floored, and clipped to [0, 3].

ii.
```python
int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. Same binning logic as the reference (4 bins of 1 meter each).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is computed from the same frame indices as the neural data.

ii.
```python
mean_pos = float(ft_pos[chunk].mean())
```

iii. Alignment via shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
# ...
mean_speed = float(ft_speed[chunk].mean())
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is averaged within each 3-frame bin, then categorized into 4 bins using global quartile thresholds computed over all retained decoder bins across all sessions.

ii.
```python
def compute_speed_thresholds(canonical_lookup, sessions, frames_per_bin):
    speed_values = []
    for session in sessions:
        # ...
        for indices in trial_frame_indices(beh, nfr):
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
    speed_values = np.asarray(speed_values, dtype=np.float32)
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return speed_values, q.astype(np.float32)

def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```

iii. The reference uses per-session rank-based quartiles, where each session's bins are independently split so that exactly 25% of frames fall in each bin. The AI uses global fixed thresholds computed across all sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global quartile thresholds (25th, 50th, 75th percentiles) are computed over all bins across all sessions. `np.searchsorted` assigns each bin's mean speed to one of 4 categories.

ii.
```python
q = np.quantile(speed_values, [0.25, 0.5, 0.75])
# ...
int(np.searchsorted(thresholds, value, side="right"))
```

iii. The reference uses per-session rank-based quartiles. The global approach means individual sessions may have unequal category distributions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the same frame indices as neural data.

ii.
```python
mean_speed = float(ft_speed[chunk].mean())
```

iii. Alignment via shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The number of frames is clipped to `min(spk.shape[1], len(beh["ft"]))` to handle behavior running past imaging. Lick frame indices are filtered for finite values and valid range. Trials with no retained frames are skipped.

ii.
```python
nfr = min(spk.shape[1], len(beh["ft"]))
# ...
lick_fr = np.asarray(beh["LickFr"], dtype=np.float64)
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```

iii. The agent handles edge cases carefully (NaN lick frames, out-of-range indices). The reference similarly clips to the number of imaged frames.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the large spike files (multi-GB each) and computing per-neuron variance for neuron selection.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
# ...
var = variance_over_columns(spk, selected_frames)
```

iii. The agent noted: "The full run is compute-bound" and optimized the variance computation to use a fixed representative subset of frames (`max_var_frames=2048`).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_frame_indices` function loops over all trials, computing a mask for each trial separately. This could be replaced with a single group-by operation. The `chunk_indices` and per-bin processing loop could also be partially vectorized.

ii.
```python
for tr in range(ntrials):
    mask = (ft_trind == tr) & is_corr & is_move
    out.append(np.flatnonzero(mask).astype(np.int64))
```

iii. The I/O cost of loading spike files dominates, so vectorizing the trial loop would provide marginal speedup.

## 12-c. What processing does the code repeat multiple times?

i. The `trial_frame_indices` function is called twice for speed threshold computation (in `compute_speed_thresholds`) and again during session conversion (in `convert_session`), recomputing the same trial frame masks. The behavior lookup is also built by loading all behavior files upfront.

ii.
```python
# In compute_speed_thresholds:
for indices in trial_frame_indices(beh, nfr):
# In convert_session:
frames_by_trial = trial_frame_indices(beh, nfr)
```

iii. The repeated trial frame computation is relatively cheap compared to spike file I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes variance over retained frames for all neurons, but only the top 128 per session are kept. The `build_behavior_lookup` function loads all behavior files upfront and performs duplicate mismatch checking, which is a data validation step not strictly needed for conversion. The code also builds a sample dataset (`sample_data.pkl`), which is extra processing beyond the required output.

ii.
```python
var = variance_over_columns(spk, selected_frames)
# variance computed for all neurons, most discarded

sample_data = build_sample_dataset(...)
```

iii. The variance computation over all neurons is necessary for ranking, even if most results are discarded. The sample dataset generation is an additional feature beyond what was requested.
