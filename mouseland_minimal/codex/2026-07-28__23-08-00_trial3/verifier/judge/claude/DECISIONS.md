# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `Imaging_Exp_info.npy` as the master index of all recordings, grouped by experiment type. It deduplicates by `(mname, datexp, blk)` to get unique sessions. It then loads all `Beh_*.npy` files into a canonical lookup keyed by session base. For each session, it loads the spike file `spk/<base>_neural_data.npy` and the retinotopy file `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
exp_info = np.load(os.path.join(root, "data", "beh", "Imaging_Exp_info.npy"), allow_pickle=True).item()
# ...
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. The AI loads all behavior files upfront into a canonical lookup (with duplicate mismatch checking), then processes each session one at a time loading spikes and retinotopy per session.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in each session entry. Subjects are collected in order of first appearance, and `subject_idx` maps each session to its subject index.

ii.
```python
for session in sessions:
    if session["mname"] not in subject_to_idx:
        subject_to_idx[session["mname"]] = len(subjects)
        subjects.append(session["mname"])
subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The AI confirmed 19 unique mice matching the paper.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` tuple from `Imaging_Exp_info.npy`. Duplicates across experiment types are removed. This yields 89 sessions matching the paper.

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

iii. The AI verified that deduplication yields 89 recordings matching the paper's reported count.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` to identify which frames belong to which trial. The AI retains only frames where `ft_CorrSpc == True` AND `ft_move > 0` (i.e., running corridor frames only). Frames are then grouped into chunks of 3 ("temporal rebinning") to form decoder time bins. Trials with no retained frames are dropped.

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

iii. The AI justified the `ft_move > 0` filter by citing the paper's restriction to running time points and referencing the notebook code `VRmove = beh['ft_move'][:nfr]>0`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring at least some retained frames after applying the corridor and running filters. No other quality filter is applied. Sessions with zero neurons or fewer than 2 trials are skipped (though the code in convert_session would raise an error for zero frames).

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
```

iii. The AI's CONVERSION_NOTES states trials with no running corridor frames are dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains deconvolved calcium traces per imaging plane, concatenated across planes. The visual area of each neuron comes from `iarea` in the retinotopy file.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. The AI confirmed these are deconvolved Suite2p outputs.

## 2-b. How is the `neural` data processed?

i. The AI selects a subset of 128 neurons per session (proportionally sampled across V1/mHV/lHV/aHV by variance), then averages every 3 retained imaging frames into one decoder time bin. The result is stored as float32.

ii.
```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
# ...
neural_bin = spk_sel[:, chunk].mean(axis=1)
# ...
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The AI justified the 128-neuron limit by noting that all-neuron conversion would produce a dataset too large for the pickle format and the decoder. The 3-frame binning reduces temporal resolution while keeping ~7-8 bins per trial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons in visual cortex areas (V1, mHV, lHV, aHV) are eligible. From those, 128 are selected per session by proportional allocation across regions, ranked by variance over retained frames.

ii.
```python
def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0  # V1
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    out[(iarea == 5) | (iarea == 6)] = 2  # lHV
    out[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return out

def select_neurons(spk, region_idx_full, selected_frames, neurons_per_session):
    # proportional allocation, variance ranking
    ...
```

iii. The AI justified neuron subsampling as necessary for tractability, with variance-based ranking to retain the most informative neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). The retained frames are those from the corridor with running, starting from the first such frame of each trial. Variable-length trials are not padded to a fixed length.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    # ...
    for chunk in chunks:
        neural_bin = spk_sel[:, chunk].mean(axis=1)
        trial_neural.append(neural_bin)
    neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The AI sets `off_start: 0.0` and `off_end: None` in metadata, acknowledging variable trial lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes, temporal rebinning is applied. Every 3 consecutive retained imaging frames are averaged into one decoder time bin. The time bin size is `frame_dt_ms * 3 ≈ 944 ms`. The raw imaging rate is ~3.17 Hz (~315 ms per frame).

ii.
```python
"time_bin_size": float(frame_dt_ms * args.frames_per_bin),
# where frames_per_bin defaults to 3
```
```python
def chunk_indices(indices, chunk_size):
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
```

iii. The AI chose 3-frame binning to reduce dataset size while keeping ~7-8 bins per trial.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the wall-clock timestamp of the sound cue per trial) and `ft` (the timestamp of each imaging frame).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
# ...
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The AI used `SoundTime` (a continuous timestamp) rather than `SoundFr` (a frame index), arguing that timestamp subtraction is more precise and doesn't depend on frame rate assumptions.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder time bin (chunk of 3 frames), the mean of `(SoundTime[trial] - ft[chunk_frames]) * SECONDS_PER_DAY` is computed, giving the average time-to-cue in seconds for that bin. The sign convention is positive before the cue and negative after.

ii.
```python
trial_input.append(
    np.array([
        np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
        day_value,
        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
        float(is_rew[tr]),
    ], dtype=np.float32)
)
```

iii. The AI converts from MATLAB datenum (days) to seconds and averages within each temporal bin.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame timestamps (`ft[chunk]`) used to construct each neural time bin, so alignment is inherent.

ii.
```python
chunk_ft = ft[chunk]
# same chunk used for neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. All data streams use the same chunk of frame indices per bin.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field of each session, which encodes the calendar date as `YYYY_MM_DD`.

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

iii. The AI uses the actual calendar date parsed from the session identifier.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the number of calendar days since the mouse's first imaging session. For each session, `training_day = (session_date - first_session_date).days`. This is broadcast across all bins of all trials in that session.

ii.
```python
first_date = min(date for _, date in entries)
for idx, date in entries:
    offsets[idx] = float((date - first_date).days)
```

iii. The AI uses calendar days rather than session ordinal, arguing this captures true elapsed time between sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the wall-clock timestamp of corridor entry per trial) and `ft` (the timestamp of each imaging frame).

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
# ...
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The AI used `Trial_start_time` rather than `StartFr` for the same precision reasons as with `SoundTime`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each decoder time bin, the mean of `(ft[chunk_frames] - Trial_start_time[trial]) * SECONDS_PER_DAY` is computed, giving the average time since trial start in seconds.

ii.
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. Converts from MATLAB datenum to seconds and averages within each temporal bin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame timestamps used for each neural bin (same `chunk` indices).

ii.
```python
chunk_ft = ft[chunk]
```

iii. Same alignment mechanism as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a boolean per-trial field indicating whether the trial is in a rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
# ...
float(is_rew[tr])
```

iii. Directly taken from the data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and used as a constant per trial, replicated across all time bins.

ii.
```python
float(is_rew[tr])
```

iii. No processing needed beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"])
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The AI uses `WallName` directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does NOT group textures into broad categories. Instead, it uses all 15 individual texture names (e.g., circle1, circle2, leaf1, leaf1_swap1, etc.) as separate categories. Each trial's `WallName` is mapped to its index in the sorted vocabulary of 15 names.

ii.
```python
stim_names = output_stimulus_names(canonical_lookup, sessions)  # 15 unique names
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
# ...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The AI argued that collapsing categories would lose information that the paper itself distinguishes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of each lick in the session.

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

iii. The AI converts lick frame indices to a boolean mask per frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A frame-level boolean mask is created from `LickFr`. For each decoder time bin (chunk of 3 frames), licking is 1 if any frame in the chunk had a lick, 0 otherwise.

ii.
```python
int(lick_mask[chunk].any())
```

iii. Binary licking per decoder bin using `any()` within the chunk.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick mask is indexed by the same frame indices used for the neural data in each chunk, so alignment is inherent.

ii.
```python
int(lick_mask[chunk].any())
# chunk is the same as used for neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. Same alignment mechanism as all variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
# ...
mean_pos = float(ft_pos[chunk].mean())
```

iii. Directly from the per-frame position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each decoder time bin, the mean position across frames in the chunk is computed, then divided by 10 (converting decimeters to meters) and floored to get a bin index 0-3.

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
# ...
mean_pos = float(ft_pos[chunk].mean())
position_to_bin(mean_pos)
```

iii. The 4 bins correspond to 0-1m, 1-2m, 2-3m, 3-4m as specified in the instructions.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 equal-length 1m bins by `floor(mean_position_dm / 10)` clipped to [0, 3].

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. Matches the instruction's specification of 4 equal-length 1m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is computed from the same frame chunk used for the neural bin.

ii.
```python
mean_pos = float(ft_pos[chunk].mean())
```

iii. Same alignment as all variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
# ...
mean_speed = float(ft_speed[chunk].mean())
```

iii. Directly from the per-frame speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each decoder time bin, the mean speed across the chunk's frames is computed. This mean speed is then binned into 4 quartile categories using global speed thresholds computed from all retained decoder bins across all sessions.

ii.
```python
def compute_speed_thresholds(canonical_lookup, sessions, frames_per_bin):
    speed_values = []
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        nfr = len(beh["ft"])
        ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
        for indices in trial_frame_indices(beh, nfr):
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
    speed_values = np.asarray(speed_values, dtype=np.float32)
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return speed_values, q.astype(np.float32)

def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```

iii. Global quartile thresholds ensure consistency across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global quantile thresholds (25th, 50th, 75th percentiles of all retained bin speeds) divide the data into 4 bins. `searchsorted` assigns each bin's mean speed to a category.

ii.
```python
q = np.quantile(speed_values, [0.25, 0.5, 0.75])
# ...
speed_to_bin(mean_speed, speed_thresholds)
```

iii. Global thresholds computed across all sessions and trials.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the same frame chunk used for the neural bin.

ii.
```python
mean_speed = float(ft_speed[chunk].mean())
```

iii. Same alignment as all variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior arrays are clipped to `min(spk.shape[1], len(beh["ft"]))` frames to handle mismatches between neural and behavior lengths; (2) `LickFr` values are filtered for `isfinite`, non-negative, and within frame range; (3) trials with no retained frames are dropped; (4) sessions that raise exceptions are skipped with an error message.

ii.
```python
nfr = min(spk.shape[1], len(beh["ft"]))
# ...
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
# ...
try:
    sess_neural, sess_input, ... = convert_session(...)
except Exception as error:
    print(...)
    continue
```

iii. The AI added explicit NaN/bounds checking for lick frames and graceful error handling per session.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files, which are very large (4-9 GB each). The AI also computes global speed thresholds upfront (requiring a pass through all behavior data) and performs variance-based neuron selection per session.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. I/O dominates, similar to the reference solution.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_frame_indices` function loops over all trials in a session, applying the mask per trial. This could be vectorized using `np.unique` or `groupby` on `ft_trInd`. The `compute_speed_thresholds` function also loops over all sessions and trials to collect speed values.

ii.
```python
for tr in range(ntrials):
    mask = (ft_trind == tr) & is_corr & is_move
    out.append(np.flatnonzero(mask).astype(np.int64))
```

iii. The per-trial loop scans the full frame array once per trial.

## 12-c. What processing does the code repeat multiple times?

i. The `trial_frame_indices` function is called twice: once in `compute_speed_thresholds` (for all sessions) and once in `convert_session` (for each session). The behavior data is also loaded twice: once in `build_behavior_lookup` and once implicitly via `canonical_lookup` during conversion. The speed threshold computation pre-processes all sessions before the main conversion loop.

ii.
```python
# In compute_speed_thresholds:
for indices in trial_frame_indices(beh, nfr):
# In convert_session:
frames_by_trial = trial_frame_indices(beh, nfr)
```

iii. The duplicate trial frame computation is a noticeable inefficiency.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `variance_over_columns` for neuron selection, which is used only for ranking neurons and is discarded afterward. The `build_behavior_lookup` function performs extensive duplicate mismatch checking across all behavior files that is only used for a diagnostic print. The `build_sample_dataset` and `choose_sample_session_indices` functions implement complex sample selection logic.

ii.
```python
var = variance_over_columns(spk, selected_frames)
# ...
duplicate_mismatches = []  # only used for printing
```

iii. The variance computation and duplicate checking are preprocessing steps not needed for the final output.
