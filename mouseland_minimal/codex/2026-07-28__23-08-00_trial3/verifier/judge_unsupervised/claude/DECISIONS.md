# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources: (1) `Imaging_Exp_info.npy` for session metadata, (2) `Beh_*.npy` files for behavioral data, and (3) `*_neural_data.npy` files for neural data plus `*_trans.npz` for retinotopy. The experiment info file is used as the authoritative list of recordings. Behavior files are scanned and deduplicated by session base key. Neural data is loaded per-session during conversion.

ii.
```python
def load_exp_info(root):
    path = os.path.join(root, "data", "beh", "Imaging_Exp_info.npy")
    return np.load(path, allow_pickle=True).item()

def behavior_files(root):
    beh_dir = os.path.join(root, "data", "beh")
    files = []
    for name in sorted(os.listdir(beh_dir)):
        if not name.startswith("Beh_") or not name.endswith(".npy"):
            continue
        full = os.path.join(beh_dir, name)
        try:
            files.append((name, np.load(full, allow_pickle=True).item()))
        except Exception:
            continue
    return files

# Neural data loaded per session:
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. The AI justified using `Imaging_Exp_info.npy` as the authoritative session list, matching the paper's count of 89 recordings in 19 mice. Neural data loading via concatenating `spks` planes matches the reference code's `load_spk()` function.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in each session's metadata from `Imaging_Exp_info.npy`. A unique list is built by iterating through sessions and assigning sequential indices.

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

iii. The AI confirmed 19 unique subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `(mname, datexp, blk)` tuples from `Imaging_Exp_info.npy`. Since the same recording can appear in multiple experimental groups, the AI deduplicates to 89 unique sessions.

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
            sessions.append({
                "mname": item["mname"],
                "datexp": item["datexp"],
                "blk": item["blk"],
                "key": key,
                "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
                "exp_item": item,
            })
    return sessions
```

iii. The AI verified that deduplication yields 89 sessions matching the paper, and that duplicate behavior entries across files are identical for core arrays.

## 1-d. How are the data split into trials?

i. Trial identity comes from the `ft_trInd` field in the behavior data, which assigns each imaging frame to a trial index. The total number of trials per session comes from `beh["ntrials"]`. Frames are grouped by trial index.

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

iii. The AI used `ft_trInd` for trial assignment, consistent with the reference code's approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring at least one valid frame (corridor + running). Trials with zero retained frames after applying `ft_CorrSpc == True` and `ft_move > 0` are dropped. No minimum trial count per session is enforced.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
    # ... process trial
```

iii. The AI's filtering ensures only trials with valid running-in-corridor data are included. No additional quality filters (e.g., minimum number of frames per trial) are applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` key in `*_neural_data.npy` files, which contains deconvolved calcium traces (Suite2p output). Multiple imaging planes are concatenated along the neuron axis.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. The AI confirmed these are deconvolved fluorescence traces consistent with the paper's description of Suite2p processing.

## 2-b. How is the `neural` data processed?

i. The AI selects 128 neurons per session (proportionally across brain regions, ranked by variance), then averages every 3 consecutive retained frames into one time bin. This is a temporal binning approach.

ii.
```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
# ...
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    trial_neural.append(neural_bin)
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The AI justified the 128-neuron cap as necessary for dataset size and decoder tractability. The reference code, however, uses spatial interpolation (`get_interpPos_spk()`) to create position-binned neural data rather than temporal binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to only visual cortex neurons (iarea codes 0-6, 8-9, excluding -1 and 7). Then the top 128 by variance are selected proportionally across V1, mHV, lHV, aHV.

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
    # ... proportional allocation, variance ranking
    var = variance_over_columns(spk, selected_frames)
    # picks top-variance neurons per region
```

iii. The region mapping matches the reference code's `neu_area_ID()`. However, the 128-neuron cap is a significant departure from the reference, which uses all neurons (20,547-89,577 per recording).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). Only frames within the corridor (`ft_CorrSpc == True`) and during running (`ft_move > 0`) are retained, ordered temporally within each trial. Every 3 consecutive retained frames are averaged into one decoder time bin.

ii.
```python
def trial_frame_indices(beh, nfr):
    ft_trind = np.asarray(beh["ft_trInd"][:nfr])
    is_corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    is_move = np.asarray(beh["ft_move"][:nfr], dtype=np.float64) > 0
    # ...
    mask = (ft_trind == tr) & is_corr & is_move
    out.append(np.flatnonzero(mask).astype(np.int64))
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI uses temporal frame ordering from corridor entry, which is consistent. However, the reference code uses spatial (position) interpolation rather than temporal binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is `median_frame_dt_ms * 3` = approximately 944 ms (314.694 ms per frame * 3 frames). Temporal rebinning averages every 3 consecutive retained imaging frames into one decoder bin.

ii.
```python
parser.add_argument("--frames-per-bin", type=int, default=3)
# ...
frame_dt_ms = compute_frame_dt_ms(canonical_lookup, sessions)
# In metadata:
"time_bin_size": float(frame_dt_ms * args.frames_per_bin),
# Per bin:
neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The AI chose 3 frames per bin to create ~1 second time bins. The reference code does not use temporal binning but spatial interpolation into 60 position bins (1 decimeter each).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh["SoundTime"]` (per-trial sound cue time in MATLAB datenum format) and `beh["ft"]` (per-frame timestamps in MATLAB datenum format).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
# ...
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
# Per bin:
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The AI used `SoundTime` rather than `SoundFr` (frame index of sound cue), converting the time difference to seconds via MATLAB datenum conversion.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder time bin, the mean difference between the sound cue time and the frame timestamps within that bin is computed, converted from MATLAB datenum (days) to seconds. This produces a continuous, signed time-to-cue value.

ii.
```python
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The instructions say "Time to sound cue, continuous, time-varying." The AI represents it as continuous seconds. However, the instructions also state "If an input is a time such as onset of some stimulus, represent it as a binary time series," which would suggest a binary representation instead.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-sound-cue is computed for the same frame chunks as the neural data, so it is inherently aligned. Each decoder bin's input uses the same frames as the neural bin.

ii.
```python
for chunk in chunks:
    chunk_ft = ft[chunk]
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    trial_input.append(np.array([
        np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
        # ...
    ]))
```

iii. Alignment is ensured by using the same frame indices for both neural and input computations.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field in session metadata (date string in format `YYYY_MM_DD`).

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

iii. The AI computes calendar days since each mouse's first imaging session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest imaging date is found. The training day for each session is the number of calendar days between the session date and the subject's first date. This is a per-session constant broadcast to all time bins.

ii.
```python
day_value = np.float32(training_days[session_idx])
# Per bin:
trial_input.append(np.array([..., day_value, ...], dtype=np.float32))
```

iii. The instructions say "Day of training, continuous, time-varying." The AI makes it constant within a session (same value for all bins in a session), which makes it "time-varying" only in a trivial sense.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `beh["Trial_start_time"]` (per-trial start time in MATLAB datenum) and `beh["ft"]` (per-frame timestamps).

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
# Per bin:
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. Uses `Trial_start_time` field which corresponds to corridor entry time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each decoder bin, the mean elapsed time since trial start is computed by subtracting the trial start time from each frame's timestamp and converting from MATLAB datenum to seconds.

ii.
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. This produces a continuous, monotonically increasing time variable within each trial, consistent with the instructions.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame chunks are used for both neural and input computation, ensuring automatic alignment.

ii.
```python
for chunk in chunks:
    chunk_ft = ft[chunk]
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    # ...
    np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
```

iii. Alignment is inherent from using the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh["isRew"]`, a per-trial boolean indicating whether the trial's corridor is rewarded.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
# Per bin:
float(is_rew[tr])
```

iii. The AI identified `isRew` as the reward availability indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[trial]` is cast to float (0.0 or 1.0) and broadcast as a constant across all time bins within a trial.

ii.
```python
float(is_rew[tr])
```

iii. This matches the instruction "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh["WallName"]`, which contains the stimulus name for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"])
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The AI used `WallName` rather than `stim_id`, resulting in 15 stimulus categories including swap variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` values across all sessions are collected, sorted, and assigned integer indices (0-14). Each trial gets a constant stimulus index broadcast across all time bins.

ii.
```python
def output_stimulus_names(canonical_lookup, sessions):
    names = set()
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        names.update(map(str, np.unique(beh["WallName"])))
    return sorted(names)

stim_names = output_stimulus_names(canonical_lookup, sessions)
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```

iii. Using `WallName` produces 15 categories. The reference code's `stim_id` mapping has 7 categories (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2). The instruction says "e.g. circle1, leaf2, etc." suggesting either approach could work, but the reference code uses fewer categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh["LickFr"]`, which contains frame indices of lick events.

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

iii. `LickFr` contains neural frame indices of each lick event.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary mask is created at the frame level from `LickFr`. For each decoder time bin (chunk of 3 frames), if any frame in the chunk has a lick, the bin is labeled 1 (licking), otherwise 0.

ii.
```python
int(lick_mask[chunk].any())
```

iii. This produces a binary time-varying output as specified.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The same frame chunks used for neural binning are used for lick detection, ensuring alignment.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    # ...
    int(lick_mask[chunk].any()),
```

iii. Alignment is inherent from shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh["ft_Pos"]`, which gives the VR position for each imaging frame.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
# Per bin:
mean_pos = float(ft_pos[chunk].mean())
```

iii. `ft_Pos` contains the animal's position in the virtual corridor in decimeters.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each decoder bin, the mean position across the chunk's frames is computed, then discretized into 4 equal-length bins.

ii.
```python
mean_pos = float(ft_pos[chunk].mean())
position_to_bin(mean_pos)
```

iii. The position is averaged within each time bin before discretization.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided by 10 (decimeters to meters), floored, and clipped to 0-3, producing 4 bins: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. The instructions say "4 equal-length, 1-m-long spatial bins." The 4m corridor with 10 decimeters per meter gives bins at [0,10), [10,20), [20,30), [30,40) decimeters, matching the 4 equal 1m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Same frame chunks are used for position averaging and neural data, ensuring alignment.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    mean_pos = float(ft_pos[chunk].mean())
```

iii. Alignment is inherent from shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh["ft_RunSpeed"]`, which gives the running speed for each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
# Per bin:
mean_speed = float(ft_speed[chunk].mean())
```

iii. `ft_RunSpeed` contains the animal's running speed per neural frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each decoder bin, the mean speed across the chunk's frames is computed. Global quartile thresholds are precomputed across all retained decoder bins in the full dataset.

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
```

iii. Global quartiles ensure consistent binning across sessions, matching the instruction "4 bins, each corresponding to 25% of the data."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed values are mapped to bins 0-3 using `np.searchsorted` with the three global quartile thresholds.

ii.
```python
def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```

iii. This produces exactly 4 bins with approximately 25% of data in each, as confirmed by the verification output showing equal distribution.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Same frame chunks are used for speed averaging and neural data, ensuring alignment.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    mean_speed = float(ft_speed[chunk].mean())
```

iii. Alignment is inherent from shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several handling strategies are used: (1) `LickFr` values are filtered for finite values before creating the lick mask, handling NaN entries. (2) Frame count is clipped to `min(spk.shape[1], len(beh["ft"]))` to handle mismatches between neural and behavioral data lengths. (3) Trials with zero valid frames are skipped. (4) Duplicate behavior entries across files are canonicalized with mismatch checking. (5) Neurons outside recognized brain regions (iarea == -1 or 7) are excluded.

ii.
```python
nfr = min(spk.shape[1], len(beh["ft"]))
# ...
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
# ...
if not chunks:
    continue
```

iii. The AI documented these handling strategies in CONVERSION_NOTES.md and verified 0 duplicate mismatches occurred.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading neural data files (each containing tens of thousands of neurons with tens of thousands of frames), (2) Computing variance over retained frames for neuron selection, and (3) Precomputing global speed thresholds by iterating over all sessions and trials.

ii.
```python
# Loading large neural arrays:
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)

# Variance computation (optimized with block processing):
def variance_over_columns(matrix, cols, block_cols=1024, max_var_frames=2048):
    # ...
```

iii. The AI added optimization for variance computation by capping at 2048 representative frames.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main conversion loop iterates over trials and then over chunks within each trial. The per-chunk operations (mean neural, mean speed, mean position, lick detection) could potentially be vectorized across chunks within a trial using strided array operations.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    for chunk in chunks:
        neural_bin = spk_sel[:, chunk].mean(axis=1)
        mean_speed = float(ft_speed[chunk].mean())
        mean_pos = float(ft_pos[chunk].mean())
```

iii. The inner loop over chunks and the outer loop over trials could be vectorized, but the variable number of frames per trial makes this non-trivial.

## 12-c. What processing does the code repeat multiple times?

i. The speed threshold computation (`compute_speed_thresholds`) iterates over all sessions calling `trial_frame_indices` for each, and then the main conversion loop calls `trial_frame_indices` again for each session. This duplicates the frame-filtering computation.

ii.
```python
# First pass in compute_speed_thresholds:
for indices in trial_frame_indices(beh, nfr):
    for chunk in chunk_indices(indices, frames_per_bin):
        speed_values.append(float(ft_speed[chunk].mean()))

# Second pass in convert_session:
frames_by_trial = trial_frame_indices(beh, nfr)
```

iii. The trial frame indices are computed twice per session -- once for speed thresholds and once for conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The variance computation for neuron selection loads the full neuron-by-frame matrix just to select 128 neurons, then discards the rest. (2) The `build_behavior_lookup` function computes provenance tracking and mismatch detection that is stored in metadata but not used by the decoder. (3) The `choose_sample_session_indices` function has elaborate selection logic for the sample dataset. (4) The full speed values array is computed but only the quartile thresholds are used.

ii.
```python
# Full speed array computed but only quartiles used:
speed_values, speed_thresholds = compute_speed_thresholds(...)
# speed_values is not used further

# Provenance stored in metadata:
"behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```

iii. These are reasonable for documentation and debugging but represent unnecessary computation for the core task.
