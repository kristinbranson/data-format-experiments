# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `Imaging_Exp_info.npy` to enumerate all 89 recordings. Behavior data is loaded from all `Beh_*.npy` files in the `data/beh/` directory, each containing a dictionary of session-keyed behavior bundles. Neural data is loaded per-session from `data/spk/{base}_neural_data.npy` files (concatenating across planes from the `spks` key). Retinotopy data is loaded from `data/retinotopy/{mouse}_{date}_trans.npz`.

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

# Neural loading per session:
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. The AI justified this by noting that `Imaging_Exp_info.npy` is the "authoritative list of imaging recordings" matching the paper's reported 89 recordings in 19 mice. It also documented that many `Beh_*.npy` files reuse the same recording under figure-specific group names and built a canonicalization step to handle duplicates.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in `Imaging_Exp_info.npy` entries. A unique subject list is built by iterating over sessions in order and collecting unique mouse names.

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

iii. The AI verified that the unique subject count (19) matches the paper's reported number.

## 1-c. How are the data split into sessions?

i. Sessions are uniquely identified by the tuple `(mname, datexp, blk)` from `Imaging_Exp_info.npy`. The AI deduplicates entries with the same key, yielding 89 unique sessions matching the paper.

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

iii. Justified by matching the paper's reported 89 recordings.

## 1-d. How are the data split into trials?

i. Trials are identified using the `ft_trInd` field from the behavior data, which assigns each imaging frame to a trial index. The total number of trials per session comes from `beh["ntrials"]`. The AI iterates over trial indices 0 through ntrials-1, collecting frame indices for each trial.

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

iii. The AI noted this matches the reference code's use of `ft_trInd` for trial assignment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they have at least one valid frame after applying the corridor (`ft_CorrSpc`) and running (`ft_move > 0`) filters. Trials with zero retained frames after these filters are dropped. Additionally, after chunking frames into groups of `frames_per_bin` (3), trials with zero complete chunks are dropped.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
    # ... process trial ...
```

iii. From CONVERSION_NOTES.md: "I retained only frames satisfying both: ft_CorrSpc == True and ft_move > 0. This matches the paper's stated restriction to running time points and excludes gray-space frames."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` field in per-session neural data files (`{base}_neural_data.npy`). These are deconvolved calcium traces from Suite2p, concatenated across imaging planes.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. The paper states: "All our analyses were based on deconvolved fluorescence traces." The reference code's `load_spk` function does the same concatenation.

## 2-b. How is the `neural` data processed?

i. The neural data is processed by: (1) selecting a subset of 128 neurons per session based on variance ranking within visual cortex regions, proportionally allocated across V1/mHV/lHV/aHV; (2) temporal binning by averaging every 3 retained frames into one decoder time bin.

ii.
```python
# Neuron selection:
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)

# Temporal binning per trial:
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The AI justified the 128-neuron limit as necessary for decoder tractability, noting the original datasets have 20,547-89,577 neurons per recording which would be too large to pickle and train on.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to only include those from retinotopically-assigned visual cortex areas (V1, mHV, lHV, aHV). Neurons with `iarea` values not mapping to these four regions (e.g., iarea == 7 or iarea == -1) are excluded. Within each region, neurons are ranked by variance over retained frames and the top neurons are selected.

ii.
```python
def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0  # V1
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    out[(iarea == 5) | (iarea == 6)] = 2  # lHV
    out[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return out

# Neurons with region_idx == -1 are excluded from selection
```

iii. This region mapping directly matches the reference code's `neu_area_ID` function in `utils.py`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). For each trial, the retained frames begin at the first frame where `ft_trInd == trial_index` AND `ft_CorrSpc == True` AND `ft_move > 0`. This effectively aligns to corridor entry since `ft_CorrSpc` marks corridor frames.

ii.
```python
mask = (ft_trind == tr) & is_corr & is_move
# Frames are taken in order within each trial, starting from corridor entry
```

iii. From CONVERSION_NOTES.md: "Temporal alignment event: corridor entry / trial start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is approximately `frame_dt_ms * 3` (3 frames per bin). The median imaging frame interval is computed globally from the `ft` timestamps and multiplied by the `frames_per_bin` parameter (default 3). Yes, temporal rebinning is applied by averaging every 3 consecutive retained frames.

ii.
```python
def compute_frame_dt_ms(canonical_lookup, sessions):
    dts = []
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        ft = np.asarray(beh["ft"], dtype=np.float64)
        if ft.size < 2:
            continue
        diffs = np.diff(ft) * SECONDS_PER_DAY * 1000.0
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            dts.append(np.median(diffs))
    return float(np.median(dts))

# In metadata:
"time_bin_size": float(frame_dt_ms * args.frames_per_bin)
```

iii. The AI noted the rebinning is needed to make the decoder tractable while preserving temporal structure.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh["SoundTime"]` (the absolute time of the sound cue for each trial) and `beh["ft"]` (the absolute timestamp of each imaging frame).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
# ...
chunk_ft = ft[chunk]
# Per-bin computation:
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. Justified by examining the behavior data structure which contains `SoundTime` as a per-trial timestamp and `ft` as per-frame timestamps, both in MATLAB datenum format.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder time bin (chunk of 3 frames), the time to sound cue is computed as the mean of `(SoundTime[trial] - ft[frame]) * SECONDS_PER_DAY` across the frames in that bin. This gives the time remaining until the sound cue in seconds (positive before the cue, negative after).

ii.
```python
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The AI chose to express this as time *to* the cue (sound_time minus frame_time), which makes it positive when before the cue and negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed per decoder time bin using the same frame indices as the neural data. Each bin's value is the mean across the frames in that bin, ensuring perfect temporal alignment.

ii.
```python
for chunk in chunks:
    chunk_ft = ft[chunk]
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    # ...
    np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
```

iii. Alignment is inherent since both neural and input data are computed from the same frame chunks.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field in the session metadata (from `Imaging_Exp_info.npy`), which contains the recording date as a string (e.g., "2022_06_17").

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

iii. Justified as "calendar days since the first imaging session for that mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest recording date is found. The day of training for each session is computed as the number of calendar days between the session's date and that subject's first recording date. This value is constant within a session and replicated across all time bins.

ii.
```python
day_value = np.float32(training_days[session_idx])
# Used per bin as:
day_value,  # constant across bins within a session
```

iii. From CONVERSION_NOTES.md: "training_day: calendar days since the subject's first imaging session."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did not include "Environment type" as a separate input. The decoder instructions specified "Reward availability" (1 if in rewarded corridor, 0 if not) rather than "Environment type." The AI derived the closest equivalent, `reward_available`, from `beh["isRew"]`.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
# Used as:
float(is_rew[tr])
```

iii. The AI followed the decoder task specification which lists "Reward availability" as the 4th input, not "Environment type."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. As noted above, the AI used `isRew` directly as a binary per-trial flag. The value is constant within each trial (1.0 for rewarded corridor, 0.0 for unrewarded), with no additional processing.

ii.
```python
float(is_rew[tr])
```

iii. This is a direct read from the behavior data with no transformation beyond casting to float.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `beh["Trial_start_time"]` (per-trial timestamp in MATLAB datenum format) and `beh["ft"]` (per-frame timestamps).

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
# Per bin:
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The AI identified `Trial_start_time` as the appropriate variable for trial onset time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each decoder time bin, the mean of `(ft[frame] - Trial_start_time[trial]) * SECONDS_PER_DAY` is computed across the frames in that bin, yielding elapsed seconds since trial start.

ii.
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. Straightforward conversion from MATLAB datenum difference to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned identically to all other variables: computed from the same frame chunks used for neural binning, ensuring per-bin temporal correspondence.

ii. Same loop structure as neural and other inputs -- all computed within the same `for chunk in chunks` loop.

iii. Alignment is inherent in the shared frame-chunking approach.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh["isRew"]`, a per-trial boolean array indicating whether the corridor is rewarded.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
# ...
float(is_rew[tr])
```

iii. Directly matches the decoder task specification: "1 if in rewarded corridor, 0 if not."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing beyond casting the boolean to float (0.0 or 1.0). The value is constant per trial and replicated across all time bins within that trial.

ii.
```python
float(is_rew[tr])  # same value for all bins in the trial
```

iii. A simple binary flag with no transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh["WallName"]`, which contains the stimulus name (e.g., "circle1", "leaf2") for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"])
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. From CONVERSION_NOTES.md: "Per-trial visual category taken directly from beh['WallName'] for each trial."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A global vocabulary of all stimulus names across all sessions is built and sorted alphabetically. Each stimulus name is mapped to an integer index. This index is constant within a trial and replicated across all time bins.

ii.
```python
def output_stimulus_names(canonical_lookup, sessions):
    names = set()
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        names.update(map(str, np.unique(beh["WallName"])))
    return sorted(names)

stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```

iii. The sorted global vocabulary ensures consistent encoding across sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh["LickFr"]`, which contains frame indices at which licks were detected.

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

iii. `LickFr` provides the frame-level lick timing used in the reference code (e.g., in `spk_2_firstLick` and `spk_2_cue` functions).

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary frame-level mask is built from `LickFr`. For each decoder time bin (chunk of 3 frames), licking is 1 if any frame in the chunk has a lick, 0 otherwise.

ii.
```python
int(lick_mask[chunk].any())
```

iii. This produces the required binary time-varying output: 0 = not licking, 1 = licking.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick mask is indexed by the same frame indices as the neural data chunks, ensuring alignment. Each decoder bin's lick value comes from the same frames as the neural activity for that bin.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    # ...
    int(lick_mask[chunk].any()),
```

iii. Alignment is inherent in the shared frame-chunk indexing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh["ft_Pos"]`, which contains the per-frame position in the corridor (in decimeters, 0-40 range for a 4m corridor).

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
mean_pos = float(ft_pos[chunk].mean())
```

iii. The reference code uses `ft_Pos` extensively for position-based analyses.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each decoder bin, the mean position across the frames in that chunk is computed.

ii.
```python
mean_pos = float(ft_pos[chunk].mean())
```

iii. Simple averaging within the time bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The mean position is discretized into 4 equal-length 1-meter bins: [0-1m, 1-2m, 2-3m, 3-4m]. Since `ft_Pos` is in decimeters (0-40 range), the bin is computed as `floor(value / 10)` clipped to [0, 3].

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. The 4m corridor divided into 4 equal 1m bins matches the decoder task specification exactly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is computed from the same frame chunks as neural data, using the mean `ft_Pos` over the chunk frames.

ii. Same loop structure -- computed within the `for chunk in chunks` loop alongside neural data.

iii. Inherent alignment through shared frame indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh["ft_RunSpeed"]`, which contains the per-frame running speed.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
mean_speed = float(ft_speed[chunk].mean())
```

iii. The reference code uses `ft_RunSpeed` in its analyses.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each decoder bin, the mean speed across the chunk's frames is computed. Global quartile thresholds are pre-computed from all retained decoder bins across all sessions.

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

iii. From CONVERSION_NOTES.md: "quartile bin of mean ft_RunSpeed, with thresholds computed globally over all retained decoder bins in the full dataset."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized into 4 quartile bins using the global 25th, 50th, and 75th percentile thresholds. `np.searchsorted` with `side="right"` assigns each speed value to a bin 0-3.

ii.
```python
def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```

iii. The decoder task specifies "4 bins, each corresponding to 25% of the data," which matches quartile-based binning.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the same frame chunks as neural data, using the mean `ft_RunSpeed` over the chunk frames.

ii. Same loop structure -- all variables computed within the shared `for chunk in chunks` loop.

iii. Inherent alignment through shared frame indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) `LickFr` entries that are NaN or out of bounds are filtered out before building the lick mask; (2) trials with zero valid frames after filtering are silently skipped; (3) duplicate behavior entries across Beh files are canonicalized, keeping the first occurrence and checking for mismatches; (4) neural frame count and behavior frame count mismatches are resolved by taking the minimum: `nfr = min(spk.shape[1], len(beh["ft"]))`.

ii.
```python
# Lick frame handling:
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]

# Frame count mismatch:
nfr = min(spk.shape[1], len(beh["ft"]))

# Duplicate behavior canonicalization:
if base not in canonical:
    canonical[base] = (filename, raw_key, beh)
    continue
```

iii. From CONVERSION_NOTES.md: "Duplicate behavior entries produced 0 core-array mismatches during canonicalization."

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) neuron variance computation over retained frames for neuron selection (`variance_over_columns`), which the AI optimized with a frame cap; (2) per-session neural data loading and concatenation; (3) the global speed threshold computation which iterates over all sessions and all trial frames.

ii.
```python
def variance_over_columns(matrix, cols, block_cols=1024, max_var_frames=2048):
    if cols.size > max_var_frames:
        keep = np.linspace(0, cols.size - 1, max_var_frames, dtype=np.int64)
        cols = cols[keep]
    # ... blocked variance computation ...
```

iii. The trajectory shows the AI identified neuron ranking as a hotspot and added a frame cap (`max_var_frames=2048`) to speed it up.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial, per-bin loop in `convert_session` iterates over each trial's chunks sequentially, appending to lists. This inner loop could potentially be vectorized by building matrices of bin indices and using advanced indexing. The speed threshold computation similarly loops over all sessions and trials.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    # ...
    for chunk in chunks:
        # per-bin computation
```

iii. The AI did not explicitly discuss vectorization opportunities; the approach prioritized correctness and readability.

## 12-c. What processing does the code repeat multiple times?

i. The `trial_frame_indices` function is called twice for each session: once during global speed threshold computation (`compute_speed_thresholds`) and once during per-session conversion (`convert_session`). The frame filtering logic (`ft_CorrSpc` and `ft_move > 0`) is thus applied redundantly.

ii.
```python
# Called in compute_speed_thresholds:
for indices in trial_frame_indices(beh, nfr):

# Called again in convert_session:
frames_by_trial = trial_frame_indices(beh, nfr)
```

iii. Not explicitly discussed by the AI. This is a minor inefficiency.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores detailed `behavior_entry_provenance` metadata mapping each session to all its source behavior files, which is not used by the decoder. The `sample_data.pkl` selection logic performs extensive heuristic scoring of sessions that is only used once. The code also computes and stores `original_neurons` count in session_info metadata, which is informational only.

ii.
```python
"behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```

iii. These serve documentation and debugging purposes rather than downstream analysis needs.
