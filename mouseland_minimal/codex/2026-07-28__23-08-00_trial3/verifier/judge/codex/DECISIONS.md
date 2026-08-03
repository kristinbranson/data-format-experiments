# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats `data/beh/Imaging_Exp_info.npy` as the master index, deduplicates recordings by `(mname, datexp, blk)`, loads every `Beh_*.npy` file into a canonical behavior lookup keyed by the session base id, and then loads spike and retinotopy files per session during conversion.

ii. 
```python
def load_exp_info(root):
    path = os.path.join(root, "data", "beh", "Imaging_Exp_info.npy")
    return np.load(path, allow_pickle=True).item()
```
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
            sessions.append(
                {
                    "mname": item["mname"],
                    "datexp": item["datexp"],
                    "blk": item["blk"],
                    "key": key,
                    "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
                    "exp_item": item,
                }
            )
```
```python
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
```
```python
    spk_path = os.path.join(root, "data", "spk", f"{session['base']}_neural_data.npy")
    ret_path = os.path.join(root, "data", "retinotopy", f"{session['mname']}_{session['datexp']}_trans.npz")

    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
    ret = np.load(ret_path, allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says `Imaging_Exp_info.npy` is the authoritative session list, that deduplicating by mouse/date/block gives 89 recordings, and that duplicated behavior entries are canonicalized by session base key after checking for mismatches.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mname` from the experiment index. The output `subjects` list is built in first-seen session order, and `subject_idx` assigns each session to that subject index.

ii. 
```python
            sessions.append(
                {
                    "mname": item["mname"],
                    "datexp": item["datexp"],
                    "blk": item["blk"],
                    "key": key,
                    "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
                    "exp_item": item,
                }
            )
```
```python
    subjects = []
    subject_to_idx = {}
    for session in sessions:
        if session["mname"] not in subject_to_idx:
            subject_to_idx[session["mname"]] = len(subjects)
            subjects.append(session["mname"])
    subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The notes say the paper reports 19 mice and that the deduplicated session list preserves those 19 unique subjects.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` tuple. The AI constructs a `base` string of the form `mouse_date_block` and removes duplicates with a `seen` set.

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
            sessions.append(
                {
                    "mname": item["mname"],
                    "datexp": item["datexp"],
                    "blk": item["blk"],
                    "key": key,
                    "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
                    "exp_item": item,
                }
            )
```

iii. The notes explicitly justify this as deduplicating figure-specific behavior bundles down to the paper’s 89 unique recordings.

## 1-d. How are the data split into trials?

i. Trials are first identified by `ft_trInd`, but the AI keeps only frames that are both inside the corridor (`ft_CorrSpc`) and moving (`ft_move > 0`). Within each trial, the retained frames are then broken into consecutive 3-frame chunks, and each chunk becomes one decoder time bin.

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
```python
def chunk_indices(indices, chunk_size):
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
```
```python
    for tr, frame_idx in enumerate(frames_by_trial):
        chunks = chunk_indices(frame_idx, frames_per_bin)
        if not chunks:
            continue
```

iii. The notes say this matches the paper’s “running time points” restriction and that decoder bins are formed by averaging every 3 retained imaging frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops any trial that has no retained `ft_CorrSpc && ft_move > 0` frames, because such a trial produces no chunks. It also errors if an entire session has no retained running corridor frames.

ii. 
```python
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
```
```python
    selected_frames = np.concatenate(frames_by_trial)
    if selected_frames.size == 0:
        raise RuntimeError(f"No running corridor frames found for session {session['base']}")
```
```python
    for tr, frame_idx in enumerate(frames_by_trial):
        chunks = chunk_indices(frame_idx, frames_per_bin)
        if not chunks:
            continue
```

iii. The notes justify the frame restriction as matching the paper’s running-only analyses; there is no separate trial-quality rule beyond that retained-frame requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `spk_obj["spks"]` in the per-session spike file, concatenated across planes, and retinotopy assignments come from `iarea` in the session’s retinotopy file.

ii. 
```python
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. The notes say the traces are deconvolved Suite2p outputs and that retinotopy is grouped into coarse regions `V1`, `mHV`, `lHV`, and `aHV`.

## 2-b. How is the `neural` data processed?

i. The AI performs substantial extra processing: it restricts to coarse visual cortex, selects up to 128 neurons per session by variance with proportional allocation across regions, and averages every 3 retained imaging frames into one neural time bin. Neural trials are stored as `float32`.

ii. 
```python
def select_neurons(spk, region_idx_full, selected_frames, neurons_per_session):
    region_counts = [(region_idx_full == ridx).sum() for ridx in range(len(REGION_NAMES))]
    targets = proportional_region_targets(region_counts, neurons_per_session)
    var = variance_over_columns(spk, selected_frames)
    selected = []
    for ridx, target in enumerate(targets):
        if target <= 0:
            continue
        candidates = np.flatnonzero(region_idx_full == ridx)
        if candidates.size == 0:
            continue
        order = np.argsort(var[candidates])[::-1]
        picked = candidates[order[:target]]
        selected.append(np.sort(picked))
```
```python
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            trial_neural.append(neural_bin)
```
```python
        neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The notes and trajectory justify this as a tractability reduction: the AI says an all-neuron, frame-level dataset would be too large for the required pickle and decoder, so it reduced each session to 128 selected visual-cortex neurons and 3-frame bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to those with retinotopy assignments in the coarse visual areas `V1`, `mHV`, `lHV`, and `aHV`, and then a fixed-size subset is selected by variance. Sessions with no retained running frames or no selected neurons are rejected with an error.

ii. 
```python
def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
    out[(iarea == 5) | (iarea == 6)] = 2
    out[(iarea == 3) | (iarea == 4)] = 3
    return out
```
```python
    selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
    if selected_neurons.size == 0:
        raise RuntimeError(f"No visual-cortex neurons selected for session {session['base']}")
```

iii. The notes justify this as keeping retinotopically assigned visual-cortex neurons and proportionally allocating the retained subset across the four coarse areas.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata declares alignment to corridor entry / trial start, but in practice the neural bins are built from the ordered retained frames in each trial after applying the `ft_CorrSpc && ft_move > 0` filter. The bins therefore follow the moving corridor frames within each trial rather than a fixed window starting at the alignment frame.

ii. 
```python
def trial_frame_indices(beh, nfr):
    ft_trind = np.asarray(beh["ft_trInd"][:nfr])
    is_corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    is_move = np.asarray(beh["ft_move"][:nfr], dtype=np.float64) > 0
```
```python
    for tr, frame_idx in enumerate(frames_by_trial):
        chunks = chunk_indices(frame_idx, frames_per_bin)
```
```python
        "temporal_alignment_event": "corridor entry / trial start",
```

iii. The notes say the temporal alignment event is corridor entry / trial start, but also explicitly say only running corridor frames are retained and then averaged into 3-frame decoder bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI estimates the imaging frame interval from `ft`, then multiplies it by `frames_per_bin` to get the decoder bin size. With the default `frames_per_bin=3`, it rebins the data into 3-frame bins.

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
```
```python
        "time_bin_size": float(frame_dt_ms * args.frames_per_bin),
        "time_binning_rule": f"Average every {args.frames_per_bin} retained imaging frames into one decoder time bin.",
```

iii. The notes say decoder time bins are formed by averaging every 3 retained imaging frames in order within each trial.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from the per-trial absolute cue times in `SoundTime` and the absolute frame timestamps in `ft`.

ii. 
```python
    ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
    sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
```
```python
            chunk_ft = ft[chunk]
```

iii. The notes say the continuous timing variables are computed from the original frame timestamps (`ft`) and trial timing fields.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder chunk, the AI computes the mean value of `(SoundTime[trial] - ft[chunk]) * 86400`, i.e. average seconds remaining until the cue across the frames in that chunk.

ii. 
```python
            trial_input.append(
                np.array(
                    [
                        np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
                        day_value,
                        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
                        float(is_rew[tr]),
                    ],
                    dtype=np.float32,
                )
            )
```

iii. The notes explicitly describe `time_to_sound_cue_s` as the mean seconds to cue within each decoder bin.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same retained frame chunk that produced the neural bin, and computing the cue offset from the `ft` timestamps of that chunk.

ii. 
```python
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
```
```python
                        np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
```

iii. The notes say the timing variables are computed from the original frame timestamps within the same decoder bins used for neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session’s subject id `mname` and date string `datexp`, taken from the experiment index.

ii. 
```python
def parse_date(date_str):
    return dt.datetime.strptime(date_str, "%Y_%m_%d").date()
```
```python
def compute_training_days(sessions):
    by_subject = defaultdict(list)
    for idx, session in enumerate(sessions):
        by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
```

iii. The notes say `training_day` is defined as calendar days since the mouse’s first imaging session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the calendar-day offset from the earliest session date for that mouse, then broadcasts that scalar across all bins in every trial of the session.

ii. 
```python
    offsets = {}
    for subject, entries in by_subject.items():
        first_date = min(date for _, date in entries)
        for idx, date in entries:
            offsets[idx] = float((date - first_date).days)
    return offsets
```
```python
    day_value = np.float32(training_days[session_idx])
```
```python
                        day_value,
```

iii. The notes justify this choice directly: `training_day` is described as calendar days since the subject’s first imaging session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives this input from the per-trial absolute start times in `Trial_start_time` and the frame timestamps in `ft`.

ii. 
```python
    ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
    trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
```
```python
            chunk_ft = ft[chunk]
```

iii. The notes say the continuous timing variables are computed from original frame timestamps and trial timing fields.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each decoder chunk, the AI computes the mean elapsed seconds from trial start, `mean((ft[chunk] - Trial_start_time[trial]) * 86400)`.

ii. 
```python
                        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
```

iii. The notes describe `time_since_trial_start_s` as mean elapsed seconds since corridor entry within each decoder bin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by evaluating the trial-start offset on the same chunk of frame timestamps that was averaged into the neural bin.

ii. 
```python
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
```
```python
                        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
```

iii. The notes say the timing inputs are computed within the same decoder bins as the neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `isRew`, the per-trial rewarded-corridor flag.

ii. 
```python
    is_rew = np.asarray(beh["isRew"], dtype=bool)
```

iii. The notes say `reward_available` is constant within trial and taken from `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI only casts the trial flag and repeats it for every decoder bin in that trial.

ii. 
```python
                        float(is_rew[tr]),
```

iii. The notes give no extra processing beyond saying it is constant within trial and taken from `isRew`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, the per-trial stimulus identifier.

ii. 
```python
    wall_name = np.asarray(beh["WallName"])
```
```python
        stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The notes say `visual_stimulus` is taken directly from `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI gathers all unique `WallName` strings across the dataset, sorts them, and uses that full vocabulary as categorical output values. It does not collapse them to four base texture families.

ii. 
```python
def output_stimulus_names(canonical_lookup, sessions):
    names = set()
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        names.update(map(str, np.unique(beh["WallName"])))
    return sorted(names)
```
```python
    stim_names = output_stimulus_names(canonical_lookup, sessions)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```
```python
        stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The notes say the converter “records the full stimulus vocabulary present across sessions” and describe the stimulus as taken directly from `WallName`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the lick frame numbers.

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

iii. The notes say `licking` is binary per decoder bin using `LickFr.astype(int)` mapped to frame bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI converts valid lick frames into a per-frame boolean mask, then labels each decoder chunk as 1 if any lick occurred in the chunk and 0 otherwise.

ii. 
```python
    lick_mask = build_lick_frame_mask(beh, nfr)
```
```python
                        int(lick_mask[chunk].any()),
```

iii. The notes justify this as “binary per decoder bin,” using lick frames mapped onto the frame/bin grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by reading the same frame indices used for the neural chunk and applying `any()` over that chunk.

ii. 
```python
            neural_bin = spk_sel[:, chunk].mean(axis=1)
```
```python
                        int(lick_mask[chunk].any()),
```

iii. The notes say licking is represented per decoder bin on the same chunked frame grid as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the framewise corridor position.

ii. 
```python
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. The notes say `position_bin` covers the 4 m textured corridor with four bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI averages `ft_Pos` across each decoder chunk, then bins the chunk mean into 1 m categories.

ii. 
```python
            mean_pos = float(ft_pos[chunk].mean())
```
```python
                        position_to_bin(mean_pos),
```

iii. The notes only justify this at a high level: `position_bin` uses four bins covering the 4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The chunk-average position is converted from decimeters into 1 m bins with `floor(value / 10.0)` and then clipped to the range 0 to 3.

ii. 
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. The notes justify four position bins spanning the textured corridor, but do not add a more detailed rationale for the exact thresholding rule.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by evaluating `ft_Pos` on the same retained frame chunk that is averaged into the neural bin.

ii. 
```python
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            mean_pos = float(ft_pos[chunk].mean())
```

iii. The notes describe position as a per-decoder-bin output on the same retained frame grid as the neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, the framewise running speed.

ii. 
```python
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. The notes say `running_speed_bin` is based on mean `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes the mean `ft_RunSpeed` for every retained decoder chunk across all sessions, takes global quartile thresholds from those chunk means, and then bins each trial chunk’s mean speed by those thresholds.

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
```python
            mean_speed = float(ft_speed[chunk].mean())
```
```python
                        speed_to_bin(mean_speed, speed_thresholds),
```

iii. The notes say running-speed quartiles are computed globally over all retained decoder bins, not per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses three global numeric thresholds from `np.quantile(..., [0.25, 0.5, 0.75])` and applies `np.searchsorted(..., side="right")` to assign bins 0 through 3.

ii. 
```python
def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```
```python
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
```

iii. The notes justify this by saying the thresholds are computed globally from the retained decoder bins in the full dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by using the same retained frame chunk that is averaged into the neural bin.

ii. 
```python
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            mean_speed = float(ft_speed[chunk].mean())
```

iii. The notes describe running speed as a per-decoder-bin output on the same chunked frame grid as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds several defensive checks: it canonicalizes duplicated behavior entries and records mismatches, truncates behavioral arrays to `nfr = min(spk.shape[1], len(beh["ft"]))`, drops non-finite and out-of-range lick frames, and fails fast on missing behavior entries or sessions with no retained frames/neurons.

ii. 
```python
    canonical_lookup, provenance, duplicate_mismatches = build_behavior_lookup(args.root)
    missing = [s["base"] for s in sessions if s["base"] not in canonical_lookup]
    if missing:
        raise RuntimeError(f"Missing behavior entries for sessions: {missing[:5]}")
```
```python
    nfr = min(spk.shape[1], len(beh["ft"]))
```
```python
    lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```
```python
    if selected_frames.size == 0:
        raise RuntimeError(f"No running corridor frames found for session {session['base']}")
```

iii. The notes justify the duplicate-handling logic because many `Beh_*.npy` files reuse the same recording under figure-specific names; they say duplicate core arrays were checked and found to have zero mismatches.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is the repeated full-data passes: loading every behavior bundle into memory for canonicalization, computing global running-speed thresholds over all retained chunks, loading and concatenating the spike planes for each session, and computing neuron variances over retained frames for variance-based neuron selection.

ii. 
```python
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
```
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
```
```python
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```
```python
    var = variance_over_columns(spk, selected_frames)
```

iii. The notes and trajectory justify the added speed-threshold and neuron-selection passes as part of the AI’s tractability reductions for the decoder.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could be vectorized or fused: the per-trial mask-building in `trial_frame_indices`, the nested session/trial/chunk loops in `compute_speed_thresholds`, the analogous trial/chunk loops in `convert_session`, and the per-region loop in `select_neurons`.

ii. 
```python
    for tr in range(ntrials):
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
```
```python
    for session in sessions:
        ...
        for indices in trial_frame_indices(beh, nfr):
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
```
```python
    for tr, frame_idx in enumerate(frames_by_trial):
        chunks = chunk_indices(frame_idx, frames_per_bin)
        if not chunks:
            continue
        ...
        for chunk in chunks:
            ...
```
```python
    for ridx, target in enumerate(targets):
        if target <= 0:
            continue
```

iii. The notes do not specifically defend these loops; they mainly justify the high-level reduction strategy.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats frame grouping work: `trial_frame_indices` is run once during the global speed-threshold pass and again during session conversion. It also builds and retains full behavior provenance even though only one canonical behavior entry per session is used for conversion.

ii. 
```python
    speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
```
```python
    frames_by_trial = trial_frame_indices(beh, nfr)
```
```python
def compute_speed_thresholds(canonical_lookup, sessions, frames_per_bin):
    ...
        for indices in trial_frame_indices(beh, nfr):
```
```python
    canonical_lookup, provenance, duplicate_mismatches = build_behavior_lookup(args.root)
```

iii. The notes justify the global thresholds and provenance tracking as sanity checks and documentation, but they do not justify the repeated trial-frame reconstruction itself.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code constructs `speed_values` for every retained chunk only to derive three thresholds, and it carries provenance / duplicate-mismatch bookkeeping into metadata even though the decoder does not use it. Those steps are useful for documentation and checks, but not for downstream decoding.

ii. 
```python
    speed_values = np.asarray(speed_values, dtype=np.float32)
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return speed_values, q.astype(np.float32)
```
```python
        "behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```

iii. The notes justify these steps as sanity checks: duplicate provenance is documented, and the global speed-threshold computation is described explicitly in the validation notes.
