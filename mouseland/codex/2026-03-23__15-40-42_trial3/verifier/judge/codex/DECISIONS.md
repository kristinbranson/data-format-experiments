# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `data/beh/Imaging_Exp_info.npy`, groups entries by recording id `<mname>_<datexp>_<blk>`, then chooses one "canonical" behavior entry per recording by scanning duplicate experiment references. During conversion it loads the chosen behavior dictionary for each session, plus that session's spike file and retinotopy file.

ii. 
```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
```
```python
for exp_type, dbs in exp_info.items():
    for db in dbs:
        rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        per_rec[rec_id].append((exp_type, db))
```
```python
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says the true session unit is the unique recording id, that duplicate experiment-group references should be validated and collapsed, and that per-session loading keeps memory manageable.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `mname`. The converter stores `subject` in each `SessionRef`, sorts sessions by subject/date/block, builds a unique subject list, and writes `subject_idx` for each session.

ii. 
```python
SessionRef(
    rec_id=rec_id,
    subject=first_db["mname"],
    date_str=first_db["datexp"],
    blk=first_db["blk"],
    ...
)
```
```python
for sess in sessions:
    if sess.subject not in subject_to_idx:
        subject_to_idx[sess.subject] = len(subjects)
        subjects.append(sess.subject)
```

iii. The notes say `mname` is the raw subject identifier and should define `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. A session is one unique recording id `<mouse>_<date>_<blk>`. The AI merges all references to the same recording, chooses one canonical behavior key for that recording, and sorts sessions by subject, parsed date, and block.

ii. 
```python
rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
per_rec[rec_id].append((exp_type, db))
```
```python
sessions.sort(key=lambda s: (s.subject, parse_date(s.date_str), int(s.blk)))
```
```python
exp_type, beh_key = choose_canonical_behavior(rec_id, refs)
```

iii. The notes justify this by saying `Imaging_Exp_info.npy` contains duplicate experiment-group references to the same recording and the paper's 89 recordings should be recovered by deduplicating on recording id.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd`, but only running frames inside the corridor are kept. Each trial is therefore the set of frame indices satisfying `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`. Trials remain variable length; there is no truncation to 32 bins and no padding.

ii. 
```python
ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
valid = finite_trial & ft_corr & ft_move
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
```

iii. The notes explicitly say the export should use only running corridor frames because that is how the paper and utility code usually analyze activity.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a separate trial-quality metric. A trial is effectively required to have at least one retained running corridor frame; otherwise `compute_trial_masks` raises an error. There is no trial padding or rescue of empty trials.

ii. 
```python
frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
if len(frame_idx) == 0:
    raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

iii. The notes justify the running-only mask, not a distinct trial-quality rule. The behavior is essentially fail-fast on empty retained trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session spike chunks in `spks`, concatenated across imaging chunks, plus retinotopy `iarea` to decide which neurons to keep and how to label their brain regions.

ii. 
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
```
```python
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
areas = utils.neu_area_ID(iarea)
```

iii. The notes say the saved `spks` arrays are already deconvolved activity traces and `iarea` is the source of visual-area assignments.

## 2-b. How is the `neural` data processed?

i. The AI concatenates the spike chunks conceptually, computes a reference-style neuron subset, then for each trial extracts only the selected neurons at the retained frame indices. Each trial matrix is stored as `float16` and left variable length.

ii. 
```python
kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
```
```python
for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
    if len(local_rows) == 0:
        continue
    trial_chunks.append(chunk[local_rows][:, frame_idx])
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. In both the notes and trajectory, the AI justifies this as a tractability compromise: exporting all neurons would make the provided decoder impractical, so it exports a paper-inspired subset instead.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by a custom selection rule, not by keeping all neurons in broad visual areas. The AI keeps the union of: `mHV` neurons with extreme familiar-stimulus `d'` on odd running corridor frames, and `aHV` neurons with reward-prediction selectivity from interpolated position-aligned activity. If nothing survives, it falls back to the strongest `mHV` neurons.

ii. 
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
```
```python
interp_spk = utils.get_interpPos_spk(
    spk[ahv_idx][:, move_idx],
    poscum_move,
    int(beh["ntrials"]),
    n_bins=60,
    lengths=float(beh["Corridor_Length"]),
)
...
local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)
ahv_mask[ahv_idx[local_keep]] = True
```
```python
keep_mask = mhv_mask | ahv_mask
```

iii. The notes say this is a deliberate "reference-style subset" chosen so `train_decoder.py` can run, and the trajectory explicitly says the AI switched from exporting all neurons to this subset for tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to trial structure through the retained `frame_idx` for each trial, where `frame_idx` starts at the first retained running-corridor frame of that trial. The code does not create a fixed-length corridor-entry window; it simply uses the retained frames from that trial.

ii. 
```python
trial_masks = compute_trial_masks(beh)
...
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The notes justify this as keeping native frame-aligned trial structure while restricting to running corridor periods.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame interval as the bin size, estimated from the median `diff(ft)` across sessions and converted to milliseconds. No temporal rebinning is applied.

ii. 
```python
dts.append(np.median(np.diff(ft)) * 86400.0)
...
"time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
```

iii. The notes say the export stays frame aligned and does not resample time.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-frame timestamps `ft` and the trial-level cue timestamp `SoundTime`.

ii. 
```python
ft = np.asarray(beh["ft"], dtype=float)
sound_time = np.asarray(beh["SoundTime"], dtype=float)
```

iii. The notes say the AI chose actual behavioral timestamps for continuous timing covariates rather than frame-number interpolation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the AI computes `(SoundTime[trial] - ft[frame]) * 86400`, producing seconds until cue. This value is positive before cue and negative after cue.

ii. 
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes justify this as using real elapsed timestamps directly.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same retained frame indices used to slice the neural trial matrix.

ii. 
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
...
trial_input = np.vstack(
    [time_to_cue, day_of_training, time_since_start, reward_available]
)
```

iii. The AI's general justification is that all exported streams stay frame aligned on the retained trial frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's `datexp` string, grouped by subject (`mname`) and converted to actual calendar dates.

ii. 
```python
def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y_%m_%d")
```
```python
for subject, sess_list in by_subject.items():
    first_date = min(parse_date(s.date_str) for s in sess_list)
```

iii. The notes say the AI interpreted day of training as calendar-day offset from the first recorded session for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found; each session gets `(session_date - first_date).days + 1`. That value is then repeated across all retained frames in each trial.

ii. 
```python
day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```
```python
day_value = np.float32(day_map[sess.subject][sess.rec_id])
day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. The notes explicitly describe this as a calendar-day offset rather than an ordinal session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-frame timestamps `ft` and the trial-level timestamp `Trial_start_time`.

ii. 
```python
ft = np.asarray(beh["ft"], dtype=float)
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
```

iii. The notes say continuous timing inputs should use actual timestamps where available.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame in a trial, the AI computes `(ft[frame] - Trial_start_time[trial]) * 86400`, producing elapsed seconds from trial start.

ii. 
```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes justify this as direct timestamp subtraction on the retained frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices as the neural activity for that trial.

ii. 
```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The same frame-alignment justification is used throughout the export.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the trial-level boolean/numeric field `isRew`.

ii. 
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
```

iii. The notes say reward availability must come from raw per-trial `isRew` because the pooled dataset contains both rewarded and unrewarded cohorts.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI simply broadcasts `isRew[trial]` across all retained frames in that trial.

ii. 
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The notes describe this as a per-trial constant repeated over time.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level string label `WallName`.

ii. 
```python
wall_name = as_str_array(beh["WallName"])
```

iii. The notes justify using `WallName` because it is populated consistently, unlike `stim_id` in some sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects all unique `WallName` strings across the dataset, sorts them, maps each name to an integer category, and repeats the code across all retained frames of the trial. It does not collapse labels to four base texture classes.

ii. 
```python
categories = sorted(
    {
        str(v)
        for sess in sessions
        for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
    }
)
category_to_idx = {name: idx for idx, name in enumerate(categories)}
```
```python
stim_code = np.full(
    len(frame_idx),
    category_to_idx[str(wall_name[trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say this preserves all naturalistic, grating, and swap labels rather than collapsing them.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`, using both the lick frame and the associated trial index.

ii. 
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
```

iii. The code itself gives the main rationale: use the raw lick events already aligned to frames and trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI selects the lick frames belonging to the current trial, then marks each retained frame as `1` if it appears in that lick-frame list and `0` otherwise.

ii. 
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The notes treat licking as a binary, time-varying output aligned to retained frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is evaluated on the same `frame_idx` array used to slice the neural matrix for that trial.

ii. 
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
...
session_output.append(trial_output)
```

iii. The same frame-aligned export logic is used as for neural and input variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level position variable `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
```

iii. The notes say raw position is stored in decimeter units over the 4 m corridor and 2 m gray space.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. On retained frames, the AI clips position to the corridor range, then divides by 10 decimeters and floors to obtain 1 m bins.

ii. 
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes justify this as the requested four equal 1 m bins over the textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)` in the raw decimeter coordinate.

ii. 
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes describe the raw units as decimeters, so dividing by 10 yields 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken on the same retained `frame_idx` used for neural activity in that trial.

ii. 
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The AI's export logic keeps all streams aligned on the retained trial frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level speed variable `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
```

iii. The notes say speed should be discretized after filtering to the retained samples.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first collects all retained running-corridor frame speeds across all sessions, computes global quartile edges, and then bins each retained frame in each trial with those edges.

ii. 
```python
trial_masks = compute_trial_masks(beh)
speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
...
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The notes justify this by saying the user requested bins containing 25% of the retained data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global 25th, 50th, and 75th percentiles of all retained `ft_RunSpeed` values pooled across sessions, producing categories `q1` to `q4`.

ii. 
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The notes explicitly describe these as global quartiles over retained samples.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is indexed on the same retained `frame_idx` as the neural trial matrix.

ii. 
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The AI's general alignment rule is to slice every time-varying stream on the same retained frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mainly uses validation and fail-fast behavior rather than repair. It checks that duplicate behavior references agree on the fields it uses, errors if no behavior entry exists, and errors if a trial has no retained running corridor frames. It does not explicitly trim behavior to the number of imaged neural frames, and it does not impute missing values.

ii. 
```python
if not candidates:
    raise KeyError(f"No behavior entry found for recording {rec_id}")
```
```python
if not same:
    raise ValueError(
        f"Duplicate references for {rec_id} are not equivalent between "
        f"{candidates[0][1]}:{candidates[0][2]} and {exp_type}:{key}"
    )
```
```python
if len(frame_idx) == 0:
    raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

iii. The notes emphasize validation of duplicate references and assume that the retained frame masks are the correct operating domain; they do not describe special repair for missing or mismatched data beyond those checks.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive parts are repeated behavior loading across multiple passes, per-session loading of the large spike and retinotopy files, the neuron-selection step in `compute_selected_neurons`, and especially the aHV reward-prediction interpolation branch using `utils.get_interpPos_spk`.

ii. 
```python
beh_all = load_beh(exp_type)
...
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
...
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
```
```python
interp_spk = utils.get_interpPos_spk(
    spk[ahv_idx][:, move_idx],
    poscum_move,
    int(beh["ntrials"]),
    n_bins=60,
    lengths=float(beh["Corridor_Length"]),
)
```

iii. The notes explicitly identify reward-prediction interpolation as expensive and describe several optimizations aimed at reducing its cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code loops over trials to build `trial_masks`, loops again over trials to build outputs, and within each trial loops over spike chunks before concatenation. It also repeatedly reloads behavior in Python loops and comprehensions instead of caching it once.

ii. 
```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
```
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        ...
```
```python
categories = sorted(
    {
        str(v)
        for sess in sessions
        for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
    }
)
```

iii. There is no substantive justification in the notes beyond the AI's focus on keeping peak memory down.

## 12-c. What processing does the code repeat multiple times?

i. The same behavior files are loaded repeatedly in `choose_canonical_behavior`, sample-session selection, `build_global_metadata`, the category-gathering comprehension, and the main conversion loop. Session/date information is also reparsed multiple times.

ii. 
```python
beh_all = load_beh(exp_type)
```
```python
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
```
```python
"time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
```

iii. The notes mention a behavior-only first pass for speed bins, but they do not justify the repeated behavior reloads beyond that.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes expensive selectivity statistics (`stim_dp`, reward-prediction interpolation, `kept_stats`) that are used only to choose a neuron subset for tractability, not as exported variables. It also accumulates `category_values` and `dts` in `build_global_metadata`, but `category_values` is unused and `dts` is not returned. Optional plotting code also produces side artifacts that are not part of the converted dataset.

ii. 
```python
category_values = set()
for sess in sessions:
    ...
    category_values.update(str(v) for v in np.asarray(beh["WallName"]).tolist())
```
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
...
stats = {
    "n_total": int(len(iarea)),
    "n_mhv_selected": int(np.sum(mhv_mask)),
    "n_ahv_selected": int(np.sum(ahv_mask)),
    "n_selected": int(np.sum(keep_mask)),
```
```python
if show_processing and processing_plotted < 2:
    session_processing_summary(...)
```

iii. The notes explicitly frame the selectivity computations as a tractability workaround rather than something required by the exported decoder dataset.
