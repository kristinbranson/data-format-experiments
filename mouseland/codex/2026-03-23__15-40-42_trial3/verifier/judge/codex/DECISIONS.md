# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `data/beh/Imaging_Exp_info.npy`, groups entries by unique recording id `<mouse>_<date>_<blk>`, chooses one "canonical" behavior entry for each recording by scoring duplicate references, and then loads behavior, spike, and retinotopy files per session during conversion.

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

iii. In `CONVERSION_NOTES.md` Step 5, the AI says session identity should be the unique recording id and duplicate experiment-group references should be merged. In the trajectory, it also notes it chose a "canonical behavior" entry after validating duplicates.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `mname` from the experiment index. The script stores `subject` on each `SessionRef`, builds a unique ordered subject list, and writes `subject_idx` per session.

ii. 
```python
sessions.append(
    SessionRef(
        rec_id=rec_id,
        subject=first_db["mname"],
        ...
    )
)
```
```python
subjects = []
subject_to_idx = {}
for sess in sessions:
    if sess.subject not in subject_to_idx:
        subject_to_idx[sess.subject] = len(subjects)
        subjects.append(sess.subject)
```

iii. The AI's notes say `mname` is the reliable subject identifier in `Imaging_Exp_info.npy`, so no derived split is needed beyond deduped session assembly.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique recording id `mname_datexp_blk`. Duplicate appearances across experiment groups are merged into a single `SessionRef`.

ii. 
```python
rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
per_rec[rec_id].append((exp_type, db))
```
```python
for rec_id, refs in per_rec.items():
    ...
    sessions.append(SessionRef(rec_id=rec_id, ...))
```

iii. `CONVERSION_NOTES.md` Step 4 explicitly says the raw index contains 142 experiment references but only 89 unique recording ids, so the AI chose the unique recording id as the true session unit.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd`, but the AI keeps only frames that are both in the textured corridor and during running. Each trial becomes the list of retained frame indices satisfying `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`.

ii. 
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ...
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    ...
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
```

iii. In Step 5 of the notes, the AI justifies this as "use only running corridor frames" to match the paper's running-only analyses and because the decoder outputs are corridor variables. The trajectory also says it settled on "frame-aligned trial construction on running corridor frames."

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference's global long-trial filter. Instead, it requires every trial to have at least one retained running corridor frame; otherwise `compute_trial_masks` raises an error. There is no percentile-based removal of abnormally long trials.

ii. 
```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
    if len(frame_idx) == 0:
        raise ValueError(f"Trial {trial} has no retained running corridor frames")
    masks.append(frame_idx)
```

iii. The notes frame this as a direct consequence of the running-only trial definition, not as a separate trial-QC rule. No justification is given for omitting the reference's 99th-percentile long-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays come from the saved deconvolved `spks` arrays in the per-session spike file plus retinotopy `iarea`, which is used to support neuron selection and brain-region labeling.

ii. 
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```
```python
spk = np.concatenate(spk_chunks, axis=0)
areas = utils.neu_area_ID(iarea)
```

iii. `CONVERSION_NOTES.md` Step 1 states there is no dF/F step in the paper code and that the saved `spks` arrays are already analysis-ready deconvolved traces.

## 2-b. How is the `neural` data processed?

i. The AI concatenates spike chunks, computes reference-style selectivity metrics, keeps only a selected neuron subset, slices each retained trial to running corridor frames, and stores each trial matrix as `float16`.

ii. 
```python
spk = np.concatenate(spk_chunks, axis=0)
...
kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
...
trial_chunks.append(chunk[local_rows][:, frame_idx])
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. In Step 6 of the notes and trajectory steps 119, 123, and 127, the AI says it intentionally introduced a "reference-style subset" because exporting all neurons would make `train_decoder.py` intractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by a custom selectivity-based subset, not by keeping all neurons in the four visual areas. The retained set is the union of:
- mHV neurons in the top or bottom 5% of familiar-stimulus `d'` among corridor-responsive neurons, with a fallback minimum of 128 strongest mHV neurons.
- aHV neurons with reward-prediction `d' >= 0.3` and nonnegative stimulus selectivity.

ii. 
```python
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
...
if int(np.sum(mhv_mask)) < 128:
    ...
    mhv_mask[take] = True
```
```python
reward_dp = dprime(mean_corr[:, late], mean_corr[:, early])
local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)
ahv_mask[ahv_idx[local_keep]] = True
...
keep_mask = mhv_mask | ahv_mask
```

iii. The AI explicitly justifies this in the notes and trajectory as a tractability compromise meant to mimic the paper's neuron-selection analyses while reducing decoder memory load.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (`trial start`) at the metadata level, but the exported neural data only include retained running corridor frames from each trial rather than every corridor frame from entry onward.

ii. 
```python
"temporal_alignment_event": "corridor entry (trial start)",
```
```python
trial_masks = compute_trial_masks(beh)
...
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The notes justify this as preserving frame alignment while matching the paper's running-only analyses. The AI did not justify the loss of non-running corridor frames relative to the reference solution.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native imaging frame interval. The AI computes a median frame interval from `ft` and stores it in milliseconds. No temporal rebinning is applied.

ii. 
```python
dts.append(np.median(np.diff(ft)) * 86400.0)
...
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
"time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
```

iii. In Step 5 and Step 10, the AI says exported trials stay on native imaging-frame bins and cites a median interval of about `314.69 ms`.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundTime` and frame-level `ft`.

ii. 
```python
ft = np.asarray(beh["ft"], dtype=float)
sound_time = np.asarray(beh["SoundTime"], dtype=float)
...
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. Step 5 of the notes says the AI preferred "actual behavioral timestamps" from `ft` and `SoundTime` instead of deriving cue time from frame numbers.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the AI subtracts that frame's timestamp from the trial's cue timestamp and converts days to seconds. The value is positive before the cue and negative after the cue.

ii. 
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes justify this as using the raw continuous timestamps directly rather than interpolating event frame numbers back onto time.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same `frame_idx` array used to extract the neural trial.

ii. 
```python
trial_chunks.append(chunk[local_rows][:, frame_idx])
...
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes repeatedly state that all exported inputs and outputs are computed on the same retained frame indices as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's recording date `datexp`, grouped by subject.

ii. 
```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    ...
    first_date = min(parse_date(s.date_str) for s in sess_list)
    for sess in sess_list:
        day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

iii. The notes say the AI intentionally used a calendar-day offset from the first recorded day for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the AI finds the earliest recording date, computes `days since first date + 1` for every session, then broadcasts that scalar across every timepoint in the trial.

ii. 
```python
day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```
```python
day_value = np.float32(day_map[sess.subject][sess.rec_id])
...
day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. In Step 5, the AI says this is a "continuous per-trial covariate required by the user task." It does not justify why calendar elapsed days are preferable to session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from trial-level `Trial_start_time` and frame-level `ft`.

ii. 
```python
ft = np.asarray(beh["ft"], dtype=float)
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
...
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes say the AI again chose the direct timestamp fields rather than inferring start time from frame numbers.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the AI subtracts the trial's `Trial_start_time` from that frame's timestamp and converts days to seconds.

ii. 
```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The justification in the notes is the same as for cue time: use raw continuous timestamps directly.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained `frame_idx` used for the neural export.

ii. 
```python
trial_chunks.append(chunk[local_rows][:, frame_idx])
...
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The AI's Step 10 spot checks explicitly verify the exported neural and input arrays agree on those retained frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial `isRew` flag.

ii. 
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
...
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The notes justify this as necessary because many sessions are fully unrewarded or partially rewarded, so reward availability must come from the raw trial flag rather than be assumed from task design.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The only processing is broadcasting the scalar `isRew[trial_idx]` across all retained timepoints of that trial.

ii. 
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The AI's notes explicitly say this should be carried per trial from raw `isRew`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. 
```python
wall_name = as_str_array(beh["WallName"])
...
stim_code = np.full(
    len(frame_idx),
    category_to_idx[str(wall_name[trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say `WallName` was preferred because `stim_id` can be missing or ambiguous in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse stimuli to four base texture classes. Instead, it gathers all unique `WallName` strings across sessions, sorts them, assigns an integer to each raw wall name, and broadcasts that per-trial category across all retained timepoints.

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

iii. Step 5 of the notes explicitly says the AI wanted to "preserve all naturalistic / grating / swap categories present in the raw trials."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii. 
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
...
lick_trial_frames = lick_fr[lick_tr == trial_idx]
```

iii. The AI's notes say it wanted a frame-aligned binary licking output and used the trial index field to associate licks with each trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI marks a retained frame as `1` if its raw frame index appears in that trial's lick-frame list and `0` otherwise.

ii. 
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The notes describe this as a binary time-varying output on the same retained frame grid as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is computed by testing the same retained neural `frame_idx` against the trial's lick-frame list.

ii. 
```python
trial_chunks.append(chunk[local_rows][:, frame_idx])
...
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. In Step 10, the AI says raw-vs-converted spot checks verified licking exactly on those retained frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
...
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. The notes say position should come from the frame-aligned corridor position trace and then be discretized into 1 m bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips positions into the 4 m textured corridor, then integer-divides by `10` to convert decimeters into four 1 m bins.

ii. 
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. In Step 5, the AI explicitly notes that raw positions are in decimeters and the requested output requires four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are `0-10`, `10-20`, `20-30`, and `30-40` decimeters, encoded as integers `0` to `3`.

ii. 
```python
"output_values": [
    categories,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ["q1", "q2", "q3", "q4"],
],
```
```python
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes justify this as the paper-consistent mapping from 40 dm of textured corridor to four 1 m categories.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position bins are read on the same retained `frame_idx` used for the neural trial.

ii. 
```python
trial_chunks.append(chunk[local_rows][:, frame_idx])
...
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes describe the export as frame aligned, with position derived directly from the retained frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed`.

ii. 
```python
speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
```
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
...
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The notes say speed quartiles should be computed from the retained frame-level running-speed samples.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI pools `ft_RunSpeed` from all retained trial frames across all sessions, computes global quartile edges with `np.quantile`, and then digitizes each retained frame against those edges.

ii. 
```python
for sess in sessions:
    ...
    trial_masks = compute_trial_masks(beh)
    speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
...
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. Step 5 of the notes explicitly says the AI wanted "global quartiles over retained samples" because the task asked for bins containing 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the three global quantile edges stored in metadata. Bins are assigned by `np.digitize(..., right=False)` and clipped into category indices `0` to `3`.

ii. 
```python
"speed_bin_edges": [float(x) for x in speed_edges.tolist()],
```
```python
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The AI justifies this in the notes as the most literal way to satisfy the user's "25% of the data" requirement.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running-speed bins are computed from `ft_RunSpeed[frame_idx]` for the same retained frame indices used for the neural export.

ii. 
```python
trial_chunks.append(chunk[local_rows][:, frame_idx])
...
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The notes say all time-varying outputs are computed on the same retained frame grid as `neural`.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles duplicate experiment references by validating agreement and choosing one canonical behavior entry. It also raises an error if a trial has no retained running corridor frames. It does not implement the reference solution's explicit trimming of behavior streams to the number of imaged frames before trial extraction.

ii. 
```python
for exp_type, key, beh in checked[1:]:
    same = (
        int(beh["ntrials"]) == int(base_beh["ntrials"])
        and np.array_equal(as_str_array(beh["WallName"]), as_str_array(base_beh["WallName"]))
        ...
    )
    if not same:
        raise ValueError(...)
```
```python
if len(frame_idx) == 0:
    raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

iii. The notes discuss duplicate-reference validation and raw spot checks, but they do not justify the lack of an explicit `n_frames = spikes.shape[1]` truncation step like the human reference uses.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading and concatenating the per-session spike chunks and, for rewarded sessions, running the reward-prediction interpolation branch over aHV neurons.

ii. 
```python
spk = np.concatenate(spk_chunks, axis=0)
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

iii. `CONVERSION_NOTES.md` Step 6 and Step 7 explicitly identify reward-prediction interpolation as the expensive branch, and the trajectory says that is what made rewarded sessions slower.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several Python loops in place: iterating over sessions, iterating over all trials to build `trial_masks`, and then iterating over every trial and every spike chunk to build each exported neural array.

ii. 
```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
```
```python
for sess_idx, sess in enumerate(sessions):
    ...
    for trial_idx, frame_idx in enumerate(trial_masks):
        trial_chunks = []
        for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
            ...
            trial_chunks.append(chunk[local_rows][:, frame_idx])
```

iii. The notes do not present a defense of these loops beyond runtime measurements and a few memory-saving choices. This is mostly evident from the implementation itself.

## 12-c. What processing does the code repeat multiple times?

i. The script repeatedly reloads behavior files and recomputes derived session metadata. `load_beh(...)` is called when choosing canonical behavior, again in metadata assembly, again when gathering categories and time-bin statistics, and again during final conversion. Trial masks are computed in metadata assembly and then recomputed during conversion.

ii. 
```python
beh_all = load_beh(exp_type)
```
```python
for sess in sessions:
    beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
    ...
    trial_masks = compute_trial_masks(beh)
```
```python
for sess_idx, sess in enumerate(sessions):
    beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
    ...
    trial_masks = compute_trial_masks(beh)
```

iii. The AI notes some speedups it did implement, but it does not justify these repeated behavior loads or repeated `compute_trial_masks` calls.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several intermediates that are not exported directly: full-session `stim_dp`, `corr_neu`, `mhv_mask`, `ahv_mask`, interpolated reward-prediction activity, and temporary `region_names`. It also builds `dts` and `category_values` in `build_global_metadata`, but `category_values` is unused. Most raw neurons are loaded and scored only to be discarded after subset selection.

ii. 
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
...
interp_spk = utils.get_interpPos_spk(...)
...
region_names = np.empty(np.sum(keep_mask), dtype=object)
```
```python
dts = []
speed_values = []
category_values = set()
...
category_values.update(str(v) for v in np.asarray(beh["WallName"]).tolist())
```

iii. The AI justifies the selectivity computations as necessary for its tractability-driven neuron subset, but from the perspective of downstream decoder use they are preparatory work that is not retained in the exported dataset.
