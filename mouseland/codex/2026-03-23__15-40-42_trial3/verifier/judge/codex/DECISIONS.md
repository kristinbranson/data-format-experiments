# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the experiment index from `data/beh/Imaging_Exp_info.npy`, groups entries by unique recording ID `<mouse>_<date>_<blk>`, chooses one canonical behavior dictionary entry for each recording from the relevant `Beh_<exp_type>.npy` file, and then loads each session's neural data from `data/spk/<rec_id>_neural_data.npy` plus retinotopy from `data/retinotopy/<mouse>_<date>_trans.npz`. Trials are built later from frame-level behavior arrays in the selected behavior dictionary.

ii.
```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()

for exp_type, dbs in exp_info.items():
    for db in dbs:
        rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        per_rec[rec_id].append((exp_type, db))

beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent says the paper has 89 unique recordings but `Imaging_Exp_info.npy` contains duplicated experiment-group references, so it decided the true session unit should be the unique recording ID. The notes also say the converter should load the same raw sources as the reference `load_spk` and `load_retino` functions.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name `mname`. The converter stores a unique ordered subject list and a `subject_idx` array mapping each session to its mouse.

ii.
```python
SessionRef(
    rec_id=rec_id,
    subject=first_db["mname"],
    date_str=first_db["datexp"],
    blk=first_db["blk"],
    ...
)

subjects = []
subject_to_idx = {}
for sess in sessions:
    if sess.subject not in subject_to_idx:
        subject_to_idx[sess.subject] = len(subjects)
        subjects.append(sess.subject)

"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int64),
```

iii. The notes explicitly state that `mname` is used for `subjects` and that session order follows the unique-recording ordering after deduplication.

## 1-c. How are the data split into sessions?

i. Sessions are split by the unique recording key `<mouse>_<date>_<blk>`. Duplicate references across experiment groups are merged, and one canonical behavior entry is selected after checking that the duplicates agree on key trial annotations.

ii.
```python
rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
per_rec[rec_id].append((exp_type, db))

same = (
    int(beh["ntrials"]) == int(base_beh["ntrials"])
    and np.array_equal(as_str_array(beh["WallName"]), as_str_array(base_beh["WallName"]))
    and np.array_equal(np.asarray(beh["isRew"]).astype(int), np.asarray(base_beh["isRew"]).astype(int))
    and np.allclose(np.asarray(beh["SoundPos"], dtype=float), np.asarray(base_beh["SoundPos"], dtype=float), equal_nan=True)
    and np.allclose(np.asarray(beh["ft_trInd"], dtype=float), np.asarray(base_beh["ft_trInd"], dtype=float), equal_nan=True)
)

sessions.sort(key=lambda s: (s.subject, parse_date(s.date_str), int(s.blk)))
```

iii. The notes say this resolves 142 experiment entries down to the paper-consistent 89 recordings, and that duplicate experiment-group entries are analysis references rather than separate neural sessions.

## 1-d. How are the data split into trials?

i. Trials are defined from `beh["ntrials"]` and frame-level trial labels `beh["ft_trInd"]`. For each trial index, the code collects the retained frame indices whose trial label matches that trial.

ii.
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ft_trial_int = np.full(ft_trial.shape, -1, dtype=np.int64)
    finite_trial = np.isfinite(ft_trial)
    ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
    ...
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        ...
        masks.append(frame_idx)
```

iii. The notes describe the export as frame-aligned and say trial construction should preserve the native frame grid while using the raw trial labels and trial-start timing fields.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply a paper-style whole-trial rejection rule. Instead, it filters frames within each trial to keep only finite trial labels, corridor frames, and running frames. If a trial has no surviving frames after this mask, the script raises an error rather than silently keeping an empty trial.

ii.
```python
ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
valid = finite_trial & ft_corr & ft_move
...
if len(frame_idx) == 0:
    raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

iii. The notes justify this by citing the paper statement that only running timepoints were analyzed and by saying the export should use running corridor frames only.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported `neural` matrices come from the `spks` arrays in each session's `*_neural_data.npy` file, plus retinotopy `iarea` labels used to select and label neurons by region.

ii.
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```

iii. The notes say the saved `spks` arrays are the Suite2p deconvolved traces used by the paper and that no extra dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The agent concatenates all `spks` chunks, computes stimulus and reward selectivity, keeps only a subset of neurons, slices those neurons on the retained frame indices for each trial, and stores the result as `float16` without temporal rebinning in the exported dataset.

ii.
```python
spk = np.concatenate(spk_chunks, axis=0)
...
kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
...
trial_chunks.append(chunk[local_rows][:, frame_idx])
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The notes say the original plan to export all neurons was abandoned because the decoder concatenates all sessions in memory, so the agent switched to a "reference-style subset" for tractability.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not keep all recorded neurons. It keeps the union of two selected populations: `mHV` neurons in the top/bottom 5% of familiar-stimulus d-prime among corridor-responsive neurons on odd running corridor frames, and `aHV` neurons with reward-prediction d-prime at least `0.3` among positively stimulus-selective neurons. If no neurons survive, it falls back to the strongest `mHV` neurons by absolute d-prime.

ii.
```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
    spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)
)

mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
...
reward_dp = dprime(mean_corr[:, late], mean_corr[:, early])
local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)
ahv_mask[ahv_idx[local_keep]] = True

keep_mask = mhv_mask | ahv_mask
```

iii. The notes and trajectory repeatedly justify this as a tractability compromise that reuses the paper's analysis logic instead of arbitrary downsampling.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry / trial start. The agent keeps the native imaging frames belonging to each trial after the running-corridor mask, and the metadata declares the temporal alignment event as corridor entry.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "corridor entry (trial start)",
    "off_start": 0.0,
    "off_end": None,
    ...
}

for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The notes say the export should be frame-aligned to trial start rather than exporting position-interpolated neural activity, because the requested decoder outputs include frame-resolved behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported data use the native imaging frame bins. The script reports the median `ft` step in milliseconds as metadata and does not rebin the exported trial matrices in time. The only interpolation in the script is internal position interpolation used during neuron selection.

ii.
```python
"time_bin_size": float(np.median([
    np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0
    for s in sessions
]) * 1000.0),

neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The notes say the notebook's imaging frame rate is about `3.17 Hz` and explicitly state that neural data stay on the native imaging frame bins in the export.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from frame timestamps `beh["ft"]` and per-trial cue timestamps `beh["SoundTime"]`.

ii.
```python
ft = np.asarray(beh["ft"], dtype=float)
sound_time = np.asarray(beh["SoundTime"], dtype=float)
...
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes say the converter uses the raw behavioral timestamps rather than approximate position-derived timing.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the agent subtracts the current frame time from that trial's `SoundTime` and converts days to seconds by multiplying by `86400`.

ii.
```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes define this variable as signed seconds relative to cue time, positive before cue and negative after cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same `frame_idx` values used to slice the neural matrix for that trial, so it has one value per exported neural time bin.

ii.
```python
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes explicitly say all decoder inputs/outputs are built on the same retained frame indices used for neural export.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata in `Imaging_Exp_info.npy`: mouse name `mname` and session date `datexp`.

ii.
```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    ...
    first_date = min(parse_date(s.date_str) for s in sess_list)
    for sess in sess_list:
        day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

iii. The notes describe this as a derived covariate requested by the decoder task rather than a field taken directly from the paper code.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the agent sorts sessions by calendar date, computes days since the mouse's first recording plus one, and repeats that scalar across all retained frames of each trial.

ii.
```python
day_value = np.float32(day_map[sess.subject][sess.rec_id])
day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. The notes say the value is continuous, per-trial, and broadcast across time to match the decoder input shape.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. The agent did not create an `Environment type` input at all. The exported `input_names` contain only `time_to_sound_cue`, `day_of_training`, `time_since_trial_start`, and `reward_available`.

ii.
```python
"input_names": [
    "time_to_sound_cue",
    "day_of_training",
    "time_since_trial_start",
    "reward_available",
],
```

iii. There is no explicit justification in the code, but the task instructions given to the agent did not ask for an `Environment type` decoder input, so the omission appears to follow the stated decoder spec rather than the duplicated question template.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. No environment/cohort/task-type field is computed or exported.

ii.
```python
"input_names": [
    "time_to_sound_cue",
    "day_of_training",
    "time_since_trial_start",
    "reward_available",
],
```

iii. The notes never mention environment type as an exported variable; they focus on the four decoder inputs required by the task.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame timestamps `beh["ft"]` and per-trial start timestamps `beh["Trial_start_time"]`.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
...
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes say the export is aligned to corridor entry and that continuous timing variables should use the real timestamp fields.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame in a trial, the agent subtracts the trial-start time from the frame timestamp and converts the result from days to seconds.

ii.
```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes describe this as elapsed seconds since corridor entry, using `Trial_start_time` and `ft`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices as the neural trial matrix, so it is exactly frame-aligned to the exported neural data.

ii.
```python
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes say all exported variables are generated from the shared retained-frame mask.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial behavior field `beh["isRew"]`.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
...
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The notes emphasize that reward availability must come from raw `isRew` instead of being inferred from task design, because many sessions are unrewarded or partially rewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent takes the trial's scalar `isRew` value and broadcasts it across all retained frames in that trial as a time-constant input channel.

ii.
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The notes say per-trial constants are repeated across frames so every trial has a consistent `(n_features, n_timepoints)` shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level behavior field `beh["WallName"]`, after the converter first chooses a canonical behavior entry for each session.

ii.
```python
wall_name = as_str_array(beh["WallName"])
...
categories = sorted({
    str(v)
    for sess in sessions
    for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
})
```

iii. The notes justify using `WallName` rather than `stim_id` because `WallName` preserves the actual stimulus labels and `stim_id` can be missing or ambiguous in some duplicate entries.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent builds a global sorted category list from all sessions, maps each trial's `WallName` to an integer code, and repeats that code across the trial's retained frames.

ii.
```python
category_to_idx = {name: idx for idx, name in enumerate(categories)}
...
stim_code = np.full(
    len(frame_idx),
    category_to_idx[str(wall_name[trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say this preserves all naturalistic, grating, and swap categories present in the raw data.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from lick frame indices `beh["LickFr"]` and lick trial labels `beh["LickTrind"]`.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
...
lick_trial_frames = lick_fr[lick_tr == trial_idx]
```

iii. The notes say licking should be represented as a frame-aligned binary time series.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the converter takes the lick frames assigned to that trial and marks each retained frame as `1` if its frame index appears in the lick-frame list, else `0`.

ii.
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The notes say this produces a binary time-varying output on the same frame grid as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary licking vector is evaluated directly on the retained `frame_idx` values, so each exported neural time bin has a matching lick label.

ii.
```python
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
trial_output = np.vstack([stim_code, licking, pos_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. The notes explicitly state that licking is built from raw lick frames restricted to the same retained frame indices used for neural export.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level position variable `beh["ft_Pos"]`.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
...
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. The notes say position output should come from frame-aligned position traces and be discretized over the 4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent takes the retained-frame positions, clips them to the corridor range `[0, 39.999)` decimeters, and converts them to 0-based 10-decimeter bins.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes justify this by saying the paper/code represent the texture corridor as 40 decimeters and the task requires four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four equal-width bins: `0-1m`, `1-2m`, `2-3m`, and `3-4m`, implemented as decimeter intervals `[0,10)`, `[10,20)`, `[20,30)`, `[30,40)`.

ii.
```python
"output_values": [
    categories,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ["q1", "q2", "q3", "q4"],
],

pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes explicitly describe the corridor as 40 dm of texture area and map that to four 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position categories are computed on the same retained frame indices as the neural trial matrix, so they are time-varying and frame-aligned.

ii.
```python
neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes say position-derived outputs are computed from frame-aligned position traces on the shared retained-frame mask.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level behavior variable `beh["ft_RunSpeed"]`.

ii.
```python
speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
...
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The notes say running speed should be binned after all frame filtering decisions are applied.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first pools running-speed values from all retained frames across all sessions, computes global quartile cut points, and then digitizes each retained frame's speed using those edges.

ii.
```python
speed_values = np.concatenate(speed_values).astype(np.float32)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The notes explicitly justify this as matching the user request that each speed bin contain about 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into four quartile bins labeled `q1` through `q4` using the global 25th, 50th, and 75th percentile edges of retained-frame `ft_RunSpeed`.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
...
"output_values": [
    categories,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ["q1", "q2", "q3", "q4"],
],
```

iii. The notes say quartiles are computed globally over all retained samples so occupancy is near 25% per class.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running-speed bins are computed from `ft_RunSpeed[frame_idx]`, so they are aligned one-to-one with the exported neural time bins.

ii.
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
trial_output = np.vstack([stim_code, licking, pos_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. The notes say running-speed output is built from raw frame-aligned running speed on the same retained frame indices used for neural export.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mostly handles such cases with validation, masking, and fallbacks rather than imputation. It raises errors if no behavior entry exists or duplicate references disagree, converts non-finite d-prime values to zero, ignores frames with non-finite `ft_trInd`, skips reward-prediction interpolation unless cumulative position is monotonic, and falls back to strongest `mHV` neurons if selection yields none.

ii.
```python
if not candidates:
    raise KeyError(f"No behavior entry found for recording {rec_id}")
...
out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
...
finite_trial = np.isfinite(ft_trial)
...
if np.all(np.diff(poscum_move) >= 0):
    ...
if int(np.sum(keep_mask)) == 0:
    ...
    keep_mask[take] = True
```

iii. The notes frame this as conservative handling: validate duplicates, preserve real raw values where possible, and fail loudly on broken trial structure rather than silently inventing data.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the very large neural recordings, concatenating the `spks` chunks, running the position interpolation used for `aHV` reward-neuron selection, and then building trial matrices across all sessions.

ii.
```python
spk = np.concatenate(spk_chunks, axis=0)
...
interp_spk = utils.get_interpPos_spk(
    spk[ahv_idx][:, move_idx],
    poscum_move,
    int(beh["ntrials"]),
    n_bins=60,
    lengths=float(beh["Corridor_Length"]),
)
...
for trial_idx, frame_idx in enumerate(trial_masks):
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        ...
        trial_chunks.append(chunk[local_rows][:, frame_idx])
```

iii. The notes explicitly identify reward-prediction interpolation as the expensive branch and give per-session runtime estimates dominated by that work.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-mask loop over `range(ntrials)`, the per-session/per-trial/per-chunk export loops, and repeated Python-level metadata scans across sessions could all have been reduced with more vectorized batching or caching.

ii.
```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))

for sess_idx, sess in enumerate(sessions):
    ...
    for trial_idx, frame_idx in enumerate(trial_masks):
        trial_chunks = []
        for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
            ...
```

iii. The notes mention efficiency concerns and describe several speedups the agent added, implying these remaining loops are where most overhead remains.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly reloads behavior dictionaries, recomputes trial masks, rescans `WallName` categories, and recomputes per-session frame-timestep summaries. It also performs one full metadata pass before the actual conversion pass.

ii.
```python
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
...
trial_masks = compute_trial_masks(beh)
...
categories = sorted({
    str(v)
    for sess in sessions
    for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
})
...
"time_bin_size": float(np.median([
    np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0
    for s in sessions
]) * 1000.0),
```

iii. The notes say the converter intentionally does a behavior-only first pass for speed quartiles and metadata, then a second pass with neural loading for conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The most notable discarded processing is the expensive position interpolation used only to select `aHV` reward-prediction neurons; that interpolated activity is not exported. The script also builds some unused metadata intermediates such as `category_values` and `dts`, and it can optionally generate processing plots that are not part of the converted dataset.

ii.
```python
dts = []
speed_values = []
category_values = set()
...
category_values.update(str(v) for v in np.asarray(beh["WallName"]).tolist())

interp_spk = utils.get_interpPos_spk(...)
mean_corr = interp_spk[:, :, 5:40].mean(axis=2)
...
if show_processing and processing_plotted < 2:
    session_processing_summary(...)
```

iii. The notes explicitly say interpolation is retained because it supports reference-style neuron selection, even though the exported neural data remain frame-aligned and do not store the interpolated arrays.
