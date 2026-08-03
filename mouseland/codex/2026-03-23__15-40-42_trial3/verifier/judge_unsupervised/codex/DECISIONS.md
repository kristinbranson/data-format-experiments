# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads the master session index from `data/beh/Imaging_Exp_info.npy`, groups entries by unique recording ID `<mouse>_<date>_<blk>`, chooses one canonical behavior dictionary entry per recording, and then loads per-session behavior, neural, and retinotopy files during conversion. Trials are not preloaded globally; they are reconstructed session by session from behavior frame annotations.

ii. ```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()

def collect_sessions(sample: bool) -> list[SessionRef]:
    exp_info = load_exp_info()
    per_rec: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            per_rec[rec_id].append((exp_type, db))
```

```python
for sess_idx, sess in enumerate(sessions):
    beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
    spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
    spk_chunks = list(spk_obj["spks"])
    ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
```

iii. The notes say the agent resolved the 142 experiment entries down to the paper-consistent 89 unique recordings and used behavior keys only as annotations for those recordings. The trajectory and notes describe this as necessary because the experiment index contains duplicate references to the same neural recording.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse name `mname`. Each session stores `subject=first_db["mname"]`, and `build_global_metadata()` creates an ordered unique subject list plus a per-session `subject_idx`.

ii. ```python
sessions.append(
    SessionRef(
        rec_id=rec_id,
        subject=first_db["mname"],
        date_str=first_db["datexp"],
        blk=first_db["blk"],
        canonical_exp_type=exp_type,
        canonical_beh_key=beh_key,
        refs=tuple((exp, db.get("stimtype")) for exp, db in refs),
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

iii. In the notes, the agent explicitly states that `mname` is the subject identifier and that the exported subject/session indexing should follow the unique recording order after deduplication.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique recordings keyed by `<mouse>_<date>_<blk>`, not by experiment-group entries in `Imaging_Exp_info.npy`. If the same recording appears under multiple experiment groups, the script chooses one canonical behavior entry after checking that the key trial-level fields agree.

ii. ```python
for exp_type, dbs in exp_info.items():
    for db in dbs:
        rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        per_rec[rec_id].append((exp_type, db))
```

```python
def choose_canonical_behavior(
    rec_id: str, refs: list[tuple[str, dict]]
) -> tuple[str, str]:
    ...
    key = f"{rec_id}_{db['stimtype']}" if "stimtype" in db else rec_id
    ...
    same = (
        int(beh["ntrials"]) == int(base_beh["ntrials"])
        and np.array_equal(as_str_array(beh["WallName"]), as_str_array(base_beh["WallName"]))
        and np.array_equal(np.asarray(beh["isRew"]).astype(int), np.asarray(base_beh["isRew"]).astype(int))
        and np.allclose(np.asarray(beh["SoundPos"], dtype=float), np.asarray(base_beh["SoundPos"], dtype=float), equal_nan=True)
        and np.allclose(np.asarray(beh["ft_trInd"], dtype=float), np.asarray(base_beh["ft_trInd"], dtype=float), equal_nan=True)
    )
```

iii. The notes call this out as a major consistency decision: the paper says 89 recordings in 19 mice, while the raw experiment index has 142 references. The agent justified deduplication as the only way to match the paper’s session count.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the frame-wise behavior field `ft_trInd`. The script converts finite trial indices to integers and, for each trial number `0..ntrials-1`, gathers the frame indices that belong to that trial and also satisfy the frame filter.

ii. ```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ft_trial_int = np.full(ft_trial.shape, -1, dtype=np.int64)
    finite_trial = np.isfinite(ft_trial)
    ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    masks = []
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        ...
        masks.append(frame_idx)
```

iii. The notes describe the export as frame-aligned rather than position-interpolated, so the agent used the native per-frame trial index field instead of reconstructing trials from interpolated position bins.

## 1-e. How are trials filtered based on quality controls?

i. There is no separate trial-quality-control stage. Instead, every trial is retained if it has at least one frame that is finite in `ft_trInd`, inside the corridor (`ft_CorrSpc`), and during movement (`ft_move > 0`). A trial with zero such frames causes the conversion to fail.

ii. ```python
valid = finite_trial & ft_corr & ft_move
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
    if len(frame_idx) == 0:
        raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

iii. The notes justify this by saying the reference paper/code restrict analyses to running timepoints and do not describe an additional global trial rejection rule. The trajectory does not show a separate trial QC policy beyond this frame-level restriction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` arrays in each session’s `*_neural_data.npy` file. Retinotopy `iarea` is also loaded and used to decide which neurons to keep and how to label their brain regions.

ii. ```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```

```python
spk = np.concatenate(spk_chunks, axis=0)
areas = utils.neu_area_ID(iarea)
```

iii. The notes explicitly say the saved `spks` arrays are treated as the deconvolved activity traces used by the paper, and that no additional fluorescence preprocessing is applied.

## 2-b. How is the `neural` data processed?

i. The script concatenates the `spks` chunks into a full neuron-by-frame matrix, computes a reference-style neuron subset, then slices the retained neurons by per-trial frame masks and stores each trial as `float16`. The subset is the union of mHV familiar-stimulus-selective neurons and aHV reward-prediction neurons.

ii. ```python
spk = np.concatenate(spk_chunks, axis=0)
areas = utils.neu_area_ID(iarea)
...
mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
...
interp_spk = utils.get_interpPos_spk(
    spk[ahv_idx][:, move_idx],
    poscum_move,
    int(beh["ntrials"]),
    n_bins=60,
    lengths=float(beh["Corridor_Length"]),
)
...
keep_mask = mhv_mask | ahv_mask
```

```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        if len(local_rows) == 0:
            continue
        trial_chunks.append(chunk[local_rows][:, frame_idx])
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The notes say the initial plan to export all visual-area neurons was revised because `train_decoder.py` concatenates all session data in memory. The agent explicitly calls the subset choice a compromise to keep the decoder tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are not filtered by a generic recording-quality threshold. They are filtered by task-specific selectivity rules: mHV neurons must be corridor-responsive and in the top/bottom 5% of familiar-stimulus `d'`; aHV neurons must have reward-prediction `d' >= 0.3` and nonnegative stimulus `d'`. If too few neurons survive, fallback rules add the strongest mHV neurons.

ii. ```python
stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
    spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)
)

mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
```

```python
reward_dp = dprime(mean_corr[:, late], mean_corr[:, early])
local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)
ahv_mask[ahv_idx[local_keep]] = True
...
if int(np.sum(keep_mask)) == 0:
    mhv_candidates = np.where(areas["mHV"])[0]
    order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
    take = mhv_candidates[order[: min(128, len(order))]]
    keep_mask[take] = True
```

iii. The notes describe this as a “reference-style neuron subset” borrowed from `Get_coding_direction` and `Get_dprime_rewPred_neuron`, and explicitly distinguish it from the original plan to keep all visual-area neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The exported trial lists are declared to be aligned to corridor entry (trial start), but the actual neural matrices begin at the first retained running-corridor frame within each trial. The link to trial start is preserved through the use of `ft_trInd` and the paired `time_since_trial_start` input.

ii. ```python
"metadata": {
    ...
    "temporal_alignment_event": "corridor entry (trial start)",
    "off_start": 0.0,
    "off_end": None,
    "trial_filter": "Retain only running corridor frames: ft_CorrSpc & (ft_move > 0).",
}
```

```python
trial_masks = compute_trial_masks(beh)
...
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes argue that the requested decoder outputs are frame-resolved and should stay frame-aligned, but they also acknowledge that only running corridor frames are retained. That means the event label is “trial start,” while the exported neural sequence can start later.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging frame interval as the time bin. The script estimates `time_bin_size` from the median difference in the `ft` timestamps and does not apply temporal rebinning before export. The only rebinning in the script is spatial interpolation used internally for neuron selection.

ii. ```python
"time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
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

iii. The notes cite the notebook frame rate of about 3.17 Hz and say the export stays frame-aligned. The agent used position interpolation only as part of the reward-prediction neuron selection logic.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the per-frame timestamps `ft` and the per-trial sound times `SoundTime`.

ii. ```python
ft = np.asarray(beh["ft"], dtype=float)
sound_time = np.asarray(beh["SoundTime"], dtype=float)
...
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. In Step 5 of the notes, the agent maps `time_to_sound_cue` directly to `ft` and `SoundTime`, with an explicit rationale that this preserves true behavioral timing in seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the script subtracts the frame timestamp from that trial’s `SoundTime` and converts days to seconds by multiplying by `86400`.

ii. ```python
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes say the value should be positive before the cue, near zero at cue onset, and negative after it. No additional smoothing or binarization is applied.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned frame-by-frame using the same `frame_idx` mask used to slice the neural matrix for that trial.

ii. ```python
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
    current_ft = ft[frame_idx]
    time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The notes repeatedly describe the export as frame-aligned, and this variable is computed directly on the retained neural frames rather than on an independent time grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session metadata in `Imaging_Exp_info.npy`: the subject name and the session date string `datexp`.

ii. ```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    by_subject: dict[str, list[SessionRef]] = defaultdict(list)
    for sess in sessions:
        by_subject[sess.subject].append(sess)
    ...
    day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

iii. The notes say this variable has no direct reference-code function and was derived from session chronology to satisfy the decoder specification.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the script finds the earliest session date among the exported recordings, computes each session’s calendar-day offset from that date plus 1, and repeats that scalar across all retained frames of every trial in the session.

ii. ```python
first_date = min(parse_date(s.date_str) for s in sess_list)
for sess in sess_list:
    day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```

```python
day_value = np.float32(day_map[sess.subject][sess.rec_id])
day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. The notes justify this as a continuous per-trial covariate required by the task. The trajectory does not show any attempt to recover non-imaging training days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the per-frame timestamps `ft` and the per-trial `Trial_start_time`.

ii. ```python
ft = np.asarray(beh["ft"], dtype=float)
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
...
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The Step 5 notes map this variable directly to `ft` and `Trial_start_time`, again with the stated goal of preserving true behavioral timing.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame in a trial, the script subtracts the trial start timestamp from the frame timestamp and converts the result from days to seconds.

ii. ```python
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes explicitly say this uses “true behavioral time, not frame count alone.”

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on exactly the same retained frame indices used for the neural trial matrix, so the time vector and neural columns correspond one-to-one.

ii. ```python
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
    current_ft = ft[frame_idx]
    time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The notes say the export is frame-aligned and use this variable as the explicit bridge back to corridor-entry timing after filtering out non-running frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level boolean/0-1 field `isRew`.

ii. ```python
is_rew = np.asarray(beh["isRew"], dtype=float)
...
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The notes justify this by saying pooled sessions include rewarded, unrewarded, and test conditions, so reward availability must come from the raw per-trial annotation rather than a task-wide assumption.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script treats it as a per-trial constant and broadcasts that scalar across all retained frames of the trial.

ii. ```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
trial_input = np.vstack(
    [time_to_cue, day_of_training, time_since_start, reward_available]
).astype(np.float32, copy=False)
```

iii. The notes explicitly say per-trial constants should be repeated across frames so each input trial has a uniform `(n_input, n_timepoints)` shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level texture label `WallName`, not from `stim_id`.

ii. ```python
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
wall_name = as_str_array(beh["WallName"])
stim_code = np.full(
    len(frame_idx),
    category_to_idx[str(wall_name[trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say `WallName` is the reliable stimulus label because duplicate `swap1`/`swap2` references can differ in `stim_id` bookkeeping while agreeing on the actual wall names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script collects all unique `WallName` strings across sessions, sorts them, maps them to integer class IDs, and repeats the trial’s category ID across all retained frames in that trial.

ii. ```python
categories = sorted({...})
category_to_idx = {name: idx for idx, name in enumerate(categories)}
```

```python
stim_code = np.full(
    len(frame_idx),
    category_to_idx[str(wall_name[trial_idx])],
    dtype=np.int16,
)
```

iii. The notes say this preserves the natural stimulus labels requested by the decoder task, such as `circle1`, `leaf2`, and swap variants, instead of collapsing them into the broader categories used by `get_cat_id`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from lick frame annotations `LickFr` and their associated trial indices `LickTrind`.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
...
lick_trial_frames = lick_fr[lick_tr == trial_idx]
```

iii. The notes map licking to these raw frame-level lick annotations because the export is frame-aligned and the decoder task asks for a binary time-varying lick output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script casts lick-frame values to integers, selects the lick frames for the current trial, and marks each retained neural frame as `1` if its frame index appears in that lick list and `0` otherwise.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
...
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The notes describe this as a frame-aligned binary lick raster. The trajectory does not show any more elaborate lick-time interpolation; the decision was to stay on the imaging-frame grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by testing lick events against the same retained `frame_idx` array used for the neural trial matrix.

ii. ```python
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
    trial_output = np.vstack([stim_code, licking, pos_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. The notes say the exported outputs should be time-varying “if at all possible,” and this is the agent’s direct frame-matched alignment choice.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-wise position trace `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
...
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. The notes state that the raw position units are decimeters, with the textured corridor occupying `0..40` and gray space `40..60`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script takes `ft_Pos` on the retained frames, clips it to the textured corridor range, and converts it to discrete bins with floor division by 10 decimeters.

ii. ```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes justify clipping because the export excludes gray-space frames and the decoder task asks specifically for 4 equal 1 m bins within the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four bins: `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)` in raw decimeter units, exported with labels `0-1m`, `1-2m`, `2-3m`, and `3-4m`.

ii. ```python
"output_values": [
    categories,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ["q1", "q2", "q3", "q4"],
],
```

```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The Step 5 notes state the binning explicitly and tie it to the paper-consistent 4 m corridor representation.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned frame-by-frame by sampling `ft_Pos` on the same retained `frame_idx` used for the neural data.

ii. ```python
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
    pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes say position-derived outputs are computed from the frame-aligned position trace rather than from a separate interpolated representation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-wise running-speed trace `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
...
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The notes explicitly map running speed to the raw frame-aligned behavior field `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script makes a first pass through behavior to gather all retained running-speed samples across sessions, computes global quartile edges, and then digitizes each trial’s retained `ft_RunSpeed` values into those bins.

ii. ```python
for sess in sessions:
    beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
    trial_masks = compute_trial_masks(beh)
    speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])

speed_values = np.concatenate(speed_values).astype(np.float32)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```

```python
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The notes justify this directly from the decoder task requirement that the four speed bins each contain about 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by the 25th, 50th, and 75th percentiles of all retained `ft_RunSpeed` samples, giving four quartile bins labeled `q1` through `q4`.

ii. ```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
...
"output_values": [
    categories,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ["q1", "q2", "q3", "q4"],
],
```

iii. The notes say the quartiles are computed after all filtering decisions are applied so the occupancy is near 25% by construction.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned frame-by-frame by taking `ft_RunSpeed` on the same retained frame indices used to slice the neural matrix.

ii. ```python
for trial_idx, frame_idx in enumerate(trial_masks):
    ...
    speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. The notes describe this as another frame-aligned output derived on the native imaging time grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses several ad hoc guards rather than a unified missing-data policy. It ignores NaN trial indices, checks duplicate behavior references for agreement with `equal_nan=True`, converts zero-variance `d'` cases to zero via `nan_to_num`, and falls back to strongest mHV neurons if selection would otherwise be empty. But if a recording lacks a usable behavior entry or a trial has no retained frames, the script raises an exception instead of repairing or skipping the case.

ii. ```python
denom[denom == 0] = np.nan
out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
```

```python
finite_trial = np.isfinite(ft_trial)
ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
...
if not candidates:
    raise KeyError(f"No behavior entry found for recording {rec_id}")
...
if len(frame_idx) == 0:
    raise ValueError(f"Trial {trial} has no retained running corridor frames")
```

```python
if int(np.sum(keep_mask)) == 0:
    mhv_candidates = np.where(areas["mHV"])[0]
    order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
    take = mhv_candidates[order[: min(128, len(order))]]
    keep_mask[take] = True
```

iii. The notes mention NaN-aware duplicate checks and fallback neuron selection, but they do not describe any policy for imputing or skipping malformed trials. The trajectory shows the agent preferred hard failures for missing structural pieces.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are full-session neural loading, concatenating all `spks` chunks, computing the neuron subset, especially the aHV `get_interpPos_spk()` interpolation, and then slicing the retained neurons trial by trial. The logs show rewarded sessions with the reward-prediction branch are the slowest.

ii. ```python
spk = np.concatenate(spk_chunks, axis=0)
...
interp_spk = utils.get_interpPos_spk(
    spk[ahv_idx][:, move_idx],
    poscum_move,
    int(beh["ntrials"]),
    n_bins=60,
    lengths=float(beh["Corridor_Length"]),
)
```

```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        ...
        trial_chunks.append(chunk[local_rows][:, frame_idx])
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The Step 6 and Step 7 notes explicitly identify reward-prediction interpolation and per-session neural processing as the dominant runtime costs.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `compute_trial_masks()`, the per-session scans over behavior files for categories/speed metadata, and the nested per-trial/per-chunk slicing loop inside `convert_dataset()` are the clearest vectorization targets. The code repeatedly constructs lists of frame indices and repeatedly concatenates chunk slices.

ii. ```python
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
    ...
    masks.append(frame_idx)
```

```python
for sess in sessions:
    beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
    ...
    trial_masks = compute_trial_masks(beh)
    speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
```

```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        ...
        trial_chunks.append(chunk[local_rows][:, frame_idx])
```

iii. The notes mention only one explicit speedup, restricting reward interpolation to aHV neurons, but the implementation still leaves these Python loops in place.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly loads behavior files, repeatedly recomputes trial masks, and repeatedly scans `WallName`/`ft` metadata in separate passes for subject metadata, speed quartiles, category collection, `time_bin_size`, and then the actual conversion. It also loads behavior again inside sample-session selection.

ii. ```python
def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```

```python
for sess in sessions:
    beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
    ...
    trial_masks = compute_trial_masks(beh)
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

iii. The notes say the global speed-bin estimation intentionally uses a first behavior-only pass, but the result is that the same behavior structures are revisited several times before conversion is complete.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several values that are either unused or only used for logging: `category_values` in `build_global_metadata()` is never returned, `stim_pos`/`stim_neg` in `kept_stats` are only printed/plotted, and the expensive d-prime/interpolation machinery is used only to decide which neurons to keep while discarding the rest of the raw recording. The compiled reference interpolation helper also emits progress prints (`10000`, `20000`) that are not used downstream.

ii. ```python
dts = []
speed_values = []
category_values = set()
for sess in sessions:
    ...
    dts.append(np.median(np.diff(ft)) * 86400.0)
    ...
    category_values.update(str(v) for v in np.asarray(beh["WallName"]).tolist())
```

```python
stats = {
    "n_total": int(len(iarea)),
    "n_mhv_selected": int(np.sum(mhv_mask)),
    "n_ahv_selected": int(np.sum(ahv_mask)),
    "n_selected": int(np.sum(keep_mask)),
    "stim_pos": stim_pos,
    "stim_neg": stim_neg,
}
```

```python
keep_mask = mhv_mask | ahv_mask
...
return kept_indices.astype(np.int64), region_idx, stats
```

iii. The notes admit that the script now does extra reference-style neuron-selection work as a tractability compromise. That work is not part of the downstream decoder inputs/outputs themselves; it exists only to reduce the exported neural dimensionality.
