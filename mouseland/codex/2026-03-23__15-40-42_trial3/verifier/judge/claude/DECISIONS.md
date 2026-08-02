# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources per session: (1) `Imaging_Exp_info.npy` as a master session index containing 23 experiment groups with 142 entries that map to 89 unique recordings, (2) per-session neural data from `data/spk/<mouse>_<date>_<blk>_neural_data.npy` containing `spks` arrays, (3) per-session behavior data from `data/beh/Beh_<exp_type>.npy`, and (4) retinotopy/area data from `data/retinotopy/<mouse>_<date>_trans.npz`. The AI first deduplicates experiment entries to 89 unique recording IDs using `<mouse>_<date>_<blk>` as the session key, then iterates over all sessions to load and process data.

ii.
```python
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
    # ...

# In convert_dataset loop:
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that loading follows the reference code's `load_spk`, `load_exp_beh`, and `load_retino` functions. The deduplication from 142 experiment entries to 89 unique recordings matches the paper's statement of "89 recordings in 19 mice."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field from `Imaging_Exp_info.npy`. A unique list of subjects is built in session order, and each session is assigned a `subject_idx` pointing into this list.

ii.
```python
def build_global_metadata(sessions):
    subjects = []
    subject_to_idx = {}
    for sess in sessions:
        if sess.subject not in subject_to_idx:
            subject_to_idx[sess.subject] = len(subjects)
            subjects.append(sess.subject)
    # ...

data = {
    'subjects': subjects,
    'subject_idx': np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int64),
    # ...
}
```

iii. The AI noted that 19 unique imaging mice were found, matching the paper's "19 mice" count.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique recording IDs `<mouse>_<date>_<blk>`. When a recording appears in multiple experiment groups in `Imaging_Exp_info.npy`, duplicate references are merged and validated to ensure they have consistent behavior data. One canonical behavior entry is chosen per recording.

ii.
```python
@dataclass(frozen=True)
class SessionRef:
    rec_id: str
    subject: str
    date_str: str
    blk: str
    canonical_exp_type: str
    canonical_beh_key: str
    refs: tuple

def collect_sessions(sample: bool) -> list[SessionRef]:
    exp_info = load_exp_info()
    per_rec: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            per_rec[rec_id].append((exp_type, db))
    sessions = []
    for rec_id, refs in per_rec.items():
        first_db = refs[0][1]
        exp_type, beh_key = choose_canonical_behavior(rec_id, refs)
        sessions.append(SessionRef(...))
    sessions.sort(key=lambda s: (s.subject, parse_date(s.date_str), int(s.blk)))
    return sessions
```

iii. The AI documented this deduplication approach in CONVERSION_NOTES.md Steps 2 and 4, noting that the 142 experiment entries reduce to 89 unique recordings matching the paper.

## 1-d. How are the data split into trials?

i. Trials are split using the behavior field `ft_trInd`, which assigns each imaging frame to a trial index. The code iterates over trial indices from 0 to `ntrials-1` and extracts frame indices for each trial, keeping only frames that pass the corridor-running filter.

ii.
```python
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
        if len(frame_idx) == 0:
            raise ValueError(f"Trial {trial} has no retained running corridor frames")
        masks.append(frame_idx)
    return masks
```

iii. The AI noted that trial structure comes from the behavior dictionaries and that all trials (including unrewarded ones) are retained.

## 1-e. How are trials filtered based on quality controls?

i. Trials are NOT filtered out entirely. Instead, within each trial, only frames satisfying `ft_CorrSpc & (ft_move > 0)` (corridor space AND running) are retained. Every trial with at least one valid frame is included. If a trial has zero valid frames, a `ValueError` is raised (i.e., the code assumes all trials have at least one running corridor frame).

ii.
```python
valid = finite_trial & ft_corr & ft_move
masks = []
for trial in range(int(beh["ntrials"])):
    frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
    if len(frame_idx) == 0:
        raise ValueError(f"Trial {trial} has no retained running corridor frames")
    masks.append(frame_idx)
```

iii. The AI justified this in CONVERSION_NOTES.md: "Use only running corridor frames: This matches the paper statement 'We only considered timepoints during running for analysis' and the reference-code masks built from `ft_CorrSpc` and `ft_move > 0`."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in `<mouse>_<date>_<blk>_neural_data.npy` files. These are Suite2p deconvolved fluorescence traces. The `spks` field contains a list of neuron-block arrays that are concatenated along axis 0 to form a (neurons × frames) matrix. Area assignments come from the `iarea` field in retinotopy files.

ii.
```python
spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
spk_chunks = list(spk_obj["spks"])
ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
iarea = np.asarray(ret["iarea"])
```

iii. The AI noted in CONVERSION_NOTES.md: "The saved `spks` arrays are the deconvolved activity traces from Suite2p, not raw fluorescence; no additional dF/F computation is needed."

## 2-b. How is the `neural` data processed?

i. The AI applies a complex neuron selection pipeline rather than keeping all neurons:
   1. Computes stimulus selectivity d' between two "familiar" stimuli using only odd-trial running corridor frames.
   2. Identifies "corridor-responsive" neurons (`corr_neu`) that respond more in corridor than gray space.
   3. Selects mHV neurons in the top/bottom 5th percentile of d' among corridor-responsive mHV neurons.
   4. For sessions with rewards, selects aHV neurons using position-interpolated activity and a reward-prediction d' threshold (≥0.3).
   5. Takes the union of mHV and aHV selected neurons.
   6. The selected neurons' raw deconvolved traces (not position-interpolated) are exported for retained frames.
   7. Neural values are stored as float16.

ii.
```python
def compute_selected_neurons(spk_chunks, beh, iarea, session_id):
    spk = np.concatenate(spk_chunks, axis=0)
    areas = utils.neu_area_ID(iarea)
    # ... familiar pair identification ...
    stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
    corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
        spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1))
    mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)
    # ... aHV reward-prediction selection ...
    keep_mask = mhv_mask | ahv_mask
    # ...
```

iii. The AI justified this in CONVERSION_NOTES.md Step 6: "Initial mapping plan was revised during implementation because the provided decoder concatenates all session data in memory. Exporting all visual-area neurons would be intractable. The implemented compromise uses a reference-style neuron subset."

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no separate quality filter beyond the neuron selection described in 2-b. The AI does not apply a global quality threshold (e.g., SNR, firing rate minimum). Instead, the d'-based selection serves as both the neuron curation and quality filter. Only neurons in mHV or aHV brain regions that pass the selectivity criteria are retained. V1 and lHV neurons are entirely excluded.

ii.
```python
keep_mask = mhv_mask | ahv_mask
# Only mHV and aHV regions are selected; V1 and lHV are excluded
region_names[:] = "mHV"
region_names[np.isin(kept_indices, np.where(ahv_mask)[0])] = "aHV"
```

iii. The AI noted: "Neuron curation in the provided code is task-specific rather than a single global quality filter: analyses often require neurons to be in named visual areas and to respond more in corridor than gray space."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). For each trial, only frames belonging to that trial's corridor-running period are extracted. The first retained frame of each trial corresponds to the first running frame after corridor entry. There is no fixed pre-event window; `off_start` is set to 0.0.

ii.
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        if len(local_rows) == 0:
            continue
        trial_chunks.append(chunk[local_rows][:, frame_idx])
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)

# metadata:
'temporal_alignment_event': "corridor entry (trial start)",
'off_start': 0.0,
'off_end': None,
```

iii. The AI documented: "Exported trials are aligned to corridor entry (Trial_start_time), while cue timing is computed from raw SoundTime and frame timestamps ft."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data uses the native imaging frame rate with no rebinning. The time bin size is the median inter-frame interval across sessions, approximately 314.69 ms (~3.17 Hz imaging rate).

ii.
```python
'time_bin_size': float(np.median([
    np.median(np.diff(np.asarray(
        load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float
    ))) * 86400.0
    for s in sessions
]) * 1000.0),
```

iii. The AI noted in CONVERSION_NOTES.md: "Neural data stay on the native imaging frame bins (median 314.69 ms), which is consistent with the reference frame-based deconvolved traces."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from two behavior fields: `SoundTime` (per-trial sound cue time in fractional days) and `ft` (per-frame timestamp in fractional days).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=float)
ft = np.asarray(beh["ft"], dtype=float)
# ...
current_ft = ft[frame_idx]
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The AI documented this mapping in CONVERSION_NOTES.md Step 5: "For each retained frame, compute signed seconds to cue as `(SoundTime[trial] - ft[frame]) * 86400`."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the signed difference `(SoundTime[trial] - ft[frame])` is computed and converted from fractional days to seconds by multiplying by 86400. Positive values mean the frame is before the cue; negative values mean after.

ii.
```python
time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The sign convention (positive = before cue) is documented in CONVERSION_NOTES.md Step 5.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to cue is computed on exactly the same retained frame indices as the neural data (`frame_idx` from `compute_trial_masks`), so there is a 1:1 correspondence between neural time bins and time-to-cue values.

ii.
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    # neural uses frame_idx
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
    # time_to_cue uses the same frame_idx
    current_ft = ft[frame_idx]
    time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. The AI validated alignment in Step 10 sanity checks using `np.allclose()`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session date string `datexp` in `Imaging_Exp_info.npy`, which encodes the recording date in `YYYY_MM_DD` format.

ii.
```python
def subject_day_map(sessions: list[SessionRef]) -> dict[str, dict[str, float]]:
    by_subject: dict[str, list[SessionRef]] = defaultdict(list)
    for sess in sessions:
        by_subject[sess.subject].append(sess)
    day_map: dict[str, dict[str, float]] = defaultdict(dict)
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
        for sess in sess_list:
            day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
    return day_map
```

iii. The AI noted this is a "derived quantity" not directly present in the reference code but required by the decoder task specification.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest recording date is found. Each session's day of training is computed as `(session_date - first_date).days + 1`, so the first session is day 1. This per-session scalar is broadcast across all frames in all trials of that session.

ii.
```python
day_value = np.float32(day_map[sess.subject][sess.rec_id])
# ...
day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. The AI documented: "`day_of_training`; continuous per-trial covariate required by user task."

## 4-a (Environment type). What variables in the raw data is `input` *Environment type* derived from?

i. The AI did NOT include "Environment type" as a separate decoder input. The decoder inputs are: time_to_sound_cue, day_of_training, time_since_trial_start, and reward_available. The visual environment (corridor type) is captured as an output variable (`visual_stimulus_category` from `WallName`), and reward status is captured as `reward_available` from `isRew`.

ii.
```python
'input_names': [
    "time_to_sound_cue",
    "day_of_training",
    "time_since_trial_start",
    "reward_available",
],
```

iii. The AI followed the decoder task specification which lists exactly four inputs without "Environment type."

## 4-b (Environment type). What processing is involved in computing `input` *Environment type*?

i. Not applicable — "Environment type" was not included as a decoder input. The closest analog is `reward_available` (binary 0/1 per trial from `isRew`), which indicates whether the corridor is rewarded, effectively encoding the environment type relevant to the task.

ii. N/A

iii. The AI's decision aligns with the decoder task specification which does not list "Environment type" as an input.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from two behavior fields: `Trial_start_time` (per-trial corridor entry time in fractional days) and `ft` (per-frame timestamp in fractional days).

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
ft = np.asarray(beh["ft"], dtype=float)
# ...
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. The AI documented this in CONVERSION_NOTES.md Step 5.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame in a trial, the elapsed time since corridor entry is computed as `(ft[frame] - Trial_start_time[trial]) * 86400` to convert from fractional days to seconds. Values are always non-negative since frames occur after trial start.

ii.
```python
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Validated in Step 10: "All checked `time_since_trial_start` traces are monotonic within trial."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on exactly the same retained frame indices as neural data, providing 1:1 temporal alignment.

ii.
```python
# Same frame_idx used for neural and inputs
current_ft = ft[frame_idx]
time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Verified via sanity checks in Step 10.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the behavior field `isRew`, a per-trial binary indicator of whether the corridor is rewarded.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=float)
# ...
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. The AI noted: "Reward availability must be carried per trial from raw `isRew`, not assumed from task design" since many sessions are entirely unrewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value (0 or 1) is broadcast to fill all retained frames within that trial. No additional processing is applied.

ii.
```python
reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
trial_input = np.vstack([time_to_cue, day_of_training, time_since_start, reward_available])
```

iii. The AI documented: "Trial-constant 0/1, repeated over frames."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from the behavior field `WallName`, a per-trial string indicating the visual texture shown in that corridor (e.g., "circle1", "leaf1", "rock1", etc.).

ii.
```python
wall_name = as_str_array(beh["WallName"])
# ...
stim_code = np.full(len(frame_idx), category_to_idx[str(wall_name[trial_idx])], dtype=np.int16)
```

iii. The AI chose `WallName` over `stim_id` because `stim_id` contains NaN values for some sessions, while `WallName` is always populated.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` strings across all sessions are collected and sorted alphabetically. Each string is mapped to an integer index. The per-trial stimulus category is broadcast across all retained frames within that trial. This results in 15 unique categories.

ii.
```python
categories = sorted({
    str(v)
    for sess in sessions
    for v in np.asarray(load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]["WallName"]).tolist()
})
category_to_idx = {name: idx for idx, name in enumerate(categories)}
# ...
stim_code = np.full(len(frame_idx), category_to_idx[str(wall_name[trial_idx])], dtype=np.int16)
```

iii. The AI documented that 15 category strings are present across the dataset.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from two behavior fields: `LickFr` (frame indices where licks occurred) and `LickTrind` (trial index for each lick event).

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
```

iii. The AI identified these from the reference code's lick utilities.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are filtered to that trial using `LickTrind`. A binary vector is created over retained frames: 1 if the frame index appears in the trial's `LickFr` entries, 0 otherwise.

ii.
```python
lick_trial_frames = lick_fr[lick_tr == trial_idx]
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. The AI noted: "Binary frame vector: 1 if one or more licks fall in retained frame, else 0."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking binary vector is computed on the same `frame_idx` array used for neural data, ensuring 1:1 temporal alignment. A frame is marked as licking if its raw frame index matches any lick event frame for that trial.

ii.
```python
licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. Verified in Step 10 sanity checks.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from the behavior field `ft_Pos`, a per-frame position value in decimeters within the corridor (0–40 dm for the 4 m texture corridor).

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
# ...
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. The AI noted that raw positions are in decimeter units, with 40 dm corresponding to the 4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are clipped to [0, 39.999] dm to ensure they fall within the texture corridor range, then discretized into 4 bins.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The AI documented: "Discretize corridor position into four 1 m bins over the 4 m texture corridor."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 equal-length 1 m bins using integer division by 10 dm: bin 0 = [0, 10) dm = 0–1 m, bin 1 = [10, 20) dm = 1–2 m, bin 2 = [20, 30) dm = 2–3 m, bin 3 = [30, 40] dm = 3–4 m.

ii.
```python
pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
# output_values: ["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. The AI noted this matches the instruction requirement for "4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read from `ft_Pos` at the same `frame_idx` indices used for neural data, providing direct 1:1 alignment.

ii.
```python
pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. Same frame-based alignment as all other variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from the behavior field `ft_RunSpeed`, a per-frame running speed value.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
# ...
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Documented in CONVERSION_NOTES.md Step 5 variable mapping.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global quartile edges are computed across ALL retained frames from ALL sessions in a first pass over the behavior data. Each frame's speed is then digitized into 4 bins using these global edges.

ii.
```python
def build_global_metadata(sessions):
    # ...
    for sess in sessions:
        beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
        trial_masks = compute_trial_masks(beh)
        speed_values.append(np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)])
    speed_values = np.concatenate(speed_values).astype(np.float32)
    speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
    return subjects, subject_to_idx, day_map, speed_edges
```

iii. The AI documented: "Quartiles should each contain ~25% of retained samples" matching the instruction for "4 bins, each corresponding to 25% of the data."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three quartile edges (25th, 50th, 75th percentiles) are computed globally. `np.digitize` maps each speed value into bins 0–3: bin 0 = below 25th percentile, bin 1 = 25th–50th, bin 2 = 50th–75th, bin 3 = above 75th.

ii.
```python
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. The AI verified that speed bin distribution is approximately [0.250, 0.250, 0.250, 0.250].

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read from `ft_RunSpeed` at the same `frame_idx` indices used for neural data, providing direct 1:1 alignment.

ii.
```python
speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```

iii. Same frame-based alignment as all other variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
   - NaN values in `ft_trInd` are handled by checking `np.isfinite()` before converting to integer indices.
   - NaN values in `stim_id` are handled by using `WallName` instead.
   - If a session has duplicate behavior entries across experiment groups, they are validated for consistency before choosing one.
   - If the mHV d'-selected neuron count is below 128, a fallback selects the top 128 by absolute d'.
   - If no neurons pass the combined mHV/aHV selection, a final fallback takes the strongest mHV neurons.
   - Trials with zero valid frames raise an error (the code does not silently drop trials).

ii.
```python
# NaN handling in trial indices
ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
finite_trial = np.isfinite(ft_trial)
ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)

# Fallback for low neuron count
if int(np.sum(mhv_mask)) < 128:
    mhv_candidates = np.where(corr_neu & areas["mHV"])[0]
    if len(mhv_candidates) > 0:
        order = np.argsort(np.abs(stim_dp[mhv_candidates]))[::-1]
        take = mhv_candidates[order[: min(128, len(order))]]
        mhv_mask = np.zeros_like(mhv_mask)
        mhv_mask[take] = True
```

iii. The AI documented edge-case handling in CONVERSION_NOTES.md Step 10.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
   1. **Neuron selection** (`compute_selected_neurons`): Requires loading the full neural matrix, computing d' across all neurons, and for rewarded sessions, running position interpolation on aHV neurons.
   2. **Per-session neural loading**: Loading large `.npy` files with 20k–90k neurons per session.
   3. **Global metadata computation** (`build_global_metadata`): First pass loading all behavior files and computing trial masks and speed quartiles.

   Full conversion took ~1014 seconds (16.9 minutes) for 89 sessions.

ii.
```python
# Main bottleneck: neuron selection per session
kept_idx, region_idx, kept_stats = compute_selected_neurons(spk_chunks, beh, iarea, sess.rec_id)
# Full neural concatenation for d' computation
spk = np.concatenate(spk_chunks, axis=0)
```

iii. The AI estimated 11–18 s/session in CONVERSION_NOTES.md Step 7, with rewarded sessions being slower due to position interpolation.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop in `convert_dataset` iterates over trials within each session, extracting neural data trial-by-trial. This could potentially be vectorized by using advanced indexing to extract all trials at once, though the variable trial lengths make this challenging. The chunk-based neural extraction (iterating over `spk_chunks` and `chunk_local_rows`) could also be streamlined by pre-concatenating the selected neuron rows.

ii.
```python
for trial_idx, frame_idx in enumerate(trial_masks):
    trial_chunks = []
    for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
        if len(local_rows) == 0:
            continue
        trial_chunks.append(chunk[local_rows][:, frame_idx])
    neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. The AI noted the chunk-based approach was chosen to avoid concatenating the full neural matrix for all neurons, though only selected neurons' data is extracted.

## 12-c. What processing does the code repeat multiple times?

i. Several processing steps are repeated:
   1. **Behavior loading**: `load_beh()` is called multiple times for the same sessions — in `choose_canonical_behavior`, `collect_sessions` (for sample selection), `build_global_metadata`, and the main `convert_dataset` loop.
   2. **Trial mask computation**: `compute_trial_masks()` is called both in `build_global_metadata` (for speed quartiles) and in the main conversion loop.
   3. **`as_str_array` conversions**: Called multiple times on the same fields.

ii.
```python
# In build_global_metadata:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
trial_masks = compute_trial_masks(beh)

# Then again in convert_dataset:
beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
trial_masks = compute_trial_masks(beh)
```

iii. The AI acknowledged some redundancy but prioritized keeping per-session processing self-contained to manage memory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Key unnecessary processing includes:
   1. **Full neural matrix concatenation for neuron selection**: The entire `spk` matrix (all neurons × all frames) is concatenated in `compute_selected_neurons` even though only selected neurons are ultimately exported.
   2. **Position interpolation for aHV selection**: For rewarded sessions, `get_interpPos_spk` is called on aHV neurons to compute reward-prediction d'. This interpolated data is discarded after neuron selection.
   3. **The neuron selection itself**: The d'-based neuron selection is a significant computation that reduces neuron count from ~52k to ~1.2k per session. If a simpler selection (e.g., all visual-area neurons) were used, this entire computation would be unnecessary.
   4. **Duplicate behavior validation** in `choose_canonical_behavior`: Validates consistency across experiment-group duplicates for every session.
   5. **Category collection across all sessions**: A second pass collects all unique WallName values even though this could be done during the main conversion pass.

ii.
```python
# Full neural matrix loaded just for neuron selection d' computation
spk = np.concatenate(spk_chunks, axis=0)
# Position interpolation only used for aHV selection, then discarded
interp_spk = utils.get_interpPos_spk(spk[ahv_idx][:, move_idx], ...)
mean_corr = interp_spk[:, :, 5:40].mean(axis=2)
```

iii. The AI noted the aHV interpolation is restricted to aHV neurons only as a speedup, but the fundamental computation is still discarded after selection.
