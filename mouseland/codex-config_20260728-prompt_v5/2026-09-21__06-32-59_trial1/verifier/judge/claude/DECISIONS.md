# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates all `Beh_*.npy` files in the behavior directory (excluding three behavior-only files without imaging), collects every behavior key, and groups them by raw session key (mouse_date_block). It also loads `Imaging_Exp_info.npy` to get subject names. For each raw session, it loads spike planes from `spk/<session_id>_neural_data.npy` and retinotopy from `retinotopy/<mouse_date>_trans.npz`. Multiple behavior views of the same raw session are merged into one `SessionSpec`.

ii.
```python
def iter_behavior_files() -> list[Path]:
    files = []
    for path in sorted(BEH_DIR.glob("Beh_*.npy")):
        if path.name in {
            "Beh_no_pretrain.npy",
            "Beh_pretrain_on_grat_image.npy",
            "Beh_pretrain_on_nat_image.npy",
        }:
            continue
        files.append(path)
    return files

def build_session_specs() -> list[SessionSpec]:
    exp_info = load_exp_info()
    ...
    raw_to_views: dict[str, list[BehaviorView]] = defaultdict(list)
    for beh_path in iter_behavior_files():
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for full_key, beh in beh_all.items():
            raw_key = parse_raw_session_key(full_key)
            raw_to_views[raw_key].append(BehaviorView(...))
```

```python
planes = load_spike_planes(spec.raw_key)
iarea = load_retinotopy(spec.raw_key)
```

iii. The AI documented that the same recording appears in multiple behavior files and merged them to avoid duplicating neural data, choosing to iterate behavior files directly rather than through `Imaging_Exp_info.npy`.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from `Imaging_Exp_info.npy` entries via the `mname` field. Subjects are the sorted unique mouse names across all selected sessions.

ii.
```python
raw_to_subject: dict[str, str] = {}
for records in exp_info.values():
    for rec in records:
        raw_key = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
        raw_to_subject[raw_key] = rec["mname"]
...
subjects = sorted({spec.subject for spec in selected_specs})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
```

iii. The AI noted that 19 unique mice are present, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is a unique raw imaging recording, identified by `mouse_date_block`. The AI parses this from behavior keys by taking the first 5 underscore-separated parts. Multiple behavior views of the same raw session are merged rather than duplicated.

ii.
```python
def parse_raw_session_key(full_key: str) -> str:
    parts = full_key.split("_")
    return "_".join(parts[:5])
```

iii. The AI documented that 89 unique raw sessions exist, matching the paper's "89 recordings."

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` (trial index per frame), `ft_CorrSpc` (corridor space flag), AND `ft_move > 0` (running flag). Only frames where all three conditions are true are included in a trial. Trials with zero qualifying frames are dropped.

ii.
```python
def get_trial_frame_indices(beh, nframes):
    ft_trind = np.asarray(beh["ft_trInd"][:nframes])
    ft_corr = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
    ft_move = np.asarray(beh["ft_move"][:nframes]) > 0
    ntrials = int(beh["ntrials"])
    frame_sets = []
    for trial in range(ntrials):
        idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
        frame_sets.append(idx)
    return frame_sets
```

iii. The AI justified this by noting the reference code's `fr_valid = VRmove & isCorridor` pattern in `utils.py`, and that the paper says only running timepoints were considered for analyses.

## 1-e. How are trials filtered based on quality controls?

i. The only filter is that trials with zero valid frames (after the running+corridor mask) are dropped. Sessions with fewer than 2 valid trials raise an error. No trial-length-based filtering is applied.

ii.
```python
for trial, frame_idx in enumerate(trial_frame_sets):
    if frame_idx.size == 0:
        continue
    ...
if len(neural_trials) < 2:
    raise ValueError(f"Session {spec.raw_key} has fewer than 2 valid trials after processing")
```

iii. The AI did not mention any trial length filtering in CONVERSION_NOTES.md. Because the `ft_move > 0` mask already excludes stationary frames, long-sitting trials would contribute fewer frames but are not explicitly dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike file (a list of per-plane arrays), concatenated per-trial by slicing each plane at the selected frame indices. The visual area comes from `iarea` in the retinotopy file.

ii.
```python
def load_spike_planes(raw_key):
    path = SPK_DIR / f"{raw_key}_neural_data.npy"
    obj = np.load(path, allow_pickle=True).item()
    return obj["spks"]

neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16)
```

iii. The AI noted that Suite2p deconvolved traces are used directly, matching the reference.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed beyond slicing to the selected frames and casting to float16. Unlike the reference solution which concatenates all planes into one matrix first, the AI concatenates per-trial slices from each plane to avoid a full-session copy.

ii.
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16, copy=False)
```

iii. The AI justified float16 storage as a space optimization since the decoder converts to float32 internally.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ALL neurons are retained, including those outside the four named visual areas (V1, mHV, lHV, aHV). Neurons with area codes not in those groups are assigned to an "other" region rather than being dropped.

ii.
```python
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "other"]

def build_brain_region_idx(iarea):
    idx = np.full(iarea.shape, 4, dtype=np.int8)  # default to 'other'
    idx[iarea == 8] = 0   # V1
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
    idx[np.isin(iarea, [5, 6])] = 2  # lHV
    idx[np.isin(iarea, [3, 4])] = 3  # aHV
    return idx
```

iii. The AI stated: "The public reference code does not apply additional global neuron QC beyond Suite2p preprocessing; region labels are metadata, not a filter." And: "Preserve neurons with codes -1 or 7 as other rather than dropping them."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry by selecting frames where `ft_trInd == trial` and `ft_CorrSpc` is true (and `ft_move > 0`). The first selected frame is the first running corridor frame of the trial. Variable-length trials are preserved.

ii.
```python
idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
```

iii. The AI noted that `StartFr` precedes the first usable corridor frame by 1-4 frames and chose to use the mask-based approach instead.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate of 3.17 Hz is used, giving ~315.46 ms per bin.

ii.
```python
NOMINAL_FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / NOMINAL_FRAME_RATE_HZ
```

iii. The AI noted that the imaging frame is the finest resolution available.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos` (the corridor position of the sound cue per trial) and `ft_Pos` (corridor position at each frame), combined with the fixed virtual corridor speed of 60 cm/s (6 dm/s).

ii.
```python
sound_pos = np.asarray(beh["SoundPos"][: int(beh["ntrials"])], dtype=np.float32)
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. The AI justified using position-derived time rather than frame timestamps because excluding non-running frames with wall-clock timestamps produced unrealistic ranges.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to cue is computed as `(SoundPos - ft_Pos[frame]) / corridor_speed`, where `corridor_speed = 6.0 dm/s`. This converts spatial distance to the cue into a time estimate using the fixed virtual speed. Positive values mean the cue is ahead, negative means past.

ii.
```python
CORRIDOR_SPEED_DM_PER_S = 6.0
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. The AI noted this was corrected after the initial approach using wall-clock timestamps produced inflated ranges due to excluded pause periods.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from `ft_Pos` at the same frame indices used for the neural data, so it is automatically aligned.

ii.
```python
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. All data streams use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date parsed from the raw session key and the mouse's first imaging date.

ii.
```python
def parse_date_and_block(raw_key):
    parts = raw_key.split("_")
    date = dt.date(int(parts[1]), int(parts[2]), int(parts[3]))
    block = int(parts[4])
    return date, block

subject_first_date = {
    subject: min(date for date, _ in date_blocks)
    for subject, date_blocks in subject_dates.items()
}
training_day = float((date - subject_first_date[subject]).days) + 0.01 * (block - 1)
```

iii. The AI stated this "respects within-mouse chronology and preserves the intended 'training day' context."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The training day is computed as the number of calendar days since the mouse's first imaging date, plus a small block offset (0.01 * (block - 1)) to distinguish multiple blocks on the same date. This is a continuous value, not a session ordinal.

ii.
```python
training_day = float((date - subject_first_date[subject]).days) + 0.01 * (block - 1)
```

iii. The AI documented this as "mouse-relative continuous session day." The range is [0.0, 92.0].

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft_Pos` (corridor position at each frame) and the fixed virtual corridor speed of 60 cm/s.

ii.
```python
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. Same position-based time derivation as for time_to_sound_cue.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `ft_Pos[frame] / corridor_speed`, using the fixed virtual speed of 6.0 dm/s. This gives the time the mouse has been running through the corridor, excluding any pauses.

ii.
```python
CORRIDOR_SPEED_DM_PER_S = 6.0
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. The AI noted this avoids inflated time values from excluded pause periods. The range is [0.0, 6.7] s, matching 4m / 0.6 m/s.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from `ft_Pos` at the same frame indices as the neural data.

ii.
```python
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. All data streams use the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks rewarded trials.

ii.
```python
is_rew = np.asarray(beh["isRew"][: int(beh["ntrials"])], dtype=np.int8)
reward_trial = np.full(frame_idx.size, float(is_rew[trial]), dtype=np.float16)
```

iii. Straightforward binary per-trial variable.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float and broadcast across all frames of the trial. No additional processing.

ii.
```python
reward_trial = np.full(frame_idx.size, float(is_rew[trial]), dtype=np.float16)
```

iii. N/A.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (per-trial wall texture name), mapped through a session-specific label map built by merging `stim_id` and `UniqWalls` across all behavior views of that session.

ii.
```python
def build_wall_to_stimulus_map(raw_key, views):
    label_map = {}
    for view in views:
        uniq_walls = list(map(str, np.asarray(view.beh["UniqWalls"])))
        stim_ids = np.asarray(view.beh["stim_id"])
        for wall, stim_id in zip(uniq_walls, stim_ids):
            if np.isnan(stim_id):
                continue
            canonical = CANONICAL_STIM_BY_ID[int(stim_id)]
            label_map[wall] = canonical
    ...
```

iii. The AI built a complex mapping system to resolve wall names to canonical stimulus identities.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses 8 fine-grained stimulus categories: `circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2`. Each trial's `WallName` is mapped to one of these via the session-specific label map. The value is per-trial, broadcast across all frames.

ii.
```python
OUTPUT_STIMULI = [
    "circle1", "circle2", "circle3",
    "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
...
literal_wall = str(wall_name[trial])
canonical_wall = spec.wall_to_stimulus[literal_wall]
stimulus_idx = OUTPUT_STIM_TO_IDX[canonical_wall]
```

iii. The AI documented: "Visual stimulus output will use session-specific canonical categories" and "circle3 will be preserved as its own category." The AI chose to preserve sub-category distinctions rather than grouping into broad texture types.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame indices of lick events) and `LickTrind` (trial index of each lick).

ii.
```python
lick_frames = np.asarray(beh["LickFr"], dtype=int)
lick_trials = np.asarray(beh["LickTrind"], dtype=int)
```

iii. The AI used both LickFr and LickTrind to assign licks to specific trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks are selected where `LickTrind == trial`, then intersected with the trial's frame indices. A binary vector is produced: 1 if a lick frame falls on that frame, 0 otherwise.

ii.
```python
def binarize_licks(lick_frames, trial_lick_mask, frame_idx):
    out = np.zeros(frame_idx.size, dtype=np.uint8)
    frames = np.asarray(lick_frames[trial_lick_mask], dtype=np.int64)
    kept = np.intersect1d(frames, frame_idx, assume_unique=False)
    offsets = np.searchsorted(frame_idx, kept)
    valid = (offsets >= 0) & (offsets < frame_idx.size) & (frame_idx[offsets] == kept)
    if np.any(valid):
        out[offsets[valid]] = 1
    return out
```

iii. The AI filters licks by trial index first, then by frame intersection, which is more complex than needed but should produce correct results.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary vector is indexed at the same frame indices as the neural data via the intersection operation.

ii.
```python
lick_trial_mask = lick_trials == trial
licking = binarize_licks(lick_frames, lick_trial_mask, frame_idx)
```

iii. All streams use the same frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the corridor position at each imaging frame, in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
```

iii. Directly from the behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (floor division) and clipped to [0, 3], giving four 1-meter bins.

ii.
```python
def position_to_bins(ft_pos_dm):
    pos_dm = np.asarray(ft_pos_dm, dtype=np.float32)
    bins = np.floor(pos_dm / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. Same logic as the reference: 4 equal 1-meter spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 dm and clipping to [0,3] produces bins: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
bins = np.floor(pos_dm / 10.0).astype(np.int16)
return np.clip(bins, 0, 3)
```

iii. Matches the task specification for 4 equal-length 1-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed at the same frame indices as the neural data.

ii.
```python
pos_bins = position_to_bins(ft_pos[frame_idx])
```

iii. All streams use the same frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float32)
speed_trial = ft_speed[frame_idx].astype(np.float32)
```

iii. Directly from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed values are collected across ALL sessions and then binned into global quartiles using `np.quantile` at [0.25, 0.5, 0.75]. The `np.digitize` function assigns each frame to one of 4 bins.

ii.
```python
def apply_speed_bins(processed_sessions):
    all_speeds = np.concatenate(
        [speed for session in processed_sessions for speed in session["speed_trials"]]
    )
    quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
    for session in processed_sessions:
        for trial_idx, speed_trial in enumerate(session["speed_trials"]):
            bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
            session["output"][trial_idx][3] = bins
    return quantiles
```

iii. The AI chose global quartiles over per-session quartiles, stating this "gives consistent output classes across sessions, which is better suited to a cross-session decoder."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all included frames across all sessions. `np.digitize` assigns bins based on these edges: bin 0 = lowest 25%, bin 3 = highest 25%.

ii.
```python
quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
bins = np.digitize(speed_trial, quantiles[1:-1], right=False)
```

iii. The speed bin distribution in the output shows [0.250, 0.250, 0.250, 0.250], confirming equal quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is indexed at the same frame indices as the neural data.

ii.
```python
speed_trial = ft_speed[frame_idx].astype(np.float32)
```

iii. All streams use the same frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior arrays are truncated to the number of neural frames (`nframes = planes[0].shape[1]`). Trials with zero qualifying frames are skipped. Sessions with fewer than 2 valid trials raise an error. Lick frames outside the trial's frame set are ignored via the intersection logic.

ii.
```python
nframes = int(planes[0].shape[1])
...
if frame_idx.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The AI documented that behavior can run past imaging and that streams are cut to imaged frames.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files, which total hundreds of GB. The AI noted that the full conversion with 4 workers completed in about 5 minutes.

ii.
```python
planes = load_spike_planes(spec.raw_key)  # reads large .npy file
```

iii. The AI implemented parallel processing (up to 4 workers) for the full run to mitigate I/O bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `get_trial_frame_indices` scans the frame index array once per trial. The lick binarization per trial could also be vectorized. The per-plane per-trial concatenation of neural data is another candidate.

ii.
```python
for trial in range(ntrials):
    idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
```

iii. The AI noted these are negligible compared to I/O costs.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded once in `build_session_specs()` to build the session specs (including stimulus mappings), then the behavior reference is carried forward. However, each session re-reads the spike file independently (no caching across sessions in the same behavior file group unlike the reference).

ii.
```python
# In build_session_specs:
beh_all = np.load(beh_path, allow_pickle=True).item()
# Then in process_session, behavior is accessed via spec.behavior_ref
```

iii. The AI's approach loads behavior files once during setup and neural files once per session.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI retains neurons in the "other" brain region category (585,641 neurons), which the reference solution drops entirely. These neurons are outside the four named visual areas and add significant data volume without clear analytical value. The complex `build_wall_to_stimulus_map` logic using `stim_id` and `UniqWalls` merging is also unnecessary since `WallName` directly names the textures.

ii.
```python
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "other"]
idx = np.full(iarea.shape, 4, dtype=np.int8)  # default to 'other'
```

iii. The AI justified keeping all neurons as: "The public reference code does not apply additional global neuron QC beyond Suite2p preprocessing."
