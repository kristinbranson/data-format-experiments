# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script loads the behavior index from `beh/Imaging_Exp_info.npy`, loads every behavior file named by that index into a canonical lookup, enumerates sessions from filenames in `spk/`, and then loads one spike file and one retinotopy file per session. Behavior is keyed by `mname_datexp_blk` with an optional `stimtype` suffix for swap sessions.

ii. ```python
def load_canonical_behavior_sessions():
    exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()
    canonical = {}

    for exp_type, records in exp_info.items():
        behavior_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh = np.load(behavior_path, allow_pickle=True).item()
        ...
            full_key = full_key_from_record(record)
            base_key = f"{record['mname']}_{record['datexp']}_{record['blk']}"
            session = beh[full_key]
```
```python
def get_spk_session_keys():
    session_keys = []
    for filename in sorted(os.listdir(SPK_DIR)):
        if not filename.endswith("_neural_data.npy"):
            continue
        session_keys.append(filename[: -len("_neural_data.npy")])
    return session_keys
```
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
retino = np.load(retino_path, allow_pickle=True)
```

iii. In the trajectory, the agent said it would use every imaging session from `Imaging_Exp_info.npy`, honor swap-session `stimtype` suffixes, and align raw `spk` traces to the behavior frame grid.

## 1-b. How are the data split into subjects?

i. Subjects are split by parsing the mouse name from each session key, then taking sorted unique mouse names and building `subject_idx` from them.

ii. ```python
def parse_base_key(base_key):
    match = BASE_KEY_RE.match(base_key)
    ...
    return match.group("mouse"), match.group("date"), match.group("blk")
```
```python
def build_subject_metadata(session_keys):
    subjects = sorted({parse_base_key(key)[0] for key in session_keys})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    subject_idx = np.array([subject_to_idx[parse_base_key(key)[0]] for key in session_keys], dtype=np.int64)
    return subjects, subject_idx
```

iii. The trajectory does not give extra justification beyond using the session key schema already present in the raw files.

## 1-c. How are the data split into sessions?

i. Sessions are defined by spike filenames of the form `mouse_date_block_neural_data.npy`. Behavior metadata for duplicate experiment-type listings is deduplicated to one canonical `base_key`, but the actual converted session list comes from the spike directory.

ii. ```python
base_key = f"{record['mname']}_{record['datexp']}_{record['blk']}"
...
if base_key not in canonical:
    canonical[base_key] = {
        "behavior": session,
        "exp_type": exp_type,
        "full_key": full_key,
        "record": dict(record),
    }
```
```python
session_keys = get_spk_session_keys()
```

iii. In the trajectory, the agent explicitly noted that swap sessions need the `stimtype` suffix in behavior keys, but the underlying imaging recording is still the base `mouse_date_block` session.

## 1-d. How are the data split into trials?

i. Trials are split by `ft_trInd`, but only frames inside the corridor and with positive movement are kept. This means the script uses running corridor frames rather than all corridor frames in a trial.

ii. ```python
ft_trind = np.asarray(behavior["ft_trInd"][:n_frames])
ft_corr = np.asarray(behavior["ft_CorrSpc"][:n_frames], dtype=bool)
ft_move = np.asarray(behavior["ft_move"][:n_frames], dtype=np.float32)
retained_mask = ft_corr & (ft_move > 0)
```
```python
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
    if len(frame_idx) == 0:
        continue
```

iii. The agent justified this in the trajectory as “paper-matched” retention of running frames inside the corridor and said this removes stationary reward-collection periods.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has no retained frames after the running-corridor mask. A whole session is dropped if fewer than two trials survive. The script does not apply the reference long-trial outlier filter.

ii. ```python
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
    if len(frame_idx) == 0:
        continue
```
```python
if len(neural_trials) < 2:
    return None
```

iii. In the trajectory, the only explicit trial-quality rationale was to keep running corridor frames and to skip sessions with fewer than two valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spks` in the session spike file and retinotopy labels `iarea` in the date-matched retinotopy file.

ii. ```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
```
```python
retino = np.load(retino_path, allow_pickle=True)
return build_region_index(retino["iarea"])
```

iii. The trajectory states the converter should use deconvolved traces from the raw `spk` files and restrict to the retinotopy groups used in the paper.

## 2-b. How is the `neural` data processed?

i. The script scores neurons within the four visual regions, keeps at most 128 neurons per region based on corridor responsiveness and variance, concatenates the selected neurons across planes, slices only retained running-corridor frames, and stores each trial as `float16`.

ii. ```python
selected_global = select_visual_neurons(
    spk_planes=spk_planes,
    region_idx_full=region_idx_full,
    corridor_mask=retained_mask,
    gray_mask=gray_running_mask,
    max_per_region=max_neurons_per_region,
)
session_matrix = build_selected_session_matrix(spk_planes, selected_global)
```
```python
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```

iii. The trajectory says the agent added this curation for tractability because the raw spike directory is about 405 GB, and described the retained neurons as deterministic high-variance corridor-responsive visual-cortex neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, neurons outside retinotopically assigned `V1`, `mHV`, `lHV`, and `aHV` are removed. Then, within each region, the script prefers neurons with higher mean activity in corridor than gray space and selects the highest-variance neurons up to a fixed cap of 128 per region.

ii. ```python
def build_region_index(iarea):
    region_idx = np.full(len(iarea), -1, dtype=np.int16)
    region_idx[iarea == 8] = 0
    region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    region_idx[np.isin(iarea, [5, 6])] = 2
    region_idx[np.isin(iarea, [3, 4])] = 3
    return region_idx
```
```python
if gray_mask.any():
    gray_mean = plane_valid[:, gray_mask].mean(axis=1)
    responsive = corridor_mean > gray_mean
...
region_selected = sort_take_desc(resp_scores, resp_indices, max_per_region)
```

iii. The trajectory says the agent wanted a defensible curation step so the converted artifact would stay tractable, and explicitly described the neuron selection as restricted to visual cortex with a per-region cap.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script says the alignment event is trial start / corridor entry, but the actual per-trial neural arrays begin at the first retained running corridor frame, not necessarily the first corridor-entry frame.

ii. ```python
"temporal_alignment_event": "Trial start / corridor entry, with paper-matched retention of running frames inside the corridor only.",
```
```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```

iii. The trajectory justification was that trials should be built from corridor entry onward while retaining only running corridor frames on the shared imaging frame grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied; each kept neural column is one retained imaging frame. However, the metadata `time_bin_size` is written from a constant `TIME_BIN_MS = 1000 * 24 * 3600 * 0.314693525...`, which numerically equals about `27,189,520.6` ms rather than about `315` ms.

ii. ```python
TIME_BIN_MS = 1000.0 * 24.0 * 3600.0 * 0.31469352543354034
...
"time_bin_size": float(TIME_BIN_MS),
```

iii. In the trajectory, the agent said the behavior arrays were already frame-aligned to imaging at about `0.315 s` per sample, so the code appears intended to preserve frame resolution without rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` and the per-frame timestamps `ft`.

ii. ```python
ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)
sound_time = np.asarray(behavior["SoundTime"], dtype=np.float64)
```

iii. The trajectory says the agent inspected cue timing fields in the behavior arrays and used the already frame-aligned timing information.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the script computes `(sound_time[trial] - ft_time[frame_idx]) * 24 * 3600`, so the variable is positive before the cue and negative after it.

ii. ```python
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The trajectory indicates the agent wanted a time-varying decoder input on the same frame grid as the neural data rather than a binary cue-onset series.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by evaluating the cue-relative time on the same retained `frame_idx` used to slice the neural data for that trial.

ii. ```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The trajectory repeatedly emphasizes that all inputs and outputs should be built on the same retained imaging-frame grid as the neural arrays.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session key, specifically mouse identity and session date parsed from `mouse_YYYY_MM_DD_block`.

ii. ```python
def date_from_base_key(base_key):
    _, date_str, _ = parse_base_key(base_key)
    return datetime.strptime(date_str, "%Y_%m_%d").date()
```
```python
mouse, _, _ = parse_base_key(session_key)
```

iii. The trajectory notes that the agent checked training-day ordering from the session metadata before locking in the rule.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the script finds the first session date and computes calendar days since that first date. The resulting scalar is broadcast across all retained frames of every trial in that session.

ii. ```python
first_day[mouse] = min(date_from_base_key(key) for key in keys)
...
training_day[session_key] = float((date_from_base_key(session_key) - first_day[mouse]).days)
```
```python
training_day = np.full(len(frame_idx), day_of_training, dtype=np.float32)
```

iii. The trajectory does not give a deep justification beyond checking session ordering; the metadata later describes the rule explicitly as “Calendar days since the first imaging session for that mouse.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` and per-frame timestamps `ft`.

ii. ```python
ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)
trial_start_time = np.asarray(behavior["Trial_start_time"], dtype=np.float64)
```

iii. The trajectory shows the agent inspected start-frame and trial-start timing fields and chose to use the explicit per-trial time field.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the script computes `(ft_time[frame_idx] - trial_start_time[trial]) * 24 * 3600` and stores it as a float32 time-varying input.

ii. ```python
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The trajectory says the input/output streams should be derived on the same retained frame grid from corridor entry onward.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by computing the time difference on exactly the same retained `frame_idx` that is used to slice the neural matrix.

ii. ```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The trajectory justification is the same shared-frame-grid alignment used for the other per-frame variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew`.

ii. ```python
is_rew = np.asarray(behavior["isRew"], dtype=bool)
```

iii. The trajectory treats rewarded versus unrewarded corridors as already present in the behavior arrays.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` flag is cast to float and broadcast across all retained frames of the trial.

ii. ```python
reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)
```

iii. There is no additional explicit justification in the trajectory beyond putting every decoder variable onto the same time base.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. ```python
wall_names = np.asarray(behavior["WallName"])
stim_category = STIM_CATEGORY_TO_IDX[canonical_stimulus_category(wall_names[trial])]
```

iii. The trajectory says the converter collapses wall identities to the broad stimulus categories `circle`, `leaf`, `rock`, and `wood`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script strips swap suffixes and trailing digits from wall names, maps the result to one of four canonical categories, converts that category to an integer code, and broadcasts the code across all retained frames of the trial.

ii. ```python
def canonical_stimulus_category(wall_name):
    category = re.sub(r"_swap[12]$", "", str(wall_name))
    category = re.sub(r"\d+$", "", category)
    ...
    return category
```
```python
stim_category = STIM_CATEGORY_TO_IDX[canonical_stimulus_category(wall_names[trial])]
stim_out = np.full(len(frame_idx), stim_category, dtype=np.int8)
```

iii. The trajectory explicitly says the converter “collapses wall identities to broad stimulus categories.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, `LickTrind`, `LickPos`, and the retained trial frame indices.

ii. ```python
lick_trial_index = np.asarray(behavior["LickTrind"], dtype=np.float64)
lick_frame_idx = np.asarray(behavior["LickFr"], dtype=np.float64)
lick_pos = np.asarray(behavior["LickPos"], dtype=np.float64)
```

iii. The trajectory shows the agent examined lick frame, lick position, and lick trial-index fields before choosing the alignment rule.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script keeps licks whose positions fall within the textured corridor, finds the nearest retained frame for each lick, requires the lick to be within half a frame of that retained frame, and sets those retained frames to 1 in a binary lick vector.

ii. ```python
valid = np.isfinite(lick_frame_idx) & np.isfinite(lick_pos) & (lick_pos >= 0.0) & (lick_pos < texture_length)
...
insert = np.searchsorted(retained, lick_frame_idx)
...
lick_binary[np.unique(nearest[nearest_dist <= 0.5])] = 1
```

iii. The trajectory does not spell out this exact algorithm, but it repeatedly says behavioral outputs should live on the same retained running-corridor frame grid as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned by assigning them to the nearest retained trial frame in `frame_idx`, not by simply truncating `LickFr` to the original imaging frame grid.

ii. ```python
licking = nearest_retained_licks(
    retained_frame_idx=frame_idx,
    lick_frame_idx=lick_frame_idx[trial_lick_mask],
    lick_pos=lick_pos[trial_lick_mask],
    texture_length=texture_length,
)
```

iii. The trajectory’s general justification is to put all outputs on the same retained frame grid as the neural arrays after the running/corridor masking step.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from per-frame position `ft_Pos`.

ii. ```python
ft_pos = np.asarray(behavior["ft_Pos"][:n_frames], dtype=np.float32)
```

iii. The trajectory says the outputs are built from the behavior arrays already aligned to imaging frames.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script clips each retained position to the corridor length, divides the corridor into four equal-length bins using `np.linspace`, digitizes the positions into those bins, and stores the result as int8.

ii. ```python
def make_position_bins(position, texture_length):
    pos = np.clip(np.asarray(position, dtype=np.float32), 0.0, float(texture_length) - 1e-6)
    bin_edges = np.linspace(0.0, float(texture_length), 5)
    return np.clip(np.digitize(pos, bin_edges[1:-1], right=False), 0, 3).astype(np.int8)
```

iii. The trajectory says the converter produces four position bins and keeps all decoder variables on the retained frame grid.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholding uses four equal-width spatial bins spanning `0` to `texture_length` (default `40.0`), with bin edges at one-quarter, one-half, and three-quarters of the corridor.

ii. ```python
texture_length = float(behavior.get("Texture_Length", 40.0))
...
bin_edges = np.linspace(0.0, float(texture_length), 5)
return np.clip(np.digitize(pos, bin_edges[1:-1], right=False), 0, 3).astype(np.int8)
```

iii. The trajectory does not discuss this thresholding in detail, beyond describing the output as four position bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from `ft_Pos` on the same retained `frame_idx` used for the neural data.

ii. ```python
position_out = make_position_bins(ft_pos[frame_idx], texture_length)
```

iii. The trajectory justification is the general retained-frame-grid alignment applied to every time-varying variable.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from per-frame running speed `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(behavior["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. The trajectory refers to running speed as one of the behavioral outputs derived directly from the behavior arrays.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script pools speeds from all sessions, but only from frames with `ft_CorrSpc` and `ft_move > 0`, computes global 25th/50th/75th percentile cutpoints, then bins each retained per-trial speed by those global thresholds.

ii. ```python
def compute_speed_edges(session_keys, canonical_behavior):
    all_speeds = []
    for session_key in session_keys:
        sess = canonical_behavior[session_key]["behavior"]
        mask = sess["ft_CorrSpc"] & (sess["ft_move"] > 0)
        if np.any(mask):
            all_speeds.append(np.asarray(sess["ft_RunSpeed"][mask], dtype=np.float32))
    speed_values = np.concatenate(all_speeds)
    q25, q50, q75 = np.quantile(speed_values, [0.25, 0.5, 0.75])
```

iii. The trajectory says the final output uses quartile-binned running speed, and later says exact global cut points are stored in metadata.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global quantile thresholds are computed with `np.quantile`, then `np.digitize` assigns each retained frame to one of four bins.

ii. ```python
def make_speed_bins(speed, speed_edges):
    return np.clip(np.digitize(np.asarray(speed, dtype=np.float32), speed_edges, right=False), 0, 3).astype(np.int8)
```

iii. The trajectory justification is that the output should be quartile-binned and the thresholds should be retained in metadata rather than embedded in the labels.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled from `ft_RunSpeed` at the same retained `frame_idx` used to slice the neural matrix.

ii. ```python
speed_out = make_speed_bins(ft_speed[frame_idx], speed_edges)
```

iii. The trajectory’s alignment rationale is again the shared retained imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script trims behavior streams to the number of imaged frames, checks for behavior mismatches across duplicate experiment listings, checks retinotopy length consistency, drops invalid/out-of-corridor licks, skips empty trials, and skips sessions with fewer than two surviving trials.

ii. ```python
ft_trind = np.asarray(behavior["ft_trInd"][:n_frames])
ft_pos = np.asarray(behavior["ft_Pos"][:n_frames], dtype=np.float32)
...
if len(region_idx_full) != sum(plane.shape[0] for plane in spk_planes):
    raise ValueError(f"Retinotopy length mismatch for {session_key}")
```
```python
same, field = behavior_sessions_match(canonical[base_key]["behavior"], session)
if not same:
    raise ValueError(...)
```
```python
valid = np.isfinite(lick_frame_idx) & np.isfinite(lick_pos) & (lick_pos >= 0.0) & (lick_pos < texture_length)
...
if len(neural_trials) < 2:
    return None
```

iii. The trajectory explicitly mentions schema checking, dry-run validation, and a desire to catch alignment or structural issues before doing the full conversion.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are reading the very large raw spike files and then scoring/selecting neurons per session. The behavior-side work is comparatively light.

ii. ```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
selected_global = select_visual_neurons(...)
session_matrix = build_selected_session_matrix(spk_planes, selected_global)
```

iii. The trajectory explicitly says the raw `spk` directory is about `405 GB`, and later says the full run must read that directory once, score neurons per session, and write the final pickle.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main unvectorized loops are the per-plane / per-region loops in `select_visual_neurons`, the per-trial loop in `convert_session`, and repeated `np.flatnonzero((ft_trind == trial) & retained_mask)` scans across all trials.

ii. ```python
for plane_idx, plane in enumerate(spk_planes):
    ...
    for region in range(len(VISUAL_CORTEX_REGIONS)):
        ...
```
```python
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
```

iii. The trajectory does not justify these loops directly; it only says the expensive step is the one-pass read over the raw recordings.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly recomputes retained trial-frame indices trial-by-trial, repeatedly builds broadcast arrays with `np.full` for per-trial constants, and repeatedly scans plane/region subsets while ranking neurons.

ii. ```python
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
    ...
    reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)
    training_day = np.full(len(frame_idx), day_of_training, dtype=np.float32)
```
```python
for plane_idx, plane in enumerate(spk_planes):
    ...
    for region in range(len(VISUAL_CORTEX_REGIONS)):
        ...
```

iii. There is no explicit trajectory justification for this repeated work beyond the agent’s general focus on deterministic curation and keeping the conversion tractable.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes corridor-versus-gray responsiveness and variance scores only to choose a capped neuron subset, performs duplicate-behavior consistency checks that are not stored in the output, and writes extra metadata such as source-session maps that are not used by the decoder.

ii. ```python
if gray_mask.any():
    gray_mean = plane_valid[:, gray_mask].mean(axis=1)
    responsive = corridor_mean > gray_mean
...
region_selected = sort_take_desc(resp_scores, resp_indices, max_per_region)
```
```python
same, field = behavior_sessions_match(canonical[base_key]["behavior"], session)
...
"source_behavior_keys": {
    key: canonical_behavior[key]["full_key"] for key in session_keys
},
```

iii. The trajectory justifies the neuron-scoring work as a tractability measure, but does not claim the extra metadata or duplicate checks are needed for downstream decoder training.
