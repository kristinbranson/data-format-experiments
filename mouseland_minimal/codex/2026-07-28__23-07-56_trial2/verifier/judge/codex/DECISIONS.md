# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not start from `Imaging_Exp_info.npy`. It scans all spike files in `data/spk` to define the 89 session keys, scans every `Beh_*.npy` file to collect behavior views, strips `_swap1` and `_swap2` suffixes, merges all behavior views that map to the same base session, and then loads spikes and retinotopy per session.

ii. ```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```
```python
def load_behavior_views():
    views = defaultdict(list)
    for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for view_key, view in beh_dict.items():
            base_key = strip_swap_suffix(view_key)
            views[base_key].append(...)
```
```python
behavior_by_session = {
    key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
}
```

iii. `CONVERSION_NOTES.md` says the behavior exports contain 99 keys but only 89 physical recordings, so the AI chose to reconstruct sessions by stripping swap suffixes, merging repeated behavior views, and verifying that key fields agree across views.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session key prefix before the first underscore, then unique subject names are sorted and each session gets a `subject_idx`.

ii. ```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    ...
    return {"subject": subject, ...}
```
```python
subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_id[parsed[session_key]["subject"]])
```

iii. There is no separate explicit justification beyond the implementation; the notes summarize the result as 19 subjects reconstructed from the 89 physical sessions.

## 1-c. How are the data split into sessions?

i. A session is any physical recording with a spike file in `data/spk`. The AI treats `_swap1` and `_swap2` behavior keys as alternate views of the same physical session and merges them under the base spike-file session key.

ii. ```python
def strip_swap_suffix(session_key: str):
    return re.sub(r"_swap[12]$", "", session_key)
```
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```
```python
for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
    ...
    base_key = strip_swap_suffix(view_key)
    views[base_key].append(...)
```

iii. The notes and trajectory justify this as reconstructing the paper’s 89 “physical recordings” from 99 behavior keys by merging duplicated swap views.

## 1-d. How are the data split into trials?

i. Trials are taken from `beh["ntrials"]`, but the per-trial samples are not the raw frame windows. For each trial the AI keeps only frames where `ft_trInd == trial_idx`, `ft_move > 0`, and `ft_CorrSpc` is true, then interpolates each surviving trial onto four fixed position-bin centers at 5, 15, 25, and 35 dm.

ii. ```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ...
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]
```
```python
trial_positions = ft_pos[frame_idx]
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. `CONVERSION_NOTES.md` says the AI chose “4 ordered samples” because the decoder task asked for four 1 m position bins, and the trajectory says it wanted to respect a “running-only” analysis rule.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have no running-in-corridor frames after filtering, or if their maximum corridor position never reaches the last 1 m bin center. Sessions with fewer than two surviving trials are rejected.

ii. ```python
frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue
```
```python
trial_positions = ft_pos[frame_idx]
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    trial_stats["empty_running_trials"] += 1
    continue
```
```python
if len(session_neural) < 2:
    raise ValueError(f"Session {session_key} has fewer than 2 usable trials after running-frame filtering")
```

iii. The notes say only running frames inside the corridor are used and “trials without usable running corridor frames are dropped.” No separate justification is given for dropping trials that do not reach 35 dm.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from concatenated `spks` arrays in each `*_neural_data.npy` file, with neuron-area labels taken from `iarea` in the corresponding retinotopy `.npz`.

ii. ```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. The notes explicitly say neural activity comes from deconvolved traces in `*_neural_data.npy` and brain-region labels come from `data/retinotopy`.

## 2-b. How is the `neural` data processed?

i. The AI filters neurons, then for each trial it takes only running-in-corridor frames and linearly interpolates each neuron’s activity onto four spatial-bin centers. The resulting 4-column neural matrix is cast to `float16`.

ii. ```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```
```python
session_neural.append(neural_interp)
```

iii. The notes call this a “decoder-facing representation”: each trial is represented by four 1 m bins inside the corridor, with neural activity interpolated onto those bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered to retinotopy-defined visual-cortex areas (`V1`, `mHV`, `lHV`, `aHV`). If more than `max_neurons` remain, the AI ranks them by variance on running corridor frames, applies a balanced per-region quota, and caps the session to 512 neurons by default.

ii. ```python
def area_code_to_region_name(area_code: float):
    ...
    for region_name, codes in VISUAL_AREA_CODES.items():
        if area_code in codes:
            return region_name
```
```python
selected_neurons, region_idx = pick_neurons_by_variance(
    spk=spk,
    area_codes=area_codes,
    running_corridor_mask=running_corridor_mask,
    max_neurons=max_neurons,
)
```
```python
variances = np.var(spk_valid, axis=1)
...
per_region_quota = max_neurons // len(BRAIN_REGIONS)
```

iii. The notes justify the area filter as matching grouped visual areas in `code/utils.py`, and explicitly describe the 512-neuron cap as an intentional format-driven deviation to keep decoder training tractable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says trials are aligned to corridor entry, but the actual per-trial neural representation is not a contiguous time window from that event. Instead it uses the subset of running corridor frames for a trial and interpolates them onto four corridor-position bins.

ii. ```python
trial_frame_indices(beh, n_frames, trial_idx)
```
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```
```python
"temporal_alignment_event": "corridor entry (trial start)",
```

iii. The notes state the nominal alignment event is corridor entry, but also state that the export uses four 1 m spatial bins and only moving corridor frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted trials have four bins that correspond to 1 m spatial bins, not imaging frames. The metadata reports `time_bin_size = (1 / 0.60) * 1000` ms, treating 60 cm/s as a nominal VR speed. The code therefore replaces frame-wise timing with a derived spatial-bin timescale.

ii. ```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),
"binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
```

iii. The notes explicitly describe the converted trials as four 1 m spatial bins rather than raw imaging frames.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and frame timestamps `ft`, not from `SoundFr`.

ii. ```python
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
...
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. There is no explicit written justification for preferring `SoundTime`; the decision is implicit in the code.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For the kept running-corridor frames of a trial, the AI computes `SoundTime - ft` in seconds, then interpolates that per-frame quantity onto the four position-bin centers.

ii. ```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```
```python
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. The only justification available is the general “4 spatial bins” representation in the notes.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same running-corridor frame subset and the same interpolation onto the four position-bin centers that are used for the neural data.

ii. ```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
...
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. The notes say inputs and neural activity are both interpolated onto the 4-bin corridor representation.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session key, specifically the subject name and calendar date parsed from `subject_year_month_day_block`.

ii. ```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    date_obj = datetime.strptime(date_str, "%Y_%m_%d").date()
```

iii. No separate justification is given beyond the implementation.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the number of elapsed calendar days since each subject’s first recording date, stores that as a float per session, and broadcasts it across the four bins of every trial.

ii. ```python
for key, info in parsed.items():
    day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
```
```python
np.full(4, subject_day_value, dtype=np.float32)
```

iii. There is no explicit justification for using elapsed date differences rather than ordinal session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-trial `Trial_start_time` and frame timestamps `ft`, not from `StartFr`.

ii. ```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. No explicit written justification is given for using `Trial_start_time`; the choice is implicit in the implementation.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For the kept running-corridor frames of a trial, the AI computes `ft - Trial_start_time` in seconds, then interpolates that quantity onto the four position-bin centers.

ii. ```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```
```python
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. The only explicit justification is again the choice to represent every trial by four spatial bins.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by using the same running-corridor frame subset and the same interpolation onto the four position-bin centers that are used for the neural data.

ii. ```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. The notes describe a shared four-bin trial representation for neural activity and continuous task variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `beh["isRew"]`.

ii. ```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No separate justification is given; this is treated as a direct per-trial label.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond converting the trial value to `float32` and broadcasting it across the four bins of the trial.

ii. ```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No further justification is stated.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh["WallName"]`.

ii. ```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
```

iii. No separate justification is given beyond using the recorded wall texture name.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI builds a dataset-wide catalog of all unique `WallName` strings and encodes each raw wall-name string directly as a category id. It does not collapse variants like `circle1`, `circle2`, or swap textures into four broad texture classes.

ii. ```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)
```
```python
stimulus_values = stimulus_catalog(behavior_by_session)
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
...
"visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
```

iii. `CONVERSION_NOTES.md` even lists examples like `circle1` and `leaf2`, so the AI appears to have intentionally kept raw stimulus names rather than collapsing them.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTrind` and `LickPos`, not from `LickFr`.

ii. ```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
...
lick_pos_trial = lick_positions[lick_trials == trial_idx]
```

iii. No explicit written justification is given; the choice follows the AI’s spatial-bin trial representation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licking is converted to a 4-element binary vector indicating whether any lick position falls inside each 1 m position bin. If any lick is at or beyond 40 dm, the last bin is forced to 1.

ii. ```python
lick_trial = np.array(
    [
        int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) &
                   (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
        for bin_idx in range(4)
    ],
    dtype=np.int16,
)
if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
    lick_trial[-1] = 1
```

iii. The available justification is only the four spatial-bin export format.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned spatially rather than frame-wise: licking is summarized into the same four corridor-position bins used for neural activity.

ii. ```python
"licking": lick_trial,
```
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The notes say trials are exported as four 1 m spatial bins with neural and task variables interpolated onto those bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The representation is built from corridor position bins defined by `POSITION_BIN_EDGES` and `POSITION_BIN_CENTERS`, with the trial’s frame positions coming from `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```

iii. The notes explicitly justify four equal 1 m corridor bins because the decoder task requested them.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI does not threshold per-frame `ft_Pos` values into categories. Instead every kept trial is represented directly as the fixed sequence `[0, 1, 2, 3]`, one code for each spatial bin.

ii. ```python
output_trial_partial = {
    ...
    "position_bin": np.arange(4, dtype=np.int16),
    ...
}
```

iii. The notes say each trial is represented by the four 1 m bins themselves, so position becomes the bin index sequence.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are hard-coded 1 m bins spanning 0-10, 10-20, 20-30, and 30-40 dm, but the output stores those bins as a fixed ordered sequence rather than thresholding each frame independently.

ii. ```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The justification is the same decoder-facing four-bin representation described in the notes.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is inherently aligned because the neural data are also interpolated onto those same four spatial bins.

ii. ```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The notes describe a shared four-bin corridor representation across neural activity and outputs.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-wise `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. No separate justification is given beyond using the recorded running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each trial the AI interpolates frame-wise running speed from the kept running-corridor frames onto the four position-bin centers, stores those continuous values temporarily, then later bins them.

ii. ```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```
```python
"running_speed_continuous": speed_interp,
```

iii. The notes justify the trial representation as four corridor bins; they do not separately justify speed interpolation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. After all sessions are processed, the AI pools every interpolated speed value across the full dataset, computes global 25th/50th/75th percentiles, and digitizes each trial’s four speed values into four quantile bins.

ii. ```python
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)
```
```python
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. `CONVERSION_NOTES.md` reports the “global running-speed quartile edges,” which shows this was an intentional global-quantile discretization.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by spatial interpolation onto the same four corridor-position bins used for neural activity.

ii. ```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The notes describe a shared four-bin representation for neural and behavioral variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior streams to the imaged frame count, rounds `ft_trInd` after masking NaNs, checks that merged behavior views agree on important fields, ignores retinotopy NaNs by treating them as unmapped neurons, and drops trials/sessions that fail the running-corridor requirements.

ii. ```python
ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```
```python
if not _arrays_match(template[field], other_data[field]):
    raise ValueError(...)
```
```python
def area_code_to_region_name(area_code: float):
    if np.isnan(area_code):
        return None
```

iii. The notes explicitly justify the behavior-view consistency checks and the truncation to imaged frames; the extra trial dropping again follows the running-only four-bin representation.

## 12-a. What are the most time-consuming steps of the code?

i. From the code structure, the most expensive steps are loading and concatenating very large spike arrays, computing variance-based neuron selection on running-corridor frames, and repeatedly interpolating neural and behavioral features for every trial.

ii. ```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```
```python
spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
```
```python
neural_interp = interpolate_features(...)
speed_interp = interpolate_features(...)
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. There is no explicit narrative discussion of runtime in the notes beyond reporting successful full conversion; this conclusion follows directly from the algorithm.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops remain: iterating over every behavior file and view, checking every field across merged views, grouping neurons by region and sorting them, processing every trial one-by-one, constructing licking bin vectors with a Python list comprehension, and interpolating row-by-row inside `interpolate_features`.

ii. ```python
for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
    ...
    for view_key, view in beh_dict.items():
```
```python
for other in session_views:
    ...
    for field in check_fields:
```
```python
for trial_idx in range(int(beh["ntrials"])):
    ...
```
```python
lick_trial = np.array(
    [int(np.any(...)) for bin_idx in range(4)],
    dtype=np.int16,
)
```

iii. No explicit justification is given for leaving these loops as-is.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly interpolates different variables onto the same four position-bin centers for every trial, repeatedly parses/looks up region names during neuron selection, and repeatedly scans merged behavior fields for consistency.

ii. ```python
neural_interp = interpolate_features(...)
speed_interp = interpolate_features(...)
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```
```python
[BRAIN_REGIONS.index(area_code_to_region_name(area_codes[i])) for i in selected]
```

iii. No explicit justification is given. The repetition is a byproduct of the chosen interpolation-heavy representation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores intermediate continuous speed arrays only to discretize them later, performs behavior-view merge checks and source-view bookkeeping that are not used by the decoder, computes extensive metadata summary statistics, and writes both full and sample datasets even though downstream decoding only needs the converted dataset being evaluated.

ii. ```python
"running_speed_continuous": speed_interp,
```
```python
merged["source_behavior_views"] = [
    f"{item['source_file']}::{item['view_key']}" for item in sorted(...)
]
```
```python
metadata = {
    ...
    "session_source_behavior_views": session_source_views,
    "raw_neuron_count_range": ...,
    "selected_neuron_count_range": ...,
    "raw_trial_count_range": ...,
    "kept_trial_count_range": ...,
}
```

iii. The notes explicitly document some of this extra bookkeeping as sanity checking and audit trail material rather than decoder-required payload.
