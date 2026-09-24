# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the master index `beh/Imaging_Exp_info.npy`. Instead it takes the **89 files in `data/spk/` as the authoritative session list**, then loads **every** `beh/Beh_*.npy` file at once into a dictionary of "behavior views" keyed by the session id with any `_swap1`/`_swap2` suffix stripped. Because a recording is exported into several experiment-type files (99 keys for 89 recordings), a session can have several views; the AI merges them, picking a canonical view (most `UniqWalls`, then most non-NaN `stim_id`, then file/key name) and cross-checking ~20 fields (`ntrials`, `WallName`, `ft`, `ft_trInd`, `ft_Pos`, `ft_RunSpeed`, `StartFr`, `EndFr`, `SoundFr`, ...) across all views, raising if they disagree. Neural traces are then read per session from `spk/<session>_neural_data.npy` (concatenating the per-plane `spks` arrays) and the area labels from `retinotopy/<mouse>_<date>_trans.npz`. It asserts every spk session has a behavior entry.

ii.
```python
def load_behavior_views():
    views = defaultdict(list)
    for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for view_key, view in beh_dict.items():
            base_key = strip_swap_suffix(view_key)
            views[base_key].append({"source_file": beh_path.name, "view_key": view_key, "data": view})
    return views
```
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
...
missing_behavior = [key for key in spk_session_keys if key not in behavior_views]
if missing_behavior:
    raise KeyError(f"Missing behavior entries for sessions: {missing_behavior}")
behavior_by_session = {key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys}
```
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. From CONVERSION_NOTES: "The raw spike release contains 89 physical recordings in `data/spk`, which matches the paper's Methods section. The behavior exports contain 99 keys because some recordings are repeated across analysis files or exposed through paired `swap1` and `swap2` views. The converter rebuilds sessions at the 89-recording level by stripping `_swap1` and `_swap2` suffixes, merging all behavior views that point to the same physical recording, and verifying that the timing and trial annotations match across those views." Trajectory step 37: "I've resolved one key consistency issue: the behavior exports contain 99 keys, but the paper reports 89 recordings because ten `swap1`/`swap2` entries are duplicated views of the same physical session."

## 1-b. How are the data split into subjects?

i. The mouse name is the first underscore-delimited token of the session key (which is also the spk file name). Subjects are the sorted unique names (19), and `subject_idx` is the index of each session's mouse into that list. Result: 19 subjects, with the same session counts per subject as the human reference (DR10 6, TX108 7, TX119 8, TX123 8, TX140 1, VR2 7, ...).

ii.
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    ...
    return {"subject": subject, "date": date_obj, "date_str": date_str, "block": block}
```
```python
subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_id[parsed[session_key]["subject"]])
```

iii. No explicit discussion; the file/key naming convention `<mouse>_<year>_<month>_<day>_<block>` was established by inspection early in the trajectory (steps 28-33), and the AI reported "Subjects: 19" as a sanity check against the Methods statement "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one spk file, i.e. one mouse/date/block. The 89 spk files define the session set exactly; duplicated behavior keys (`_swap1`/`_swap2` and the same recording appearing in several `Beh_<exp_type>.npy` files) are merged into a single session rather than being treated as separate sessions. All 89 sessions survive into the output.

ii.
```python
def strip_swap_suffix(session_key: str) -> str:
    return re.sub(r"_swap[12]$", "", session_key)
```
```python
canonical = max(session_views, key=sort_key)
template = canonical["data"]
for other in session_views:
    for field in check_fields:
        if not _arrays_match(template[field], other["data"][field]):
            raise ValueError(f"Behavior views disagree for session {base_key} on field {field}: ...")
```

iii. Same as 1-a: the AI wanted the session count to reproduce the paper's "We performed 89 recordings in 19 mice", and it verified that the duplicate views really are the same recording before collapsing them (rather than assuming it).

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` trials the behavior declares; a frame belongs to trial `t` if `ft_trInd == t`. On top of the trial index the AI requires the frame to be **inside the texture (`ft_CorrSpc`) and moving (`ft_move > 0`)**, so each trial is the running portion of the corridor traversal (grey space excluded, stationary frames excluded). NaNs in `ft_trInd` are mapped to -1 so they never match a trial. Each trial is then collapsed to **4 samples** by interpolating onto the centres of the four 1-m position bins (see 2-b/2-e), so every trial in the exported dataset has exactly 4 "timepoints".

ii.
```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
    valid = ~np.isnan(ft_trial_idx_raw)
    ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
    ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]
```
```python
for trial_idx in range(int(beh["ntrials"])):
    frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
```

iii. CONVERSION_NOTES: "Only frames satisfying `ft_move > 0` and `ft_CorrSpc == True` are used. This follows the paper's statement that analyses only considered timepoints during running and matches the reference code's repeated use of moving corridor masks." (methods.txt: "We only considered timepoints during running for analysis"; `code/utils.py` builds `VRmove = beh['ft_move'][:nfr]>0` and `corr_fr = beh['ft_CorrSpc'][:nfr] & VRmove`). Trajectory step 48: "using all frames gives pathological trial durations because mice sometimes pause for hundreds of frames, while `ft_move > 0` inside the corridor yields the stable 20-35 frame traversals the paper analyzes."

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both applied per trial: (1) a trial with **no** moving-in-corridor frames is dropped; (2) a trial whose maximum position among those frames does not reach the last bin centre (3.5 m) is dropped, so that the 4-bin interpolation is never an extrapolation. **Neither filter ever fires**: the run reports `Empty running trials dropped: 0` and all 38,110 trials are kept. There is no trial-duration/outlier filter, so trials in which the animal stood still for many minutes are retained (the converted data contains trials whose `time_since_trial_start` reaches 1764 s and `time_to_sound_cue` reaches -1762 s). Sessions with fewer than 2 usable trials would raise, but none did.

ii.
```python
frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue

trial_positions = ft_pos[frame_idx]
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    trial_stats["empty_running_trials"] += 1
    continue
```
```python
trial_stats["ntrials_kept"] = len(session_neural)
if len(session_neural) < 2:
    raise ValueError(f"Session {session_key} has fewer than 2 usable trials after running-frame filtering")
```

iii. CONVERSION_NOTES: "Trials without usable running corridor frames are dropped." The AI's implicit position is that the `ft_move > 0` mask already removes the pathological pauses (trajectory step 48), so no separate duration filter is needed; the second filter (reaching 3.5 m) is not documented in the notes, only in the code. The AI did not re-examine total trial duration after the mask was applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session>_neural_data.npy` (a list of one neurons x frames deconvolved-trace array per imaging plane, concatenated along neurons), and `iarea` from `retinotopy/<mouse>_<date>_trans.npz` for the area of each neuron. The number of frames of the concatenated `spks` defines `n_frames`, to which all behavior streams are truncated.

ii.
```python
spk_path = SPK_ROOT / f"{session_key}_neural_data.npy"
ret_path = RET_ROOT / f"{session_key.rsplit('_', 1)[0]}_trans.npz"
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
n_frames = spk.shape[1]
```

iii. CONVERSION_NOTES: "Neural activity comes from the deconvolved traces stored in `*_neural_data.npy`. Brain-region labels come from the retinotopy files in `data/retinotopy`." This mirrors `load_spk`/`load_retino` in the paper's `code/utils.py`, which the AI read (trajectory steps 15-17). Sanity check reported: raw neuron counts 20,547-89,577, matching the Methods text exactly.

## 2-b. How is the `neural` data processed?

i. No filtering, normalisation or deconvolution is applied to the traces themselves. Instead each trial's activity is **resampled from frames onto 4 spatial bin centres (0.5, 1.5, 2.5, 3.5 m)**: within a trial the frames are sorted by `ft_Pos`, frames sharing a position are averaged, and `np.interp` evaluates each neuron's trace at the four bin centres. The result is stored as float16 with shape `(512, 4)`. Only 512 neurons per session are kept (see 2-c). So a ~23-frame traversal becomes 4 position-averaged samples.

ii.
```python
def interpolate_features(positions, values, target_positions):
    positions = np.asarray(positions, dtype=np.float32)
    order = np.argsort(positions)
    ...
    unique_pos, inverse = np.unique(positions, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    ...
    collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts
    ...
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row]).astype(np.float32)
    return interp
```
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. CONVERSION_NOTES: "The decoder task explicitly requires 4 equal 1 m position bins, so each trial is represented by 4 ordered samples corresponding to corridor positions `0-1 m`, `1-2 m`, `2-3 m`, and `3-4 m`. Within each trial, neural activity and continuous task variables are interpolated onto the centers of those bins from the retained running-in-corridor frames." It lists this as one of "Two Intentional Format-Driven Deviations": "The reference analyses often operate either on raw running frames or on finer position-interpolated activity, but the decoder task here requires exactly 4 spatial bins." (The paper's `get_interpPos_spk` does interpolate spikes onto position, but with `n_bins=60` over the 60-dm corridor, i.e. 1-dm resolution, not 4 bins.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two steps. (1) **Area filter, matching the reference**: a neuron is kept only if its `iarea` code falls in V1 `{8}`, mHV `{0,1,2,9}`, lHV `{5,6}` or aHV `{3,4}`. (2) **A hard cap of 512 neurons per session**, not present in the paper or the human reference: variance of each visual-cortex neuron is computed over all moving-in-corridor frames of the session, and the top 128 neurons by variance are taken from each of the four areas (any shortfall is filled by the next-highest-variance neurons overall). Selection is deterministic (ties broken by index). Every session therefore ends up with exactly 512 neurons — 128 per area — instead of the 17k-79k it actually has, i.e. roughly 1% of the recorded population, and the true area composition (V1 ≈ 45%, lHV ≈ 12%) is replaced by an artificial 25/25/25/25 split.

ii.
```python
VISUAL_AREA_CODES = {"mHV": {0, 1, 2, 9}, "aHV": {3, 4}, "lHV": {5, 6}, "V1": {8}}
```
```python
if len(valid_indices) <= max_neurons:
    ...
    return valid_indices, region_idx

spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
variance_map = dict(zip(valid_indices.tolist(), variances.tolist()))

per_region_quota = max_neurons // len(BRAIN_REGIONS)
selected = []
for region_name in BRAIN_REGIONS:
    candidates = np.array(region_to_indices[region_name], dtype=np.int64)
    order = sorted(candidates.tolist(), key=lambda idx: (-variance_map[idx], idx))
    selected.extend(order[: min(per_region_quota, len(order))])
```
```python
DEFAULT_MAX_NEURONS = 512
```

iii. CONVERSION_NOTES: "Neurons outside these visual-cortex groups are excluded, consistent with the reference analysis code paths that remove neurons outside visual cortex." And for the cap: "A deterministic neuron cap is applied per session after visual-cortex filtering so the expanded trial structure remains trainable with the provided decoder harness. The raw neuron counts are preserved in metadata." Trajectory step 75: "The export itself will stay compact by keeping the paper's running/corridor mask, averaging within the four required 1 m bins, and capping neurons per session with a deterministic label-free rule." The AI explicitly flags it as a deviation not required by the format.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is declared as corridor entry, and each trial's four samples are positions measured from corridor entry (0-1, 1-2, 2-3, 3-4 m), so sample 0 of every trial is the first metre after entry. Alignment is therefore **spatial rather than temporal**: trials are not cut to a common time window and are not padded, they are all forced to the same 4 spatial samples. `off_start` is set to 0.0 and `off_end` to 4 m / 60 cm s-1 = 6.67 s (the nominal traversal time).

ii.
```python
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
...
"temporal_alignment_event": "corridor entry (trial start)",
"off_start": 0.0,
"off_end": float(4.0 / 0.60),
"binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
```

iii. CONVERSION_NOTES: "Temporal alignment event: corridor entry (`Trial_start_time`, `StartFr`)"; the position axis is anchored at corridor entry, and the AI relies on the paper's statement that the VR advances at a fixed 60 cm s-1 whenever the mouse runs above threshold, which makes position and elapsed time proportional for running frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw resolution is the imaging frame, which the AI measured itself as a median frame period of 0.3147 s (3.18 Hz) and recorded in metadata as `frame_period_s_median`. The data are then **rebinned out of time entirely into 4 spatial bins per trial**. The declared `time_bin_size` is 1666.67 ms, computed as 1 m / 60 cm s-1, i.e. the nominal time to traverse one metre at the fixed VR speed. This is only nominal: measured over the exported data the actual per-bin duration has a median of 1.65 s but a 95th percentile of 10.0 s and a maximum of 344.7 s, and 13% of bins exceed 5 s, so bins are **not** of equal duration across trials or sessions.

ii.
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),
"nominal_vr_speed_cm_per_s": 60.0,
"frame_period_s_median": float(np.median(frame_periods)),
```
```python
frame_periods = []
for key in spk_session_keys:
    ft = np.asarray(behavior_by_session[key]["ft"], dtype=np.float64)
    if len(ft) > 1:
        frame_periods.append(float(np.median(np.diff(ft) * SECONDS_PER_DAY)))
```

iii. The AI never states the equal-duration claim explicitly; it justifies the spatial binning by the decoder-task requirement for 4 position bins and derives `time_bin_size` from the paper's constant VR speed ("The mice moved forward ... but the virtual corridors always moved at a constant speed (60 cm s-1) as long as mice kept running faster than the threshold"). It reported the measured median frame period as a sanity check but did not compare it, or the realised bin durations, against the value it wrote into `time_bin_size`.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundTime` (the MATLAB datenum of the sound cue on each trial) and `ft` (the datenum of each imaging frame). This is exactly equivalent to the reference's `np.interp(SoundFr, arange(n), frame_time)`: checked against the data, `SoundTime` equals the frame-time interpolation of `SoundFr` to floating-point precision.

ii.
```python
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
...
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
```

iii. Not discussed explicitly; the AI inspected `ft`, `SoundFr`, `SoundTime`, `SoundTimeDelay` early (trajectory steps 27, 44) and chose the datenum fields so no frame-to-time interpolation is needed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Per frame of the trial, `(SoundTime - ft) * 86400` gives seconds, positive before the cue and negative after it (the same sign convention as the reference). Those per-frame values are then resampled onto the 4 position-bin centres with the same position interpolation used for the neural data. Values are stored float32. Across the dataset the range is [-1762.2, 722.8] s, reflecting the very long pause trials that are never filtered out.

ii.
```python
SECONDS_PER_DAY = 24.0 * 3600.0
...
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
...
input_trial = np.vstack([
    interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
    ...
])
```

iii. CONVERSION_NOTES lists `time_to_sound_cue_s` among the inputs; the unit conversion follows from the MATLAB datenum format the AI identified when inspecting `ft`. No further rationale is given for the sign, which matches the literal reading of "time to sound cue".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same frame indices (`frame_idx`) as the neural data of that trial and then interpolated onto the same 4 position-bin centres, so it is sample-for-sample aligned with the neural array.

ii.
```python
frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
trial_positions = ft_pos[frame_idx]
neural_interp = interpolate_features(trial_positions, spk_selected[:, frame_idx], POSITION_BIN_CENTERS)
...
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. All streams in this dataset are already on the imaging-frame grid (the AI truncates every behavior stream to `n_frames = spk.shape[1]`), and the AI reuses one `frame_idx`/`trial_positions` pair for every stream of a trial, which guarantees alignment by construction.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The date embedded in the session key (`<mouse>_<year>_<month>_<day>_<block>`), parsed with `datetime.strptime`. No behavior field is used.

ii.
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    date_obj = datetime.strptime(f"{year}_{month}_{day}", "%Y_%m_%d").date()
```

iii. Not discussed; the session key is the only field that dates a recording, the same reasoning the human reference gives.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the earliest recording date is found, and each session's value is the **number of calendar days elapsed** since that mouse's first recording. The value is a per-trial constant broadcast across the trial's 4 samples. The resulting range is 0-92 days (the human reference instead counts recording sessions in order, giving 0-7).

ii.
```python
def compute_subject_day_index(session_keys: list[str]):
    parsed = {key: parse_session_key(key) for key in session_keys}
    first_day = {}
    for key, info in parsed.items():
        subj = info["subject"]
        if subj not in first_day or info["date"] < first_day[subj]:
            first_day[subj] = info["date"]
    day_index = {}
    for key, info in parsed.items():
        day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
    return day_index, parsed
```
```python
np.full(4, subject_day_value, dtype=np.float32),
```

iii. No explicit justification in the notes; the implicit reading is the literal one — "day of training" as elapsed days, treating the first recording of each mouse as day 0 (the actual start of training is not in the data; the paper says recordings were made before and after ~2 weeks of training).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `Trial_start_time` (the datenum of corridor entry for each trial) and `ft`. `Trial_start_time` was verified against the data to be identical to the reference's `np.interp(StartFr, arange(n), frame_time)`.

ii.
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. CONVERSION_NOTES: "Temporal alignment event: corridor entry (`Trial_start_time`, `StartFr`)" — the AI treats the two as the same event, which is correct in this data.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(ft - Trial_start_time) * 86400` in seconds, positive after entry, then interpolated onto the 4 position-bin centres. Because the first sample sits at 0.5 m rather than at entry, the minimum value is ~0.5-0.8 s rather than 0. The dataset range is [0.5, 1764.1] s.

ii.
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0],
```

iii. Same as 3-b: the datenum-to-second conversion and a difference from the alignment event; no further rationale given.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Identically to 3-c: same `frame_idx`, same `trial_positions`, same 4 target bin centres, so it is element-wise aligned with the neural array.

ii.
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. All streams share the imaging-frame grid and one interpolation target per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor. Same field as the human reference.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32),
```

iii. Not discussed beyond an inspection of `isRew` values per cohort (trajectory step 42), which confirmed it is all-False for the naive and unsupervised cohorts.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool-to-float cast and broadcasting to the trial's 4 samples. Range in the output is [0, 1], and it is 0 for every trial of the naive/unsupervised sessions.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32),
```

iii. N/A — the field is already the required binary variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the wall texture. `TrialStim` is not used (it is masked in the swap sessions), matching the human reference's choice.

ii.
```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)
```
```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
"visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
```

iii. Not discussed in detail; the AI enumerated the unique `UniqWalls` values across all behavior files early (trajectory step 51) and built the label set from what actually occurs in the data.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The **15 distinct wall names** that occur anywhere in the dataset (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`) are sorted naturally and used directly as the 15 categories; no collapsing to the 4 base textures (circle/leaf/rock/wood) is done. The label is per trial, broadcast across its 4 samples, and stored int16. Any one session contains only 2-4 of the 15 labels, so the label partly identifies the session/mouse; chance level for the decoder is reported as 1/15 = 0.0667.

ii.
```python
stimulus_values = stimulus_catalog(behavior_by_session)
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
...
"output_values": [stimulus_values, ["no_lick", "lick"], [...], [...]],
```

iii. The instructions the AI was given specify the output as "Visual stimulus category. e.g. circle1, leaf2, etc., per-trial", i.e. they name the fine-grained wall identities as the examples, so the AI took the raw `WallName` values as the categories. The paper's own labelling also distinguishes leaf1/leaf2/leaf3 and the spatial shuffles as different stimuli.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickTrind` (the trial index of each lick) and `LickPos` (the corridor position of each lick, in decimetres), rather than the reference's `LickFr` (frame number of each lick). The position-based fields are used because the exported samples are spatial bins, not frames.

ii.
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
...
lick_pos_trial = lick_positions[lick_trials == trial_idx]
```

iii. Not discussed explicitly; the choice follows mechanically from the decision to represent each trial as 4 position bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, a bin is 1 if at least one of that trial's licks falls in the half-open position interval of the bin (`[0,10), [10,20), [20,30), [30,40)` dm), else 0. In addition, **any lick at a position beyond 4 m — i.e. in the grey space between corridors — sets the last bin to 1**. The overall lick rate in the output is 0.091 of samples (0.12 in the sample I measured over the first 100 trials of each session), versus 0.041 for the reference's per-frame encoding; the difference is expected from the ~5x coarser bins, but the grey-space fold-in also contributes (about 9% of all licks in a rewarded session occur at position >= 4 m). Licks are not restricted to frames where the animal was running, unlike every other stream.

ii.
```python
lick_trial = np.array(
    [
        int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) & (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
        for bin_idx in range(4)
    ],
    dtype=np.int16,
)
if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
    lick_trial[-1] = 1
```

iii. CONVERSION_NOTES only lists `licking` as a binary output. No stated reason for folding grey-space licks into the last corridor bin; the implicit motivation is that reward-anticipation licking in the rewarded corridor happens late and often spills past the texture, and the AI did not want to discard it.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. By position: the lick bins are the same four 1-m intervals whose centres the neural data is interpolated onto, for the same trial index, so the licking row is element-wise aligned with the neural array.

ii.
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```

iii. Consistent with the general scheme: every stream of a trial is expressed on the same four spatial bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos` — but only as the interpolation axis. The exported position output is not read from `ft_Pos` at all; it is the index of the bin, which is the same for every trial.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
...
trial_positions = ft_pos[frame_idx]
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Follows from the spatial-binning representation: once the samples *are* the position bins, the position label is the bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. None — it is literally `np.arange(4)` for every trial in every session. Consequently the position output is exactly uniform (0.250 of samples in each of the four bins in every one of the 89 sessions) and carries no trial-to-trial information. It is also fully determined by the decoder's own inputs, because `time_since_trial_start` is monotonically increasing across the four samples of a trial; the decoder consumes `input` alongside neural PCs, so position can be read straight off the input. This shows in the accuracy: 0.94 validation balanced accuracy for position, versus 0.31 for the human reference.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. CONVERSION_NOTES: "The decoder task explicitly requires 4 equal 1 m position bins, so each trial is represented by 4 ordered samples corresponding to corridor positions `0-1 m`, `1-2 m`, `2-3 m`, and `3-4 m`." The AI does not remark that this makes the position target constant, and it reported the resulting 0.94 accuracy as a successful validation result rather than as a red flag.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-m bins spanning the 4-m texture, with edges at 0, 10, 20, 30, 40 dm (`ft_Pos` is in decimetres, 0-40 across the texture and on to 60 through the grey space). Grey space is excluded from trials altogether, so no clipping is needed. The category names are `0_to_1m`, `1_to_2m`, `2_to_3m`, `3_to_4m`. These are the same bin boundaries as the human reference's `ft_Pos // 10`.

ii.
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
...
"output_values": [..., ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"], ...],
```

iii. Directly from the decoder-task specification: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Trivially: sample k of the neural array is by construction the activity interpolated at the centre of bin k, and the position output of sample k is k.

ii.
```python
neural_interp = interpolate_features(positions=trial_positions, values=spk_selected[:, frame_idx],
                                     target_positions=POSITION_BIN_CENTERS)
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Same as 8-c: one shared set of bin centres per trial for all streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the per-frame running speed, restricted to the moving-in-corridor frames of the trial. Same source field as the human reference.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
...
speed_interp = interpolate_features(positions=trial_positions, values=ft_speed[frame_idx],
                                    target_positions=POSITION_BIN_CENTERS)[0].astype(np.float32)
```

iii. Not discussed; it is the only speed field in the behavior dict.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is averaged/interpolated onto the four bin centres exactly like the other streams, giving one continuous speed per bin. The continuous values from **all trials of all 89 sessions** are then pooled and the 25th/50th/75th percentiles of that pool become global bin edges, applied with `np.digitize`. The discretisation is therefore a single global rule rather than the reference's per-session rank split. Because only running frames enter, the zero-speed ties that motivated the reference's rank-based split largely disappear. Realised edges: 16.59, 28.70, 43.17 cm s-1; realised marginals 0.250/0.250/0.250/0.250.

ii.
```python
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)
full_output = [finalize_outputs(partials, speed_edges) for partials in output_partials]
```
```python
def finalize_outputs(session_output_partial, speed_edges):
    for trial in session_output_partial:
        speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. CONVERSION_NOTES reports "Global running-speed quartile edges: `16.592`, `28.701`, `43.168`" as a sanity check. The decoder-task text asks for "4 bins, each corresponding to 25% of the data", which the AI reads as 25% of the whole dataset; doing it globally keeps a single, interpretable speed scale across sessions and mice (a per-session split would give the same marginals but different physical meanings per session).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global quartile edges, producing categories `speed_q1..speed_q4`. The verification output confirms each category holds exactly 0.250 of the data. Per-session marginals are far from uniform (e.g. one session is 0.997 in q1, another 0.723 in q4), which is the expected consequence of a global rather than per-session split.

ii.
```python
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
...
"output_values": [..., ["speed_q1", "speed_q2", "speed_q3", "speed_q4"]],
```

iii. Directly from the task specification ("4 bins, each corresponding to 25% of the data"), interpreted globally.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Same `frame_idx` and same 4 bin centres as the neural data, so element-wise aligned; the discretisation is applied afterwards without changing the alignment.

ii.
```python
speed_interp = interpolate_features(positions=trial_positions, values=ft_speed[frame_idx],
                                    target_positions=POSITION_BIN_CENTERS)[0].astype(np.float32)
```

iii. Same shared-bin scheme as every other stream.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Three things are handled. (1) The behavior can run past the imaging, so every frame-indexed stream is truncated to `n_frames = spk.shape[1]` (`ft`, `ft_Pos`, `ft_RunSpeed`, `ft_move`, `ft_CorrSpc`, `ft_trInd`). (2) `ft_trInd` contains NaN for frames outside any trial; these are mapped to -1 so they never match a trial index — this was a fix applied after the first smoke test (trajectory step 85 patches an earlier `np.rint(...).astype(int)` that would have mangled NaNs). (3) `np.nanmax` is used for the trial position check. Inconsistent duplicate behavior views raise rather than being silently merged. Not handled: licks are never truncated to the imaged period (`LickPos`/`LickTrind` are used unfiltered), and there is no tolerance for a failed session — any exception aborts the whole conversion instead of skipping the session.

ii.
```python
n_frames = spk.shape[1]
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
```
```python
ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```

iii. The truncation mirrors the paper's code (`beh[...][:nfr]` with `nfr = spk.shape[1]`), which the AI read in `code/utils.py`. The NaN handling was found by the AI's own single-session smoke test and fixed immediately, in line with its instruction to "carefully verify your work after every step".

## 12-a. What are the most time-consuming steps of the code?

i. Two. (1) Reading and concatenating the spike files — 405 GB of float32 traces across 89 sessions, each up to 4.3 GB, plus a full copy at `np.concatenate`. (2) The per-neuron Python loops inside `interpolate_features`: for every trial it runs one `np.bincount` and one `np.interp` per neuron, i.e. 2 x 512 numpy calls per trial x 38,110 trials ≈ 39 million small calls. The one-off costs of loading all 6.6 GB of behavior files at once and computing per-neuron variance over all visual neurons (up to ~90k x ~20k floats per session) are secondary but non-trivial.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```
```python
for row in range(values.shape[0]):
    collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts
for row in range(collapsed.shape[0]):
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row]).astype(np.float32)
```

iii. The AI recognised the I/O cost and designed around it: trajectory step 75, "I'm writing the converter with the memory-heavy part done session-by-session so it can traverse the 405 GB spike directory without holding multiple recordings at once", with explicit `del spk` / `gc.collect()` per session. It ran the full conversion detached in the background and polled it (steps 100-143). It never profiled or commented on the per-neuron interpolation loops.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) `interpolate_features` loops over neurons for both the duplicate-position collapse and the interpolation; the collapse is a single matrix product / `np.add.reduceat` and the interpolation can be done on the whole matrix at once with searchsorted-based weights. (2) `trial_frame_indices` rebuilds the whole `ft_trInd` mask — `asarray`, `isnan`, `rint`, plus the `ft_move` and `ft_CorrSpc` masks — once **per trial**, scanning all ~300k frames up to 789 times per session; it should be computed once per session and the frames grouped by trial in a single pass. (3) `pick_neurons_by_variance` uses Python `sorted` over lists of up to ~90k neuron indices with a dict-lookup key function, where `np.argsort` on the variance array would do. (4) The per-bin licking test is a 4-iteration list comprehension per trial where `np.searchsorted`/`np.bincount` over all licks of a session would do.

ii.
```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
    valid = ~np.isnan(ft_trial_idx_raw)
    ...
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]
```
```python
order = sorted(candidates.tolist(), key=lambda idx: (-variance_map[idx], idx))
```

iii. Not discussed. The AI's efficiency attention went to memory (per-session load, `del`, `gc.collect`, float16 storage) and to keeping the exported artifact small, not to CPU time; with a 1 TB machine and the conversion running in the background the cost was tolerated rather than optimised.

## 12-c. What processing does the code repeat multiple times?

i. (1) The frame masks described in 12-b are recomputed for every trial. (2) The position sort / `np.unique` / `np.bincount` bookkeeping inside `interpolate_features` is redone from scratch four times per trial (neural, speed, time-to-cue, time-since-start) although all four calls share the identical `trial_positions`. (3) The behavior files are read twice in effect: once in `load_behavior_views`, and then `ft` is walked again for every session up front just to compute `frame_periods`. (4) The interpolated speed is stored twice (in `output_partial["running_speed_continuous"]` and in `speed_values`) and concatenated twice (per session, then globally). (5) `merge_behavior_views` compares ~20 fields, including 300k-element arrays, across every pair of duplicate views of every session, only to raise if they disagree.

ii.
```python
frame_periods = []
for key in spk_session_keys:
    ft = np.asarray(behavior_by_session[key]["ft"], dtype=np.float64)
    if len(ft) > 1:
        frame_periods.append(float(np.median(np.diff(ft) * SECONDS_PER_DAY)))
```
```python
session_output_partial.append(output_trial_partial)
speed_values.append(speed_interp)
```

iii. Not discussed. The duplicated view checking is deliberate and defended in CONVERSION_NOTES as a correctness guarantee ("verifying that the timing and trial annotations match across those views"); the rest appears to be incidental.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `ft_Pos` is interpolated and thresholded conceptually, but the emitted position output is a constant `np.arange(4)` — the position information is discarded. (2) The variance of every visual-cortex neuron is computed over all running frames of the session (a ~90k x ~20k reduction) purely to rank neurons, and 99% of those neurons are then dropped. (3) All 99 behavior views are held in memory and cross-compared field-by-field; only the canonical view's values are ever used. (4) `frame_periods` is computed over all 89 sessions to produce one metadata scalar that nothing downstream reads. (5) `merged["source_behavior_views"]` is threaded through the whole pipeline into metadata and unused. (6) A second 17 MB `sample_data.pkl` is built with `make_sample_dataset` on every run (required by the AI's own deliverable list, but not by the conversion). (7) `plane.astype(np.float32, copy=False)` is a no-op since `spks` is already float32.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```
```python
spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
```
```python
merged["source_behavior_views"] = [
    f"{item['source_file']}::{item['view_key']}" for item in sorted(session_views, ...)
]
```

iii. Most of these are the price of decisions the AI made deliberately and documented (the neuron cap, the view-consistency check, the extra metadata for auditability). The degenerate position output is the one case the AI appears not to have noticed: it reports 0.94 position accuracy in CONVERSION_NOTES as a validation success.
