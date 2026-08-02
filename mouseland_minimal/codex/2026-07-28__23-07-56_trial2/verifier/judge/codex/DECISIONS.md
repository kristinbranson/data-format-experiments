# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent treated every `*_neural_data.npy` file in `/app/data/spk` as a physical recording session, then reconstructed matching behavior by loading every `Beh_*.npy` file in `/app/data/beh`, stripping `_swap1` and `_swap2` suffixes, and merging all behavior views with the same base session key. It also loaded retinotopy from `/app/data/retinotopy/<mouse>_<date>_trans.npz`.

ii. 
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)

behavior_views = load_behavior_views()
behavior_by_session = {
    key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
}
```

```python
for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
    beh_dict = np.load(beh_path, allow_pickle=True).item()
    for view_key, view in beh_dict.items():
        base_key = strip_swap_suffix(view_key)
        views[base_key].append(...)
```

iii. `CONVERSION_NOTES.md` says the agent saw 89 spike recordings but 99 behavior keys, so it intentionally rebuilt the dataset at the 89-recording level by merging repeated behavior views and checking that core timing/trial fields agreed across them.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session key prefix before the first underscore, e.g. `TX60` from `TX60_2021_06_07_1`. A sorted unique subject list is created, and each session gets a `subject_idx`.

ii. 
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    ...
    return {"subject": subject, ...}
```

```python
subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_id[parsed[session_key]["subject"]])
```

iii. No separate prose justification was given beyond reconstructing sessions from file names. The code assumes the file-name subject ID is the authoritative mouse identifier, which is consistent with the raw data layout.

## 1-c. How are the data split into sessions?

i. Sessions are defined one-for-one by spike files in `/app/data/spk`. Behavior keys are merged onto those session IDs, so the exported dataset has 89 sessions.

ii. 
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```

```python
for session_idx, session_key in enumerate(spk_session_keys):
    beh = behavior_by_session[session_key]
    session_data = build_single_session(...)
```

iii. `CONVERSION_NOTES.md` explicitly says the agent rebuilt sessions at the 89-recording level because repeated behavior exports referred to the same physical recordings.

## 1-d. How are the data split into trials?

i. Within each session, the agent loops over `range(beh["ntrials"])`. Trial membership for imaging frames is defined by `ft_trInd == trial_idx`, then additionally masked to moving corridor frames.

ii. 
```python
for trial_idx in range(int(beh["ntrials"])):
    frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
```

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

iii. The notes say the agent wanted the trial representation aligned to corridor entry and limited to running-in-corridor frames, so trial extraction is done directly from `ft_trInd` plus those masks.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have no retained running-in-corridor frames, or if retained positions never reach the last 1 m bin center (`35 cm`). Sessions with fewer than two kept trials are rejected.

ii. 
```python
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue

trial_positions = ft_pos[frame_idx]
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    trial_stats["empty_running_trials"] += 1
    continue
```

```python
if len(session_neural) < 2:
    raise ValueError(f"Session {session_key} has fewer than 2 usable trials after running-frame filtering")
```

iii. `CONVERSION_NOTES.md` says “Trials without usable running corridor frames are dropped.” The extra “must reach last bin center” rule is not separately justified in prose; it follows from the 4-bin export design.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the `spks` entry in each `*_neural_data.npy` file, concatenated across imaging planes. Region labels come from retinotopy `iarea`.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. `CONVERSION_NOTES.md` explicitly states that neural activity comes from the deconvolved traces in `*_neural_data.npy` and that retinotopy files provide the brain-region labels.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, keeps only visual-cortex neurons, applies a deterministic per-session neuron cap using variance during running-in-corridor frames, then interpolates each trial’s retained neural activity onto four spatial bin centers and stores it as `float16`.

ii. 
```python
selected_neurons, region_idx = pick_neurons_by_variance(
    spk=spk,
    area_codes=area_codes,
    running_corridor_mask=running_corridor_mask,
    max_neurons=max_neurons,
)
spk_selected = spk[selected_neurons]
```

```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. The notes justify two main choices here: using the same visual-area grouping as `code/utils.py`, and adding a deliberate 512-neuron cap as a “format-driven deviation” so the decoder could train on the exported trialized dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent filters neurons by retinotopy, keeping only V1/mHV/lHV/aHV neurons. It also only uses frames where the mouse is moving and inside the corridor.

ii. 
```python
def area_code_to_region_name(area_code: float):
    ...
    for region_name, codes in VISUAL_AREA_CODES.items():
        if area_code in codes:
            return region_name
```

```python
running_corridor_mask = (
    (np.asarray(beh["ft_move"][:n_frames]) > 0)
    & np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
)
```

iii. `CONVERSION_NOTES.md` says neurons outside the visual-cortex groups are excluded, and that only moving corridor frames are used because the paper says analysis used timepoints during running.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns trials to corridor entry (`Trial_start_time`, `StartFr`) conceptually, but the exported samples themselves are not framewise time bins. Instead, each trial is represented by four spatial samples at 5, 15, 25, and 35 cm inside the corridor.

ii. 
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```

```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The notes justify this as a decoder-facing design choice: the task required 4 equal 1 m bins, so the agent exported each trial directly on those bins even though the reference analyses often worked framewise or at finer spatial resolution.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have 4 samples per trial, but those samples are spatial bins rather than true temporal bins. Metadata report `time_bin_size = (1 / 0.60) * 1000`, i.e. 1666.7 ms per 1 m bin at 60 cm/s. No explicit temporal rebinning is done; the main rebinning is spatial interpolation to four 1 m bins.

ii. 
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),
"binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
```

iii. The notes say the export uses 4 spatial bins because the decoder task required them. There is no separate justification for calling this a `time_bin_size` beyond the nominal VR speed assumption.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` and per-frame timestamps `ft`, with `ft_Pos` used to interpolate onto the exported spatial bins.

ii. 
```python
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. There is no separate written justification beyond satisfying the required decoder input. The implementation treats time-to-cue as a continuous countdown variable.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the agent computes `SoundTime - ft` in seconds, then interpolates that value from retained frame positions onto the four position-bin centers.

ii. 
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. No explicit note justifies this exact transformation; it is implied by the choice to export all time-varying variables on the same four spatial samples as neural activity.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same retained `frame_idx` and the same interpolation targets (`POSITION_BIN_CENTERS`) as the neural data.

ii. 
```python
input_trial = np.vstack(
    [
        interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
        ...
    ]
)
```

iii. The notes justify a common four-bin representation for neural and continuous variables so the decoder sees matched trial shapes.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each spike/session file name, grouped by subject.

ii. 
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    date_obj = datetime.strptime(date_str, "%Y_%m_%d").date()
```

```python
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
```

iii. No explicit prose justification is given. The code implies the agent used “days since that mouse’s first recorded session” as its proxy for day of training.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest session date is found; each session gets the integer day difference from that earliest date, and that scalar is repeated across the four exported samples in every trial of that session.

ii. 
```python
first_day[subj] = info["date"]
...
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
```

```python
np.full(4, subject_day_value, dtype=np.float32)
```

iii. No separate justification is documented. This is an inferred proxy rather than a direct label from the raw behavior tables.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. The agent did not export an environment-type input at all. No raw variable is used for this channel.

ii. 
```python
"input_names": [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_availability",
],
```

iii. There is no explicit justification in the notes or trajectory. The omission appears to follow the decoder task specification, which did not list environment type as a required input.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The variable is omitted from the exported dataset.

ii. 
```python
"input_names": [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_availability",
],
```

iii. No separate justification is documented.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` and frame timestamps `ft`, again using `ft_Pos` only for interpolation onto exported bins.

ii. 
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. No prose note explains this separately; it is the obvious counterpart to the chosen corridor-entry alignment.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent converts the difference between each retained frame time and the trial start time into seconds, then interpolates that signal onto the four position bins.

ii. 
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. No separate justification is written; it follows from the common four-bin export scheme.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned using the same trial-level retained frames and the same four interpolation targets as the neural data.

ii. 
```python
input_trial = np.vstack(
    [
        ...,
        interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0],
        ...
    ]
)
```

iii. The notes state that all continuous task variables are interpolated onto the same four corridor bins as the neural activity.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew`.

ii. 
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. The notes describe reward availability as one of the exported inputs. The implementation interprets the task’s “1 if in rewarded corridor, 0 if not” literally as a per-trial constant.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts `isRew` to float and repeats it across the four exported samples of the trial.

ii. 
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No extra processing is justified in prose; it is a direct encoding of the requested per-trial discrete variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, not from `stim_id`.

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
```

iii. No direct prose justification is given, but this choice avoids inconsistencies in `stim_id` across some merged behavior views and keeps the mapping tied to explicit stimulus names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent builds a global stimulus catalog from all session `WallName` values, maps each trial’s `WallName` to an integer category, and repeats that category across the four exported samples for that trial.

ii. 
```python
stimulus_values = stimulus_catalog(behavior_by_session)
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
```

```python
"visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
```

iii. The notes list visual stimulus category as an exported output. No separate justification is written for preferring `WallName` over `stim_id`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTrind` and `LickPos`. The agent does not use `LickFr` or `LickTime` for the exported licking channel.

ii. 
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_pos_trial = lick_positions[lick_trials == trial_idx]
```

iii. The notes say trial outputs were represented on 1 m position bins, so licking was also converted into that positional format.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the agent marks a bin as 1 if any lick position from that trial falls inside the corresponding 1 m interval. If any lick is at or beyond the last edge, the final bin is forced to 1.

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

iii. The notes justify only the shared 4-bin representation, not this exact binarization rule.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by trial and by the same four corridor bins used for neural interpolation, not by frame time.

ii. 
```python
output_trial_partial = {
    "visual_stimulus_category": ...,
    "licking": lick_trial,
    "position_bin": np.arange(4, dtype=np.int16),
    "running_speed_continuous": speed_interp,
}
```

iii. `CONVERSION_NOTES.md` says all exported trial variables were represented directly on the four ordered 1 m corridor bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is nominally derived from `ft_Pos`, but after export it is represented only by the four fixed bin identities rather than thresholding each retained raw frame.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The notes justify a 4-bin corridor representation for the entire export. There is no separate discussion of position labels because the bins are built into the export grid itself.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent does not compute a separate time series from raw positions after resampling. Instead, once each trial is represented at the 5/15/25/35 cm bin centers, the position output is just `[0, 1, 2, 3]` for every trial.

ii. 
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The justification is implicit in the notes’ decoder-facing design: the export axis itself is the set of four corridor bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The spatial bins are defined by edges `[0, 10, 20, 30, 40]` cm, corresponding to four 1 m bins over the 4 m textured corridor. However, the position output is not thresholded frame-by-frame; the categories are hard-coded to those four bins after interpolation.

ii. 
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says the trial representation corresponds to corridor positions `0-1 m`, `1-2 m`, `2-3 m`, and `3-4 m`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned trivially because the neural data themselves are stored at those four position bins.

ii. 
```python
neural_interp = interpolate_features(..., target_positions=POSITION_BIN_CENTERS)
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The notes justify using the same four-bin trial representation for neural activity and all time-varying variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed` on retained trial frames, with `ft_Pos` used to interpolate onto the exported bin centers.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
...
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. No separate prose justification is provided beyond the general choice to export continuous variables on the same four bins as neural activity.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent interpolates framewise running speed onto the four spatial bin centers for each trial, stores those continuous values temporarily, and later discretizes them.

ii. 
```python
"running_speed_continuous": speed_interp,
```

```python
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
```

iii. The notes mention global running-speed quartile edges in the sanity checks, reflecting the decision to pool all speeds before defining the 4 categories.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. After pooling all interpolated running-speed values across all sessions/trials/bins, the agent computes the 25th, 50th, and 75th percentiles and uses those global quartile thresholds to assign each sample to one of four speed bins.

ii. 
```python
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)
```

```python
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. The notes explicitly report “Global running-speed quartile edges,” and this choice follows the decoder instruction that the four bins should each correspond to 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed uses the same retained trial frames and the same interpolation onto the four corridor-bin centers as the neural data.

ii. 
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. The agent’s notes justify a common decoder-facing representation in which continuous variables and neural activity share the same four samples per trial.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles duplicated behavior exports by merging them and checking a fixed list of fields for equality. `NaN` frame trial indices are converted to `-1`. Missing fields in behavior-view comparisons are skipped. Unmapped retinotopy neurons are discarded. Trials with no retained usable frames are dropped. No broader imputation is performed.

ii. 
```python
if field not in template or field not in other_data:
    continue
if not _arrays_match(template[field], other_data[field]):
    raise ValueError(...)
```

```python
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` emphasizes merge validation across behavior views. The trajectory also records that the agent explicitly patched `ft_trInd` handling after a smoke test exposed `NaN` casting warnings.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading/concatenating huge `spks` arrays, computing per-session variance rankings on retained frames, and looping over every trial to interpolate neural/activity variables.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```

```python
for trial_idx in range(int(beh["ntrials"])):
    ...
    neural_interp = interpolate_features(...)
```

iii. The trajectory repeatedly describes full-session conversion as the long step because it had to scan all 89 large recordings and down-project each into the decoder format.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loop in `build_single_session`, the per-row loops in `interpolate_features`, the per-region ranking loop in `pick_neurons_by_variance`, and the lick-bin comprehension.

ii. 
```python
for trial_idx in range(int(beh["ntrials"])):
    ...
```

```python
for row in range(values.shape[0]):
    collapsed[row] = ...
...
for row in range(collapsed.shape[0]):
    interp[row] = np.interp(...)
```

iii. No explicit optimization note appears in the agent’s documentation; this follows directly from the code structure.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly interpolates different variables using the same `trial_positions` and `POSITION_BIN_CENTERS`, scans large behavior dictionaries multiple times, and performs a two-pass running-speed workflow: first collecting continuous speeds everywhere, then looping again to discretize them.

ii. 
```python
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
interpolate_features(positions=trial_positions, values=ft_speed[frame_idx], target_positions=POSITION_BIN_CENTERS)[0]
```

```python
all_speed_values.append(session_data["speed_values"])
...
full_output = [finalize_outputs(partials, speed_edges) for partials in output_partials]
```

iii. No separate justification is documented. The repeated interpolation is a side effect of exporting several different channels on the same 4-bin support.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps temporary `running_speed_continuous` arrays only to turn them into quartile bins later, pools `speed_values` globally only to discard them after quantile computation, computes `raw_counts`/`trial_stats`/`frame_periods` mostly for metadata and logs, and exports `position_bin` as a deterministic `[0,1,2,3]` channel that contains no trial-specific information.

ii. 
```python
output_trial_partial = {
    ...,
    "running_speed_continuous": speed_interp,
}
```

```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The notes explicitly say the 4-bin representation was chosen for decoder formatting, and the trajectory says extra summaries were gathered for sanity checks and documentation rather than for the decoder itself.
