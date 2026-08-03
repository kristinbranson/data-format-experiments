# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter uses the spike files in `/app/data/spk` as the authoritative list of physical sessions, loads every behavior dictionary from every `Beh_*.npy` file, merges duplicate behavior views that map to the same physical session key, and loads retinotopy per session when building the final export. Trials are then constructed by iterating `range(beh["ntrials"])` inside each merged session.

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
for trial_idx in range(int(beh["ntrials"])):
    frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
```

iii. The justification in `CONVERSION_NOTES.md` and trajectory step 37 is that the raw release has 89 physical recordings in `spk` but 99 behavior keys because some behavior exports expose duplicated `_swap1` and `_swap2` views, so the code rebuilds the 89 physical sessions before trialization.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed directly from the session key prefix before the first underscore, for example `TX60` from `TX60_2021_06_07_1`. Unique subject IDs are sorted, and each exported session gets a `subject_idx` pointing into that sorted subject list.

ii. 
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    return {
        "subject": subject,
        ...
    }

subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_id[parsed[session_key]["subject"]])
```

iii. No separate subject-splitting rationale was given beyond treating the mouse ID encoded in the session key as the subject ID. This matches the way the original filenames and `Imaging_Exp_info.npy` entries identify mice.

## 1-c. How are the data split into sessions?

i. Sessions are defined one-to-one with spike files in `/app/data/spk`. Each `*_neural_data.npy` filename becomes one exported session key, and any multiple behavior views that map onto that key are merged into one session record.

ii. 
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)

behavior_by_session = {
    key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
}
```

iii. The explicit justification is in `CONVERSION_NOTES.md`: the paper reports 89 recordings, so the code treats the 89 spike files as the physical session list and merges behavior duplicates back onto that list.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavior field `ntrials` and the per-frame trial assignment `ft_trInd`. For each trial index, the converter collects frame indices whose rounded `ft_trInd` equals that trial and also satisfy the running/corridor mask.

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

iii. The justification is that the reference behavior exports already provide `ntrials` and `ft_trInd`, and the paper says analyses only considered running timepoints. The agent therefore used frame-level trial stamps plus the running/corridor mask.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has no running-in-corridor frames, or if its retained positions never reach the last exported spatial bin center (`35`, corresponding to the 3 to 4 m bin). A whole session is rejected if fewer than two usable trials remain.

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
...
if len(session_neural) < 2:
    raise ValueError(f"Session {session_key} has fewer than 2 usable trials after running-frame filtering")
```

iii. `CONVERSION_NOTES.md` explicitly justifies dropping trials without usable running corridor frames. The additional full-corridor coverage check is implied by the 4-bin export scheme rather than by the paper; no separate justification beyond needing all 4 bins was documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the deconvolved activity arrays stored under the `spks` key in each `*_neural_data.npy` file. Brain-region labels are taken from the matching retinotopy file’s `iarea` array.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. `CONVERSION_NOTES.md` says the agent intentionally followed the reference code path in `utils.py`, where `load_spk` concatenates the `spks` planes and `load_retino` reads `iarea`.

## 2-b. How is the `neural` data processed?

i. After concatenating all planes, the code filters neurons to selected visual-cortex regions, optionally caps the count to 512 by variance ranking with per-region balancing, extracts running-in-corridor frames trial by trial, and linearly interpolates each trial onto 4 fixed position-bin centers. The exported matrices are then cast to `float16`.

ii. 
```python
selected_neurons, region_idx = pick_neurons_by_variance(
    spk=spk,
    area_codes=area_codes,
    running_corridor_mask=running_corridor_mask,
    max_neurons=max_neurons,
)
spk_selected = spk[selected_neurons]
...
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. The documented justification is twofold: `CONVERSION_NOTES.md` says the agent wanted to keep only visual cortex as in the paper code, and it explicitly labels both the 4-bin representation and the 512-neuron cap as intentional decoder-format deviations.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with retinotopy labels outside the grouped visual-cortex regions are removed. If more than 512 visual-cortex neurons remain, only 512 are kept, chosen by running-frame variance with approximate region balancing across `V1`, `mHV`, `lHV`, and `aHV`.

ii. 
```python
def area_code_to_region_name(area_code: float):
    ...
    for region_name, codes in VISUAL_AREA_CODES.items():
        if area_code in codes:
            return region_name
    return None
```

```python
if len(valid_indices) <= max_neurons:
    return valid_indices, region_idx
...
variances = np.var(spk_valid, axis=1)
...
selected.extend(order[: min(per_region_quota, len(order))])
```

iii. `CONVERSION_NOTES.md` justifies the visual-cortex filter as matching reference analysis code. It separately notes that the deterministic 512-neuron cap was a trainability choice for the decoder harness, not something claimed to come from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial membership is anchored to corridor entry via `Trial_start_time`/`ft_trInd`, but the final per-trial neural time axis is not frame time from trial start. Instead, each trial is resampled onto four spatial positions inside the corridor: bin centers `5, 15, 25, 35`.

ii. 
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
...
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The agent justified this in `CONVERSION_NOTES.md` as a decoder-driven compromise: the task demanded 4 equal 1 m bins, so it represented each trial directly in that spatial basis even though the instructions described temporal alignment to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported metadata claims a `time_bin_size` of `1666.666...` ms, computed from 1 m of corridor travel at a nominal VR speed of 60 cm/s. No actual temporal rebinning is performed; the code does spatial interpolation onto 4 bins and then back-fills metadata with a nominal time equivalent.

ii. 
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),
"off_end": float(4.0 / 0.60),
"binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
```

iii. The justification comes from `CONVERSION_NOTES.md`: the agent explicitly states that the representation uses 4 spatial bins and treats the nominal VR speed as the basis for the reported time-bin metadata.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the per-trial sound time `SoundTime` and the per-frame timestamps `ft`.

ii. 
```python
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
...
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. No separate prose justification was given beyond using the raw time-stamp fields already provided in the behavior export.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the code subtracts frame time from trial-specific `SoundTime`, converts the result from MATLAB datenum days to seconds, and then interpolates that continuous value onto the 4 exported spatial bins.

ii. 
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. The agent did not document a separate rationale for this computation; it is a straightforward translation of the requested variable into seconds on the same exported grid as the neural data.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using exactly the same retained trial frames and the same 4 position-bin centers used for the neural interpolation.

ii. 
```python
input_trial = np.vstack(
    [
        interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
        ...
    ]
).astype(np.float32)
```

iii. The justification in `CONVERSION_NOTES.md` is that all continuous decoder inputs were placed on the same 4-bin trial grid as the neural activity.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each session key, not from a dedicated training-day field in the raw arrays. The code parses `YYYY_MM_DD` out of filenames such as `TX60_2021_06_07_1`.

ii. 
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    date_str = f"{year}_{month}_{day}"
    date_obj = datetime.strptime(date_str, "%Y_%m_%d").date()
```

iii. There is no explicit justification in the notes beyond needing a continuous “day of training” variable and having reliable session dates in the session keys.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the converter finds the earliest recording date and computes every session’s day value as the integer day difference from that first date. That scalar is then repeated across the 4 exported bins of every trial in that session.

ii. 
```python
for key, info in parsed.items():
    if subj not in first_day or info["date"] < first_day[subj]:
        first_day[subj] = info["date"]
...
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
```

```python
np.full(4, subject_day_value, dtype=np.float32)
```

iii. No explicit reference-backed justification was given. This appears to be the agent’s own choice for turning session dates into a continuous day index.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the per-frame timestamps `ft` and the per-trial start times `Trial_start_time`.

ii. 
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The notes justify trial alignment by corridor entry, and `Trial_start_time` is the direct behavior variable for that event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For every retained frame in a trial, the code subtracts `Trial_start_time` from `ft`, converts from days to seconds, and interpolates that continuous trace onto the 4 position bins.

ii. 
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. No separate justification was documented beyond putting all continuous inputs onto the same trial grid as the neural data.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by interpolation onto the same 4 position-bin centers used for the neural data, using the same retained running/corridor frames from the same trial.

ii. 
```python
input_trial = np.vstack(
    [
        ...,
        interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0],
        ...
    ]
).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` states that continuous task variables were interpolated onto the same 4-bin representation as the neural activity.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean behavior field `isRew`.

ii. 
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32),
```

iii. The behavior notebook and raw arrays already define `isRew` as the rewarded-corridor indicator, so the agent used it directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code converts the per-trial boolean `isRew` value to float and repeats it across all 4 bins of the trial.

ii. 
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No additional processing was justified or documented; this is a direct per-trial broadcast of the raw reward indicator.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial stimulus name stored in `WallName`, with the global list of categories built from all sessions’ `WallName` values.

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

iii. The trajectory shows the agent inspected the reference behavior notebook where `WallName` is defined as the stimulus identity per corridor, so it used that field directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code collects a global natural-sort catalog of stimulus names, maps each name to an integer ID, and repeats that ID across the 4 bins of a trial.

ii. 
```python
stimulus_values = stimulus_catalog(behavior_by_session)
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
...
"visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
```

iii. The justification is implicit: the decoder output must be categorical, and the raw stimulus labels are already discrete category names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickPos` and `LickTrind`, not from `LickTime`. The converter groups licks by trial and asks whether a lick occurred in each exported 1 m position bin.

ii. 
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
...
lick_pos_trial = lick_positions[lick_trials == trial_idx]
```

iii. The reference `get_lick_raster` function the agent inspected also organizes licking by trial and lick position, so the agent mirrored that spatial treatment.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the converter builds a 4-element binary vector indicating whether any lick fell inside each of the four corridor bins. If any lick position is beyond the last explicit edge, it forces the last bin to 1.

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

iii. The only documented rationale is consistency with the 4-bin spatial export. No separate argument was given for preferring lick position over lick time in the decoder output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned in the same spatial basis as the neural data: the four output elements correspond to the same four position bins that the neural activity was interpolated onto.

ii. 
```python
output_trial_partial = {
    ...,
    "licking": lick_trial,
    ...
}
```

iii. `CONVERSION_NOTES.md` says the exported trials are 4 ordered samples corresponding to 4 corridor bins, and licking follows that same representation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The intended source variable is per-frame corridor position `ft_Pos`, which is used to place retained frames within the corridor and to drive interpolation. However, the exported position output itself is then hard-coded as the bin indices `0,1,2,3` once the trial grid has already been fixed.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The justification in the notes is that trials are represented directly as four 1 m corridor bins, so position becomes a deterministic label on that grid rather than a separately estimated trace.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code does not re-discretize a continuous per-sample position trace after interpolation. Instead, it directly emits the fixed sequence `[0, 1, 2, 3]` for every trial because the whole export is already defined on those four position bins.

ii. 
```python
output_trial_partial = {
    ...,
    "position_bin": np.arange(4, dtype=np.int16),
    ...
}
```

iii. This was explicitly justified as a decoder-format choice in `CONVERSION_NOTES.md`: once the trial representation is four spatial bins, the position output is simply the identity of those bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are fixed equal-width bins with edges `[0, 10, 20, 30, 40]` and labels corresponding to `0_to_1m`, `1_to_2m`, `2_to_3m`, `3_to_4m`.

ii. 
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
...
"output_values": [
    ...,
    ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
    ...
]
```

iii. The justification is direct from the decoder task, which explicitly requested four equal 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by construction: each of the four position labels corresponds exactly to one of the four neural samples within a trial.

ii. 
```python
"position_bin": np.arange(4, dtype=np.int16),
...
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The notes justify this as a consequence of using a four-bin spatial trial representation throughout.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the per-frame behavior variable `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. The behavior notebook excerpt the agent inspected defines `ft_RunSpeed` as running speed for each neural frame, so the code uses that field directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each trial, the code takes running speed on retained running/corridor frames, interpolates it onto the 4 position bins, pools all such values across the full dataset, and then discretizes each trial’s interpolated values afterward using global quartile thresholds.

ii. 
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
...
all_speed_values.append(session_data["speed_values"])
...
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` documents the global quartile edges as a sanity check, which indicates the agent intentionally used pooled quartiles to satisfy the decoder requirement that the four bins each cover 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. After computing global quartile cut points, the code uses `np.digitize` to assign each interpolated speed sample to one of four categories.

ii. 
```python
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)
...
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. The justification is the decoder specification itself: “Running speed discretized into 4 bins, each corresponding to 25% of the data.”

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation onto the same 4 position bins used for the neural trial matrices, and then discretized on that same aligned grid.

ii. 
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. The justification is the same decoder-format choice described in `CONVERSION_NOTES.md`: all continuous variables were put onto the 4-bin representation used by the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code trims behavior frame arrays to the neural frame count, converts `NaN` `ft_trInd` values to `-1` so they are excluded from trial assignment, skips missing merge-check fields when reconciling duplicate behavior views, averages duplicate positions during interpolation, excludes `NaN` or unknown retinotopy area codes, and drops trials with no usable running/corridor frames. It also has a special case to mark the last lick bin if any lick lies past the last explicit bin edge.

ii. 
```python
ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```

```python
if field not in template or field not in other_data:
    continue
...
unique_pos, inverse = np.unique(positions, return_inverse=True)
...
if np.isnan(area_code):
    return None
```

iii. The notes justify only part of this explicitly: merging duplicate behavior views and dropping unusable running trials. The remaining small repairs are ad hoc implementation choices to make interpolation and trialization robust.

## 12-a. What are the most time-consuming steps of the code?

i. The slowest steps are loading the very large spike arrays, concatenating all imaging planes, computing neuron variances on running frames, and then interpolating neural activity separately for every trial. Because the code processes all 89 sessions and every kept trial, those neural interpolation loops dominate runtime.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
...
variances = np.var(spk_valid, axis=1)
...
for trial_idx in range(int(beh["ntrials"])):
    ...
    neural_interp = interpolate_features(
        positions=trial_positions,
        values=spk_selected[:, frame_idx],
        target_positions=POSITION_BIN_CENTERS,
    )
```

iii. The agent did not separately discuss runtime hotspots in the notes, but these are the clearly dominant costs in the implementation it wrote.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop in `build_single_session`, the per-region ranking loop in `pick_neurons_by_variance`, the row-wise averaging and interpolation loops inside `interpolate_features`, and the per-trial output finalization loop could all be vectorized further. The lick-bin comprehension is also small but repeatedly executed.

ii. 
```python
for trial_idx in range(int(beh["ntrials"])):
    ...
```

```python
for row in range(values.shape[0]):
    collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts

for row in range(collapsed.shape[0]):
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row]).astype(np.float32)
```

iii. No explicit justification was documented. This is simply where the written code is still loop-heavy.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly interpolates onto the same 4 position bins for neural activity and for each continuous input/output stream, repeatedly parses `np.asarray(beh[field])` inside the session builder, repeatedly scans behavior arrays to collect per-trial frame indices, and repeatedly maps retinotopy codes back to region names during neuron selection.

ii. 
```python
neural_interp = interpolate_features(...)
...
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
speed_interp = interpolate_features(...)
```

iii. The agent did not comment on this repetition, but it is visible in the implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest unnecessary work is building `running_speed_continuous` only to discard it after discretization, computing `time_bin_size` as if the bins were temporal, collecting extensive metadata ranges and source-view lists that the decoder never uses, and computing `position_bin` as a trivial deterministic output once the representation has already been fixed to 4 position bins. The code also loads all visual-cortex neurons only to discard all but 512.

ii. 
```python
output_trial_partial = {
    ...
    "running_speed_continuous": speed_interp,
}
...
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

```python
"position_bin": np.arange(4, dtype=np.int16),
...
"time_bin_size": float((1.0 / 0.60) * 1000.0),
```

iii. The only part the agent explicitly acknowledged is the 512-neuron cap and the 4-bin representation as format-driven compromises. The other discarded or decoder-irrelevant work follows directly from the code structure.
