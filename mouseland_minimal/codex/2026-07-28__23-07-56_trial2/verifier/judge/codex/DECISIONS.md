# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use `Imaging_Exp_info.npy` as a master index. Instead, it enumerated all spike-session files in `data/spk`, loaded every `Beh_*.npy` file under `data/beh`, grouped behavior entries by a base session key with `_swap1`/`_swap2` stripped off, and merged those behavior views into one behavior record per spike session.

ii. 
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
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
behavior_by_session = {
    key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
}
```

iii. In trajectory step 37 the AI says the behavior exports had 99 keys but only 89 physical recordings, so it "standardiz[ed] to the 89 actual recording sessions and merg[ed] paired swap annotations back into single sessions." Steps 67 and 434 repeat that it intentionally reconstructed sessions by merging behavior views onto the 89 physical recordings.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session key string itself. The prefix before the first underscore is parsed as the mouse name, and subjects are the sorted unique set of those names.

ii. 
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    return {"subject": subject, ...}
```
```python
subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The trajectory does not dwell on subject splitting separately; it treats session keys as the canonical identifiers after reconstructing the 89 recordings (steps 37, 67, 434). The subject extraction follows directly from that session-key convention.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the spike filenames in `data/spk`. Each `*_neural_data.npy` file becomes one physical session, and behavior entries with swap suffixes are merged onto that session key.

ii. 
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```
```python
def strip_swap_suffix(session_key: str):
    return re.sub(r"_swap[12]$", "", session_key)
```

iii. In step 37 the AI explicitly says it resolved the "99 keys vs 89 recordings" issue by treating swap entries as duplicated views of the same physical session. Step 434 summarizes the same choice as reconstructing "the paper’s 89 physical recordings from the 99 behavior views."

## 1-d. How are the data split into trials?

i. Trials are not kept as all corridor frames. The AI defines each trial as the subset of frames where `ft_trInd == trial_idx`, `ft_move > 0`, and `ft_CorrSpc` is true, then collapses those retained frames onto four fixed 1 m spatial bins. Trials with no retained running frames, or that do not reach the last spatial bin center, are dropped.

ii. 
```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ...
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]
```
```python
for trial_idx in range(int(beh["ntrials"])):
    frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
    if len(frame_idx) == 0:
        continue
    trial_positions = ft_pos[frame_idx]
    if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
        continue
```

iii. Step 48 says the AI rejected using all frames because it thought that produced pathological long pauses, and preferred `ft_move > 0` inside the corridor. Steps 67 and 143 say it intentionally exported each trial as "4 ordered 1 m spatial bins" instead of raw framewise trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with no moving corridor frames after masking, and also drops trials whose retained positions never reach the 35 cm bin center. It does not implement the reference solution’s 99th-percentile long-trial exclusion.

ii. 
```python
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue
```
```python
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    trial_stats["empty_running_trials"] += 1
    continue
```

iii. Step 48 says the AI believed moving-only corridor frames yielded the "stable 20-35 frame traversals the paper analyzes." The trajectory does not mention percentile-based long-trial filtering; instead it frames the QC choice as running-only masking plus requiring complete 4-bin traversals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the concatenated `spks` arrays in each session’s `*_neural_data.npy`, and neuron region labels come from `iarea` in the session’s retinotopy `.npz` file.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. Step 24 says the AI identified the key raw structures as trial-aligned behavior, neural traces concatenated across imaging planes, and one retinotopy area label per neuron. Step 434 repeats that it used the same visual-cortex area grouping as the reference code.

## 2-b. How is the `neural` data processed?

i. The AI first restricts neurons to visual cortex, then possibly downsamples them to 512 neurons per session with a variance-ranked, region-balanced rule. Within each trial it keeps only moving corridor frames and linearly interpolates neural activity onto the four spatial-bin centers at 5, 15, 25, and 35 dm, storing the result as `float16`.

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

iii. Steps 75 and 143 describe the two deliberate deviations the AI thought were necessary: averaging/interpolating within four 1 m bins and capping neurons per session with a deterministic rule so the decoder remained tractable. Step 434 repeats both choices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered to four visual-cortex groups (`V1`, `mHV`, `lHV`, `aHV`). If more than 512 survive, the AI keeps a balanced subset ranked by variance on running-in-corridor frames.

ii. 
```python
for neuron_idx, area_code in enumerate(area_codes):
    region_name = area_code_to_region_name(area_code)
    if region_name is not None:
        region_to_indices[region_name].append(neuron_idx)
```
```python
variances = np.var(spk_valid, axis=1)
...
selected.extend(order[: min(per_region_quota, len(order))])
```

iii. Step 75 says the export would stay compact by "capping neurons per session with a deterministic label-free rule." Step 143 calls the 512-neuron cap an explicit, intentional deviation from the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. The AI says the alignment event is corridor entry, but operationally each trial is represented by four spatial bins within the corridor after removing non-moving frames. Neural activity is therefore aligned to corridor position relative to trial start rather than preserved on the original imaging-frame time axis.

ii. 
```python
"temporal_alignment_event": "corridor entry (trial start)",
```
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. Step 67 says the AI would "bin trials over the four 1 m corridor segments" while aligning to corridor entry. Step 434 describes the output as "4 ordered 1 m spatial bins" aligned to corridor entry, showing that spatial binning replaced framewise temporal alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI does not keep the original imaging-frame resolution. It encodes each trial using four spatial bins and writes a nominal bin size of `1/0.60 s`, i.e. about `1666.7 ms`, based on traversing 1 m at 60 cm/s.

ii. 
```python
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),
"binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
```

iii. Steps 67, 75, and 143 make clear that the AI intentionally replaced framewise data with a four-bin representation. It justified this as a decoder-facing representation choice rather than preserving the paper’s native 3.17 Hz temporal bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from `SoundTime` and `ft`, not from `SoundFr`. It subtracts per-frame timestamps from the trial’s sound time, then interpolates those values by position.

ii. 
```python
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The trajectory does not justify this variable choice separately. The broader rationale in steps 67 and 75 is that continuous variables should be represented on the same four spatial bins as the neural data.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI converts the difference between `SoundTime` and `ft` from MATLAB days to seconds on the retained moving frames, then interpolates that continuous signal onto the four position-bin centers.

ii. 
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```
```python
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. Steps 67 and 75 state that continuous task variables were being averaged/interpolated into the four 1 m corridor bins. The trajectory does not mention any use of `SoundFr`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same retained moving-frame positions and the same four target corridor-bin centers as the neural interpolation.

ii. 
```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
...
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. Steps 67 and 434 describe a shared four-bin trial representation for neural data and decoder inputs/outputs. That is the AI’s stated alignment logic.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives day of training from the session key itself, specifically the parsed subject and calendar date embedded in the session name.

ii. 
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    date_obj = datetime.strptime(date_str, "%Y_%m_%d").date()
```
```python
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
```

iii. The trajectory does not discuss this variable directly. The underlying assumption follows the AI’s decision to use spike-session filenames as the canonical session index (steps 37 and 434).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the AI finds the earliest recording date and stores the elapsed calendar days since that first date. That scalar is then broadcast across the four bins of every trial from that session.

ii. 
```python
first_day[subj] = info["date"]
...
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
```
```python
np.full(4, subject_day_value, dtype=np.float32)
```

iii. No dedicated justification appears in the trajectory. The decision is implicit in the session-key parsing and the AI’s overall choice to use fixed four-bin trial arrays.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives this input from `Trial_start_time` and `ft`, rather than from `StartFr`.

ii. 
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The trajectory does not discuss this field separately. As with other inputs, the general rationale in steps 67 and 75 is to place continuous quantities on the same four-bin representation as the neural data.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI subtracts `Trial_start_time` from the per-frame timestamps `ft`, converts the difference from days to seconds on moving corridor frames, and interpolates the result onto the four corridor-bin centers.

ii. 
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```
```python
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. The trajectory again justifies this only at the representation level: continuous inputs were collapsed into four 1 m corridor bins (steps 67, 75, 143).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by being interpolated from the same retained frame positions onto the same four position bins as the neural data.

ii. 
```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Steps 67 and 434 indicate that the AI’s shared alignment space for neural and behavioral variables is the four-bin corridor representation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability comes directly from `isRew` for each trial.

ii. 
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. The trajectory does not call out this decision separately. It is a straightforward per-trial flag inserted into the common four-bin representation.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. There is no substantive transformation beyond converting the per-trial value to float and broadcasting it across the four bins of the trial.

ii. 
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No separate justification is given in the trajectory. This follows the general design choice to make every trial a length-4 array.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives stimulus labels from `WallName`.

ii. 
```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
```

iii. The trajectory does not discuss stimulus labeling in detail. The representation choice is visible in the code: it builds a catalog from all distinct `WallName` strings present across sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse wall names into the four base texture categories used by the reference. Instead, it builds a global sorted catalog of all unique `WallName` values and assigns each trial the integer id of its exact wall-name string, broadcasting that id across the four bins.

ii. 
```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)
```
```python
output_trial_partial = {
    "visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
    ...
}
```

iii. The trajectory does not provide an explicit justification for keeping all wall-name variants separate. Its general emphasis is on the decoder-facing four-bin format rather than on matching the reference’s texture collapsing.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickTrind` and `LickPos`, not from `LickFr`.

ii. 
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
```

iii. The trajectory does not isolate this choice, but it is consistent with the AI’s decision to represent trial progress spatially instead of framewise temporally (steps 67, 143, 434).

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial and each of the four 1 m bins, the AI marks the licking output as 1 if any lick position falls inside that bin and 0 otherwise. If a lick occurs at or beyond the final edge, it is forced into the last bin.

ii. 
```python
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

iii. The trajectory’s stated rationale is again representational: step 67 says trials are exported over four 1 m corridor segments, and step 434 says the outputs are aligned to those ordered spatial bins.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned to the neural data through the same four spatial bins, not through the original imaging frames.

ii. 
```python
output_trial_partial = {
    "visual_stimulus_category": ...,
    "licking": lick_trial,
    "position_bin": np.arange(4, dtype=np.int16),
    "running_speed_continuous": speed_interp,
}
session_neural.append(neural_interp)
```

iii. Steps 67 and 434 state that the AI’s exported trials are four spatial bins; licking and neural activity share that binned structure.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
trial_positions = ft_pos[frame_idx]
```

iii. The trajectory repeatedly frames the conversion around 1 m corridor segments (steps 67, 143, 434), so `ft_Pos` is used as the coordinate for that representation.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI does not threshold `ft_Pos` at every retained frame. Instead, once a trial survives the moving-frame and full-corridor checks, it simply emits the fixed sequence `[0, 1, 2, 3]`, one category per 1 m bin.

ii. 
```python
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    continue
```
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Step 67 says the AI was intentionally "bin[ning] trials over the four 1 m corridor segments." In that design, position becomes a deterministic ordered label sequence rather than a framewise readout from `ft_Pos`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are hard-coded as four equal 1 m bins spanning edges `[0, 10, 20, 30, 40]` dm, but after trial selection the actual output categories are just the fixed indices `0,1,2,3`.

ii. 
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The trajectory’s justification is the same fixed-bin design described in steps 67 and 143.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned with neural activity by making both share the same four spatial bins for each trial.

ii. 
```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
...
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Step 434 explicitly says the conversion "exports each trial as 4 ordered 1 m spatial bins," which is the common alignment used for both neural and position outputs.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. The trajectory does not discuss the raw variable separately. It is part of the same four-bin corridor representation described in steps 67 and 434.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Within each trial, the AI interpolates `ft_RunSpeed` from retained moving-frame positions onto the four spatial-bin centers. Across the whole dataset, it concatenates those interpolated values and computes global 25th, 50th, and 75th percentile thresholds.

ii. 
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```
```python
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
```

iii. The trajectory only justifies the shared four-bin representation (steps 67, 75, 143). It does not separately justify using global quantile thresholds rather than the reference’s per-session rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses `np.quantile` over all interpolated speed-bin values from all sessions to get three global cut points, then bins each trial’s four speed values with `np.digitize`.

ii. 
```python
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)
```
```python
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. No explicit rationale appears in the trajectory beyond wanting "4 bins" for decoder output. The thresholding rule is an implementation choice visible in the code.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned with neural data by being interpolated onto the same four corridor-bin centers as the neural activity.

ii. 
```python
neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS)
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. Steps 67 and 434 make clear that the AI’s universal alignment space is the four ordered 1 m corridor bins.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI explicitly handles `NaN` values in `ft_trInd` by remapping them to `-1`, checks merged behavior views for field-by-field consistency, errors out if a spike session has no matching behavior views, and drops trials with no usable moving frames. It truncates frame-based streams to `n_frames = spk.shape[1]`, but it does not use the reference’s explicit `LickFr < n_frames` safeguard because it derives licking from `LickTrind`/`LickPos` instead.

ii. 
```python
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```
```python
if not _arrays_match(template[field], other_data[field]):
    raise ValueError(...)
```
```python
missing_behavior = [key for key in spk_session_keys if key not in behavior_views]
if missing_behavior:
    raise KeyError(...)
```

iii. Step 84 says the AI found that `ft_trInd` contains `NaN` values outside behavior coverage and patched the conversion to handle that explicitly. Steps 37 and 77 show that it also considered behavior-view mismatches a major failure mode and added consistency checks around merging.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are session-level spike loading and concatenation, followed by per-trial interpolation and neuron-variance ranking on large sessions. The AI processes sessions one at a time to control memory while repeatedly reading large spike files.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```
```python
variances = np.var(spk_valid, axis=1)
...
neural_interp = interpolate_features(...)
```

iii. Step 75 says the "memory-heavy part" is done session-by-session so the script can traverse the 405 GB spike directory. Many later trajectory updates are simply waiting for the full-session pass to finish, confirming that large-file I/O dominated runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python loops could have been reduced: the per-neuron region assignment, the per-row interpolation loops inside `interpolate_features`, the per-trial loop over all trials, and the per-bin lick assignment loop.

ii. 
```python
for neuron_idx, area_code in enumerate(area_codes):
    ...
```
```python
for row in range(values.shape[0]):
    collapsed[row] = np.bincount(...)
...
for row in range(collapsed.shape[0]):
    interp[row] = np.interp(...)
```
```python
for trial_idx in range(int(beh["ntrials"])):
    ...
```

iii. The trajectory does not explicitly discuss vectorization opportunities, but step 77 says the main anticipated failure modes were interpolation and shape handling, which matches where the Python loops concentrate.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly converts arrays with `np.asarray`, repeatedly interpolates separate features for the same `trial_positions` (`neural`, `time_to_sound`, `time_since_start`, `speed`), and repeatedly resolves area-code-to-region-name mappings during neuron selection and metadata construction.

ii. 
```python
neural_interp = interpolate_features(...)
...
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
speed_interp = interpolate_features(...)
```
```python
[BRAIN_REGIONS.index(area_code_to_region_name(area_codes[i])) for i in selected]
```

iii. The trajectory frames this as an acceptable cost of the chosen four-bin representation rather than as a bug. Steps 67 and 75 show the AI deliberately chose interpolation-heavy processing to force all modalities into the same bin structure.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra bookkeeping that the decoder does not need, including merged source-view provenance, raw/selected neuron count summaries, frame-period summaries, and a separate sample dataset. During conversion it also materializes intermediate `output_partial` structures and continuous running-speed values solely so it can discretize them later.

ii. 
```python
merged["source_behavior_views"] = [...]
```
```python
output_trial_partial = {
    "visual_stimulus_category": ...,
    "licking": lick_trial,
    "position_bin": np.arange(4, dtype=np.int16),
    "running_speed_continuous": speed_interp,
}
```
```python
sample = make_sample_dataset(data, sample_session_count=args.sample_session_count)
```

iii. The trajectory makes clear that some of this was added for auditability and tractability rather than decoder semantics. Steps 143 and 163 mention documenting the intentional deviations and writing additional artifacts, while step 434 lists the extra sample and note files as part of the deliverables.
