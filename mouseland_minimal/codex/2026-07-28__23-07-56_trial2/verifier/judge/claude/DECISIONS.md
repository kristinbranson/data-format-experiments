# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `data/spk/` (neural spike data as `*_neural_data.npy`), `data/beh/` (behavior data as `Beh_*.npy`), and `data/retinotopy/` (area labels as `*_trans.npz`). It enumerates all 89 spike files to define sessions, then loads all behavior files and merges behavior views that correspond to the same physical session (stripping `_swap1`/`_swap2` suffixes). Each session is processed sequentially, loading its spike, behavior, and retinotopy data on demand.

ii.
```python
def convert_dataset(max_neurons_per_session):
    spk_session_keys = sorted(
        [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
        key=natural_key,
    )
    behavior_views = load_behavior_views()
    ...
    behavior_by_session = {
        key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
    }
```

```python
def load_behavior_views():
    views = defaultdict(list)
    for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for view_key, view in beh_dict.items():
            base_key = strip_swap_suffix(view_key)
            views[base_key].append(...)
    return views
```

iii. The AI documented in CONVERSION_NOTES.md that the 99 behavior keys map to 89 physical recordings due to swap1/swap2 views and repeated entries across analysis files. The merge step verifies consistency of timing and trial annotations across views. This approach is consistent with how the reference code accesses data through `exp_info` and `Beh` dictionaries keyed by session identifiers.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from session keys by parsing the first component (e.g., `DR10` from `DR10_2022_07_12_1`). All unique subjects are sorted alphabetically and assigned integer indices.

ii.
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    ...
    return {"subject": subject, ...}

subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI's CONVERSION_NOTES.md states 19 subjects were found, matching the paper's Methods ("89 recordings in 19 mice"). The subject splitting is derived from the session key naming convention which encodes mouse name, date, and block.

## 1-c. How are the data split into sessions?

i. Sessions correspond 1:1 with spike data files in `data/spk/`. Each `*_neural_data.npy` file defines one session. The AI found 89 sessions total.

ii.
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```

iii. The CONVERSION_NOTES.md confirms 89 sessions matching the paper's 89 recordings. The session definition is straightforward: one neural data file = one session.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavior data field `ntrials`. For each session, the code iterates over trial indices from 0 to `ntrials-1`. For each trial, it finds frames belonging to that trial using `ft_trInd`, filtered by running (`ft_move > 0`) and corridor (`ft_CorrSpc`) masks.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
    ...

def trial_frame_indices(beh, n_frames, trial_idx):
    ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
    ...
    ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
    ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]
```

iii. This is consistent with the reference code's trial indexing via `ft_trInd` and the paper's statement that "We only considered timepoints during running for analysis."

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if (a) no running-in-corridor frames exist for that trial, or (b) the maximum position reached in the trial is less than the last bin center (35 decimeters = 3.5 m). Sessions with fewer than 2 usable trials raise an error.

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

iii. The CONVERSION_NOTES.md reports 0 empty running trials were dropped across all sessions. The filtering ensures interpolation to the 4 spatial bin centers is valid (needs data up to at least 35 dm). There is no explicit trial count filtering from the reference code applied (e.g., the reference `lick_response` function's 200-trial cap is not applied here, which is appropriate since that was for behavioral analysis figures, not for the neural decoder).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from deconvolved spike traces stored in `*_neural_data.npy` files under the `spks` key, which contains per-plane arrays that are concatenated across imaging planes.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```

iii. This matches the reference code's `load_spk` function: `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)` and the paper's statement that "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The AI spatially interpolates neural activity. For each trial, it takes only running-in-corridor frames, gets their positions (`ft_Pos`), and interpolates the spike data onto 4 spatial bin centers at positions [5, 15, 25, 35] decimeters. It uses a custom `interpolate_features` function that averages duplicate positions, then uses `np.interp` for linear interpolation. The result is stored as float16.

ii.
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

```python
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)

def interpolate_features(positions, values, target_positions):
    ...
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row])
    return interp
```

iii. The reference code uses `spk_pos_interp` which interpolates using `scipy.interpolate.interp1d` with `fill_value='extrapolate'` onto cumulative position (60 bins per corridor including grey space). The AI's approach differs in: (1) using `np.interp` instead of `scipy.interpolate.interp1d`, (2) interpolating onto 4 bins instead of 60, (3) using within-corridor position instead of cumulative position, (4) operating trial-by-trial with a running+corridor mask instead of on all moving frames concatenated. These differences reflect the decoder task's 4-bin requirement. The AI's approach also applies float16 compression.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: (1) only neurons with valid visual cortex area codes are kept (V1, mHV, lHV, aHV), and (2) if more than 512 neurons pass, a variance-based selection with per-region balancing is applied to cap at 512 neurons per session.

ii.
```python
VISUAL_AREA_CODES = {
    "mHV": {0, 1, 2, 9},
    "aHV": {3, 4},
    "lHV": {5, 6},
    "V1": {8},
}

def pick_neurons_by_variance(spk, area_codes, running_corridor_mask, max_neurons):
    ...
    if len(valid_indices) <= max_neurons:
        ...
        return valid_indices, region_idx
    ...
    per_region_quota = max_neurons // len(BRAIN_REGIONS)
    ...
```

iii. The visual cortex area code mapping matches the reference code's `neu_area_ID` function exactly. The 512-neuron cap is documented in CONVERSION_NOTES.md as an intentional format-driven deviation for decoder trainability. The reference code does not apply a neuron cap; it uses all visual cortex neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI aligns by corridor position rather than time: each trial's data represents activity at 4 spatial positions within the corridor, starting from corridor entry (position 0) to end (position 40 dm). Since the VR speed is fixed at 60 cm/s when running, spatial position is approximately proportional to time since corridor entry.

ii.
```python
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)

neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The CONVERSION_NOTES.md states the alignment event is "corridor entry (trial start)." The spatial binning approach is consistent with the reference code's position-interpolated representation (`spk_pos_interp`), which also uses spatial position rather than time. The constant VR speed (60 cm/s) ensures spatial bins map to approximately equal time bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is not organized in temporal bins but in spatial bins. Each trial has 4 "time points" corresponding to 4 spatial bins of 1 meter each. The AI sets `time_bin_size` to `(1.0 / 0.60) * 1000.0` milliseconds (approximately 1666.67 ms), interpreting this as the time to traverse 1 meter at the constant VR speed of 60 cm/s. No temporal rebinning is applied; instead, spatial interpolation replaces temporal binning.

ii.
```python
metadata = {
    ...
    "time_bin_size": float((1.0 / 0.60) * 1000.0),
    ...
    "off_end": float(4.0 / 0.60),
    "binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
    ...
}
```

iii. The original calcium imaging frame rate is ~3.17 Hz (frame period ~0.315 s). The reference code interpolates into 60 positional bins per 6-meter corridor, giving ~10 cm per bin. The AI's 4 spatial bins of 1 m each is a much coarser resolution, as required by the decoder task instructions. The `time_bin_size` metadata value is a derived approximation based on constant VR speed rather than the actual imaging frame period.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh["SoundTime"]` (time of sound cue delivery) and `beh["ft"]` (timestamps for each neural frame).

ii.
```python
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. `SoundTime` is described in the data dictionary as "time of sound cue" and `ft` as "time stamp for each neural frame." The difference gives the time remaining until the sound cue, converted from MATLAB datenum (days) to seconds by multiplying by 86400.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the time to sound cue is computed as `(SoundTime - ft) * 86400` for each frame, giving seconds until the sound cue (positive before, negative after). This is then spatially interpolated onto the 4 bin centers.

ii.
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
...
input_trial = np.vstack([
    interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
    ...
])
```

iii. The computation is straightforward: time difference converted to seconds, then interpolated to spatial bins. The sign convention means positive values indicate the cue hasn't happened yet, negative values mean it has passed.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is interpolated onto the same 4 spatial bin centers as the neural data, using the same trial frame positions. This ensures exact alignment with the neural data.

ii.
```python
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. Same spatial interpolation method as neural data ensures alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session key, which encodes the date (e.g., `DR10_2022_07_12_1`). The AI computes the number of days since the subject's first recording date.

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

iii. The day of training is computed as the integer number of days since the subject's first session in the dataset. This value is constant per session (all trials in a session get the same value).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each session, the day index is computed as `(session_date - first_date_for_subject).days`. This is broadcast as a constant across all 4 spatial bins in a trial.

ii.
```python
np.full(4, subject_day_value, dtype=np.float32)
```

iii. The value is a simple integer (number of days), constant per session, replicated across spatial bins. No normalization or scaling is applied.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does NOT include "Environment type" as a decoder input. The instructions list only 4 decoder inputs: Time to sound cue, Day of training, Time since trial start, and Reward availability. Environment type is not among them.

ii.
```python
"input_names": [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_availability",
],
```

iii. The AI followed the instruction specification exactly, which does not include Environment type as a decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable - Environment type is not included as a decoder input in the AI's conversion.

ii. N/A

iii. N/A

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `beh["ft"]` (frame timestamps) and `beh["Trial_start_time"]` (time when the animal enters each corridor).

ii.
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. `Trial_start_time` is described in the data dictionary as "time when animal enters each corridor" and `ft` as "time stamp for each neural frame."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, computed as `(ft - Trial_start_time) * 86400` giving seconds since corridor entry. Then spatially interpolated onto the 4 bin centers.

ii.
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Straightforward time difference in seconds from trial start, interpolated to match spatial binning.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same spatial interpolation onto 4 bin centers as the neural data.

ii.
```python
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Aligned through shared spatial interpolation targets.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh["isRew"]`, a boolean array indicating whether each trial is a rewarded trial.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. `isRew` is described in the data dictionary as "boolean value indicating if the trial is a reward trial." The value is 1.0 for rewarded trials and 0.0 for unrewarded trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value for the trial is converted to float (0.0 or 1.0) and broadcast as a constant across all 4 spatial bins.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No additional processing beyond type conversion. The instructions specify "1 if in rewarded corridor, 0 if not, discrete, per-trial," which is exactly what `isRew` provides.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh["WallName"]`, which gives the stimulus name for each trial (e.g., "leaf1", "circle1").

ii.
```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
output_trial_partial = {
    "visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
    ...
}
```

iii. `WallName` is described in the data dictionary as "name of stimuli in each corridor."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A global catalog of all unique stimulus names is built across all sessions, sorted naturally. Each name is mapped to an integer ID. The stimulus ID is constant across all 4 spatial bins within a trial.

ii.
```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)

stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
```

iii. The output_values for this variable lists all unique stimulus names found. The instructions say "e.g. circle1, leaf2, etc., per-trial" - the AI correctly treats it as a per-trial categorical value replicated across spatial bins.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh["LickTrind"]` (trial index of each lick) and `beh["LickPos"]` (position of each lick within the corridor).

ii.
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
...
lick_pos_trial = lick_positions[lick_trials == trial_idx]
```

iii. `LickTrind` is "trial stamp of each lick" and `LickPos` is "positional stamp of each lick" per the data dictionary.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick positions are extracted. For each of the 4 spatial bins (defined by edges [0, 10, 20, 30, 40]), a binary value is set to 1 if any lick falls within that bin, 0 otherwise. Licks at position >= 40 are assigned to the last bin.

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

iii. The instructions specify "Licking, binary, time-varying. 0 = not licking, 1 = licking." The AI converts this to a binary per-spatial-bin representation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned spatially: lick positions are binned into the same 4 spatial bins as the neural data (0-10, 10-20, 20-30, 30-40 decimeters).

ii.
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
lick_trial = np.array([
    int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) & (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
    for bin_idx in range(4)
])
```

iii. Alignment is by spatial position rather than time, matching the neural data's spatial binning.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position in corridor is not derived from raw data variables per se - it is the spatial bin index itself (0, 1, 2, 3), since the data is organized by spatial position.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Since the data is binned spatially, the position output is simply the bin index [0, 1, 2, 3] for each trial.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing is needed - the bin index is simply `np.arange(4)`. Since data is organized by spatial bins, position is trivially known.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The instructions say "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins, time varying." The AI's approach makes this trivial since the data is already spatially binned.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 categories: bin 0 (0-1m), bin 1 (1-2m), bin 2 (2-3m), bin 3 (3-4m). No thresholding is needed since this is the inherent structure of the spatial binning.

ii.
```python
"output_values": [
    ...
    ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
    ...
]
```

iii. The 4 equal 1-meter bins match the instruction requirement exactly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is inherently aligned because both neural data and position share the same 4-bin spatial structure. The position output at bin i simply indicates the spatial location of the corresponding neural data at bin i.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Alignment is trivial under the spatial binning approach.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh["ft_RunSpeed"]`, the running speed for each neural frame.

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

iii. `ft_RunSpeed` is described in the data dictionary as "running speed for each neural frame."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed at each frame is spatially interpolated onto the 4 bin centers. The continuous speed values from all sessions are pooled to compute global quartile edges (25th, 50th, 75th percentiles). Speed is then discretized into 4 bins using these edges.

ii.
```python
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)

def finalize_outputs(session_output_partial, speed_edges):
    ...
    speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. The instructions specify "Running speed discretized into 4 bins, each corresponding to 25% of the data." The AI correctly uses quartiles of the pooled speed distribution to define bin edges.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded using global quartile edges (q25=16.592, q50=28.701, q75=43.168 cm/s). `np.digitize` assigns each speed value to one of 4 bins: [0] below q25, [1] q25-q50, [2] q50-q75, [3] above q75.

ii.
```python
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. The output values are labeled `["speed_q1", "speed_q2", "speed_q3", "speed_q4"]`. The quartile-based binning ensures approximately 25% of data per bin as specified.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 4 spatial bin centers as the neural data, ensuring exact alignment.

ii.
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0]
```

iii. Same spatial interpolation approach as neural data ensures alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) NaN values in `ft_trInd` are detected and those frames are excluded. (2) Trials with no running-in-corridor frames are skipped. (3) Trials that don't reach position 35 dm are skipped. (4) Duplicate behavior views are merged after consistency checking. (5) `ft_trInd` values are rounded to integers via `np.rint()` to handle floating point issues. (6) Lick positions at the corridor boundary (>=40) are assigned to the last bin.

ii.
```python
ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```

```python
if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
    lick_trial[-1] = 1
```

iii. The CONVERSION_NOTES.md reports 0 empty running trials dropped, suggesting data quality was high. The NaN handling in `ft_trInd` is prudent since the data dictionary notes NaN means "outside the behavior recorded."

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each session's large spike data file (`*_neural_data.npy`, up to ~4.3 GB), (2) The neuron variance ranking step which requires computing variance across all running corridor frames for all valid neurons, (3) The per-trial spatial interpolation loop over all trials and neurons.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
...
spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
```

iii. The trajectory shows the full conversion took several minutes for 89 sessions. The AI processes sessions sequentially with explicit `gc.collect()` calls to manage memory.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The `interpolate_features` function loops over rows (neurons) for both collapsing duplicates and interpolation. (2) The lick binning uses a Python list comprehension with explicit per-bin checks. (3) The trial loop in `build_single_session` processes one trial at a time, but the interpolation could potentially be batched.

ii.
```python
# Row-by-row interpolation loop
for row in range(collapsed.shape[0]):
    collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts
for row in range(collapsed.shape[0]):
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row])
```

```python
# Per-bin lick check
lick_trial = np.array([
    int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) & (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
    for bin_idx in range(4)
])
```

iii. The row-by-row loops in `interpolate_features` over neurons (up to 512) could potentially be vectorized using 2D interpolation or batch operations, though `np.interp` only supports 1D inputs.

## 12-c. What processing does the code repeat multiple times?

i. (1) The `area_code_to_region_name` function is called multiple times for the same neuron during variance ranking and region index construction. (2) `interpolate_features` is called separately for neural data, time_to_sound, time_since_start, and speed - each call recomputes position sorting and unique positions from the same trial positions. (3) The behavior views are loaded and merged upfront for all sessions, even though sessions are processed one at a time.

ii.
```python
# Called once during pick_neurons_by_variance and again for region_idx
region_name = area_code_to_region_name(area_codes[i])
...
region_idx = np.array(
    [BRAIN_REGIONS.index(area_code_to_region_name(area_codes[i])) for i in selected],
)
```

iii. The repeated position sorting in `interpolate_features` is a minor inefficiency since it processes the same trial positions for different value arrays.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `frame_periods` computation (median frame period across all sessions) is computed upfront but only used in metadata - it requires loading `ft` arrays for all sessions. (2) The behavior merge process loads and verifies many fields (like `ft_RunCum`, `ft_PosCum` etc.) that are never used in the conversion. (3) The `running_corridor_mask` is computed twice - once for the variance ranking and implicitly again inside `trial_frame_indices` for each trial. (4) The `interpolate_features` function computes full interpolation for the position output, but position bins are trivially `np.arange(4)` and don't need interpolation.

ii.
```python
# Frame periods computed upfront but only used for metadata
frame_periods = []
for key in spk_session_keys:
    ft = np.asarray(behavior_by_session[key]["ft"], dtype=np.float64)
    if len(ft) > 1:
        frame_periods.append(float(np.median(np.diff(ft) * SECONDS_PER_DAY)))
```

iii. The position bin output is trivially `[0,1,2,3]` for every trial since data is spatially binned - no computation is needed. This is the most clearly "unnecessary" processing, as the decoder will always see position as perfectly predictable from the bin index.
