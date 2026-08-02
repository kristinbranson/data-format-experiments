# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, which contains entries organized by experimental group (e.g., `sup_train1_before_learning`). It iterates over the 7 `sup_*` groups, loading behavior data from `Beh_<group>.npy` files, neural spike data via `utils.load_spk()`, and retinotopy assignments from `<mouse>_<date>_trans.npz` files. Only the supervised/rewarded task cohort (`sup_*` groups) is included; unsupervised and naive cohorts are excluded.

ii.
```python
# Loading experiment info
exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()

# Loading behavior for a group
def load_behavior(group_name):
    path = os.path.join(BEH_DIR, f"Beh_{group_name}.npy")
    return np.load(path, allow_pickle=True).item()

# Loading spike data
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)

# Loading retinotopy
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
```

iii. From CONVERSION_NOTES.md: "I restricted the export to the rewarded task cohort because the decoder input specification requires a meaningful per-trial `reward_availability` variable. The unsupervised and naive cohorts do not have rewarded corridors, so including them would collapse that input and make the converted dataset inconsistent with the requested decoder task."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in the experiment info entries. A sorted list of unique subject names is built from all sessions in the catalog, and each session is mapped to a subject index.

ii.
```python
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
# ...
subject_idx.append(subject_to_index[session["subject"]])
```

iii. The AI followed the same subject identification used throughout the reference code, which uses `ndb['mname']` as the subject identifier. The CONVERSION_NOTES.md reports 5 mice in the final dataset.

## 1-c. How are the data split into sessions?

i. Sessions are identified by a composite key `{mname}_{datexp}_{blk}` (mouse name, date, block). When the same recording appears in multiple `sup_*` groups (e.g., a session in both `sup_train1_before_learning` and `sup_test1`), it is deduplicated by keeping only the first match in `SUP_GROUP_PRIORITY` order. The behavior key may additionally include a `stimtype` suffix for sessions with swap stimuli.

ii.
```python
SUP_GROUP_PRIORITY = [
    "sup_train1_before_learning", "sup_train1_after_learning",
    "sup_test1", "sup_train2_before_learning",
    "sup_train2_after_learning", "sup_test2", "sup_test3",
]

def build_session_catalog(exp_info):
    session_candidates = defaultdict(list)
    for group_name in SUP_GROUP_PRIORITY:
        for entry in exp_info[group_name]:
            base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            behavior_key = base_session_id
            if "stimtype" in entry:
                behavior_key = f"{behavior_key}_{entry['stimtype']}"
            session_candidates[base_session_id].append({...})
    catalog = []
    for base_session_id in sorted(session_candidates):
        candidates = sorted(
            session_candidates[base_session_id],
            key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
        )
        canonical = candidates[0]
        catalog.append({...})
    return catalog
```

iii. From CONVERSION_NOTES.md: "The raw metadata contains 33 supervised analysis rows but only 28 unique task recordings. Duplicates such as `sup_test1` and `sup_train2_before_learning` were deduplicated by recording id (`mouse_date_block`)."

## 1-d. How are the data split into trials?

i. Trial count is taken from `beh['ntrials']`. The continuous spike data is interpolated onto a position grid of 60 bins (one per 10cm across 6m of VR path) using `utils.spk_pos_interp()`, which returns an array of shape `(n_neurons, n_trials, 60)`. Each trial then corresponds to one slice along the trial dimension. Only the first 40 bins (the 4m corridor) are kept.

ii.
```python
ntrials = int(beh["ntrials"])
# ...
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
# ...
for trial_idx in range(ntrials):
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. The reference code's `get_interpPos_spk` function performs the same interpolation to `(neurons, trials, 60)`. The AI reuses `utils.spk_pos_interp` directly, matching the reference processing pipeline.

## 1-e. How are trials filtered based on quality controls?

i. No trials are explicitly filtered or excluded. All trials within each session are included in the output. The AI does not apply any trial-level quality control beyond what is implicit in the running-only frame selection during interpolation.

ii.
```python
for trial_idx in range(ntrials):
    # All trials are processed; no filtering condition
    trial_stimulus = str(trial_wall_names[trial_idx])
    # ...
```

iii. The CONVERSION_NOTES.md does not mention any trial filtering. The reference code similarly processes all trials within a session for the interpolation step. Some reference figure code uses odd/even trial splits for cross-validation, but those are analysis-specific, not data curation steps.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the deconvolved Suite2p calcium traces stored in `*_neural_data.npy` files. These are loaded via `utils.load_spk()`, which concatenates the `spks` arrays from the saved dictionary.

ii.
```python
# From utils.py:
def load_spk(db, root=''):
    fn = '%s_%s_%s_neural_data.npy'%(db['mname'],db['datexp'],db['blk'])
    spk_path = os.path.join(root, fn)
    spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']],0)
    return spk
```

iii. From methods.txt: "All our analyses were based on deconvolved fluorescence traces." The CONVERSION_NOTES.md states: "Uses the deconvolved Suite2p traces in `data/spk/*_neural_data.npy`."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes these processing steps: (1) only running frames are selected (`ft_move > 0`), (2) the running-frame data is spatially interpolated onto a 60-bin position grid using `utils.spk_pos_interp()`, (3) only the first 40 bins (4m corridor portion) are retained, (4) data is cast to float16 for storage efficiency.

ii.
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. From CONVERSION_NOTES.md: "Uses running-only frames: `beh['ft_move'][:nframes] > 0`" and "Uses accumulated-position interpolation exactly in the style of `utils.spk_pos_interp(...)`." This matches the reference code's data processing notebook.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in three stages: (1) only neurons within grouped visual cortex areas (V1, mHV, lHV, aHV) are considered; neurons labeled `outside_visual_cortex` are excluded, (2) a d-prime selectivity metric is computed between the familiar rewarded and familiar unrewarded corridor activity, and neurons with `|d'| >= 0.3` are preferred, (3) a maximum of 512 neurons per session is enforced with balanced sampling across the 4 brain regions (128 per region minimum), then remaining slots filled by highest |d'| neurons.

ii.
```python
DP_THRESHOLD = 0.3
MAX_NEURONS = 512
MIN_AREA_NEURONS = MAX_NEURONS // len(AREA_SELECTION_ORDER)  # 128

def select_neurons(dp, region_idx, max_neurons):
    candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
    selected = []
    for region_name in AREA_SELECTION_ORDER:
        region_code = REGION_TO_INDEX[region_name]
        region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
        order = region_candidates[np.argsort(abs_dp[region_candidates])[::-1]]
        for neuron_idx in order[:MIN_AREA_NEURONS]:
            selected.append(int(neuron_idx))
    # Fill remaining slots with highest |d'| visual cortex neurons
    if len(selected) < max_neurons:
        # ... (fills from remaining candidates by |d'|)
    return np.sort(selected)
```

iii. From CONVERSION_NOTES.md: "Prefer neurons with `|d'| >= 0.3`, matching the paper's selective-neuron threshold used throughout the figure code" and "Cap each session at 512 neurons." The 512 cap and balanced sampling are the AI's own design choices for tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). The position interpolation via `spk_pos_interp()` maps running-frame neural activity onto a uniform position grid using accumulated VR position. Each trial's data starts at position 0 (corridor entry) and spans 40 bins (4m of corridor). This is a spatial alignment, not a temporal one, but since the VR moves at a constant speed (60 cm/s), there is a direct mapping between position and time.

ii.
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]
# Each trial starts at bin 0 = corridor entry
```

iii. From CONVERSION_NOTES.md: "Alignment event: corridor entry / trial start" and "Reference interpolation grid: 60 bins across the 6 m trial path."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is defined as the time to traverse one 10cm spatial bin at the fixed VR speed of 60 cm/s: `0.1m / 0.6m/s = 166.67 ms`. No temporal rebinning is applied; the data is spatially interpolated directly from running frames. Each spatial bin implicitly corresponds to a fixed time interval due to the constant VR speed.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0  # 166.6667 ms
```

iii. From CONVERSION_NOTES.md: "Bin width: 10 cm" and "Implied time bin: 10 cm / 60 cm s^-1 = 0.1666667 s = 166.6667 ms."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundPos']` (position of sound cue delivery per trial) and the position grid index.

ii.
```python
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
# ...
cue_position = float(sound_positions[trial_idx])
```

iii. From CONVERSION_NOTES.md: "`time_to_sound_cue_s`: signed cue time minus current bin time."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound cue position (in 10cm bin units) is converted to time by dividing by 6.0 (since VR speed = 60cm/s = 6 bins/s). Then the current bin's time is subtracted, yielding the time remaining until (or elapsed since) the sound cue. Positive values mean the cue has not yet occurred; negative means it has passed.

ii.
```python
time_since_start = position_units / 6.0  # position_units = np.arange(40)
# ...
cue_position / 6.0 - time_since_start,  # time to sound cue for each bin
```

iii. The AI's trajectory notes confirm they used position-based time conversion consistent with the constant VR speed assumption.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both the neural data and the time-to-cue input share the same 40-bin position grid, so they are inherently aligned. Each bin index maps to the same position along the corridor.

ii.
```python
# Both neural and input use the same 40-bin grid:
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,  # 40 bins
    # ...
])
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))  # 40 bins
```

iii. No explicit temporal alignment step is needed because all variables are constructed on the same position grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` (date of experiment) field in the experiment info entries.

ii.
```python
def attach_session_days(catalog):
    by_subject = defaultdict(list)
    for session in catalog:
        by_subject[session["subject"]].append(session)
    for subject_sessions in by_subject.values():
        subject_sessions.sort(key=lambda item: parse_date(item["date"]))
        first_date = parse_date(subject_sessions[0]["date"])
        for index, session in enumerate(subject_sessions):
            session_date = parse_date(session["date"])
            session["training_day_index"] = float((session_date - first_date).days)
```

iii. From CONVERSION_NOTES.md: "`day_of_training`: elapsed days since the first rewarded task imaging session for that mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by date. The number of calendar days elapsed since that mouse's first recording date is computed. This value is constant across all time bins within a trial and across all trials within a session.

ii.
```python
session["training_day_index"] = float((session_date - first_date).days)
# ...
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32),
```

iii. The AI chose calendar days rather than session rank (ordinal day count). Both `training_day_index` (calendar days) and `training_day_rank` (ordinal) were computed, but `training_day_index` was used for the decoder input.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did not include an "Environment type" input. The decoder task instructions specify exactly four inputs: Time to sound cue, Day of training, Time since trial start, and Reward availability. "Environment type" is not among them.

ii. N/A - not implemented.

iii. From the trajectory: The AI restricted to the rewarded task cohort (`sup_*` groups only), meaning all included data comes from the same experimental paradigm. The closest concept to "environment type" is `reward_availability`, which distinguishes rewarded from unrewarded corridors within the task.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. Not applicable - this input was not included in the converted dataset.

ii. N/A

iii. The "Environment type" concept is partially captured by the `reward_availability` input (rewarded vs. unrewarded corridor) and the `visual_stimulus_category` output (which identifies the specific corridor stimulus).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the position bin index. Since the VR moves at a constant speed, position maps linearly to time from trial start (corridor entry).

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)  # 0..39
time_since_start = position_units / 6.0  # seconds: 0, 0.167, 0.333, ..., 6.5
```

iii. This is a deterministic computation from the spatial bin index, not directly from a raw data variable. The constant VR speed of 60 cm/s maps each 10cm bin to a fixed time increment.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each position bin index (0 to 39) is divided by 6.0 (bins per second at 60cm/s) to convert to seconds. This yields a linearly increasing time signal from 0 to approximately 6.5 seconds.

ii.
```python
time_since_start = position_units / 6.0
# Included in input as:
input_trial = np.vstack([
    # ...
    time_since_start,
    # ...
])
```

iii. No complex processing is involved; this is a straightforward linear mapping from position to time.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned by construction on the same 40-bin position grid. Bin 0 corresponds to corridor entry (time = 0), and each subsequent bin is one position step (10cm = 166.67ms) later.

ii. Same grid as neural data; no additional alignment needed.

iii. Both neural and input variables share the position-interpolated grid, ensuring alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a boolean array indicating whether each trial is a rewarded trial.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
# ...
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32),
```

iii. From CONVERSION_NOTES.md: "`reward_availability`: 1 for rewarded familiar corridors, 0 otherwise."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value for each trial is converted to float (1.0 or 0.0) and repeated across all 40 position bins to make it time-varying (constant within trial). No further processing is applied.

ii.
```python
np.full(
    N_POSITION_BINS_CORRIDOR,
    float(trial_is_rewarded[trial_idx]),
    dtype=np.float32,
),
```

iii. The instructions specify "1 if in rewarded corridor, 0 if not, discrete, per-trial", which is exactly what `beh['isRew']` provides.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']` (stimulus name per trial) and `beh['UniqWalls']` (unique stimulus names in the session).

ii.
```python
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
# ...
trial_stimulus = str(trial_wall_names[trial_idx])
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
# ...
np.full(N_POSITION_BINS_CORRIDOR, stimulus_to_index[trial_stimulus], dtype=np.int16),
```

iii. From CONVERSION_NOTES.md: "`visual_stimulus_category`: constant across time within each trial." The stimulus names are collected globally across all sessions and sorted naturally.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique stimulus names across all sessions are collected and sorted using natural sort order. Each trial's stimulus name is mapped to an integer index in this global list. The index is repeated across all 40 time bins (per-trial constant).

ii.
```python
def collect_stimulus_names(catalog):
    stimulus_names = set()
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
    return sorted(stimulus_names, key=natural_sort_key)

stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
```

iii. The 14 unique stimulus categories (circle1-3, leaf1-3, leaf1_swap1/2, rock1-2, wood1, wood1_swap2, wood2, wood5) span the different experimental phases.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickPos']` (position of each lick event) and `beh['LickTrind']` (trial index of each lick event).

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. From CONVERSION_NOTES.md: "`licking`: binary per 10 cm bin, derived from `LickPos` within the corridor."

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events belonging to that trial are identified using `LickTrind`. Their positions are filtered to the corridor range (0 to 40 bins), floored to integer bin indices, and used to set binary indicators. Any bin with at least one lick gets value 1.

ii.
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    if lick_positions.size == 0:
        return lick_bins
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
    if valid_positions.size == 0:
        return lick_bins
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins

lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. This matches the instructions: "Licking, binary, time-varying. 0 = not licking, 1 = licking."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick positions are in the same 10cm-bin coordinate system as the neural data's position grid. Both use the corridor position as the shared axis, so lick events at position X map to bin floor(X) in the same 40-bin grid.

ii. The `LickPos` values directly index into the same position grid used for neural interpolation.

iii. No additional alignment is needed; both share the corridor position coordinate.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position in corridor is not derived from a raw data variable per se; it is a deterministic function of the bin index in the 40-bin grid. Each group of 10 consecutive bins maps to one 1-meter spatial bin.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
# Result: [0,0,0,0,0,0,0,0,0,0, 1,1,..., 2,2,..., 3,3,...]
```

iii. This is constructed from the position grid, not from raw behavioral data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are grouped into 4 equal segments of 10 bins each (1 meter each). Each bin is assigned the integer category corresponding to its meter-long segment.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
# Used directly in output:
output_trial = np.vstack([
    # ...
    position_indices,
    # ...
])
```

iii. This matches the instructions: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The 4m corridor is divided into 4 equal 1m segments. Bins 0-9 -> category 0 (0-1m), bins 10-19 -> category 1 (1-2m), bins 20-29 -> category 2 (2-3m), bins 30-39 -> category 3 (3-4m). This is a hard partition, not a threshold.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
position_values = [f"{start}-{start + 1}m" for start in range(4)]
# output_values[2] = ["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. The equal-length spatial bins follow the instructions exactly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Aligned by construction. The position categories are derived from the same 40-bin position grid that the neural data is interpolated onto. Bin 0 of neural data corresponds to bin 0 of the position output.

ii. Same grid; no additional alignment.

iii. All variables share the position-interpolated grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, which provides the running speed for each neural frame.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
```

iii. Running speed is frame-level data that is interpolated onto the same position grid as the neural data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The frame-level running speed is filtered to running-only frames (same as neural data), then spatially interpolated onto the 60-bin position grid using `utils.spk_pos_interp()`, and truncated to 40 bins. The interpolated speed values across all sessions are pooled to compute global quartile edges. Each bin's speed is then discretized into 4 quartile categories.

ii.
```python
# Interpolate speed onto position grid
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR]

# Compute global quartile edges
speed_values = np.concatenate(all_speed_values)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

# Digitize
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. From CONVERSION_NOTES.md: "Running-speed quartile edges: 15.0726, 24.9147, 37.9999."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all interpolated speed values across the entire dataset. `np.digitize` assigns each speed value to one of 4 bins (0-3), where bin 0 is the lowest quartile and bin 3 is the highest.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. This matches the instructions: "Running speed discretized into 4 bins, each corresponding to 25% of the data."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated using the same `spk_pos_interp` function with the same accumulated position and corridor length, producing a speed value for each of the 40 position bins. This matches the neural data's position grid.

ii.
```python
# Same interpolation as neural data:
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. Alignment is guaranteed by using the same interpolation pipeline.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI raises explicit errors for unexpected conditions (non-finite SoundPos, missing familiar corridor frames, wrong corridor length) rather than silently handling them. No sessions in the actual data triggered these errors. Lick bins with no licks return all zeros. The d' denominator includes a small epsilon (1e-12) to prevent division by zero.

ii.
```python
# Non-finite sound position check
if not np.isfinite(cue_position):
    raise ValueError(...)

# Corridor length check
if corridor_length != N_POSITION_BINS_TOTAL:
    raise ValueError(...)

# Empty lick handling
if lick_positions.size == 0:
    return lick_bins  # all zeros

# d' epsilon to prevent division by zero
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
```

iii. The reference code's `dprime` function does not include an epsilon, allowing NaN for neurons with zero variance. The AI's epsilon prevents this but changes the result for such neurons.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading the full spike matrices (`utils.load_spk`) which concatenates large `.npy` arrays (up to ~90k neurons), (2) the spatial interpolation (`utils.spk_pos_interp`) which loops over neurons one at a time and calls `scipy.interpolate.interp1d` for each, (3) computing d' over all running corridor frames for neuron selection.

ii.
```python
# Spike loading - huge arrays
spk = utils.load_spk(...)  # loads and concatenates all neuron planes

# Interpolation - loops over neurons
def spk_pos_interp(raw_spk=[], accum_pos=[], corridorLen=[], new_shape=[]):
    for s in range(raw_spk.shape[0]):  # loop through neurons
        spk_resh.append(np.reshape(interp_value(raw_spk[s, :], accum_pos/corridorLen, linPos), ...))
```

iii. The reference notebook notes data processing "takes about 8 hours." The AI's per-session processing mirrors this pattern.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the per-trial loop (lines 370-408) that constructs input/output arrays. Since `position_indices`, `time_since_start`, and constant-valued inputs are the same across trials, they could be pre-computed as full trial-by-time matrices. Additionally, `bin_licks` is called per trial but could be vectorized using scatter operations across all trials at once.

ii.
```python
for trial_idx in range(ntrials):
    # This per-trial construction is repeated identically for many variables
    input_trial = np.vstack([
        cue_position / 6.0 - time_since_start,
        np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], ...),
        time_since_start,
        np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), ...),
    ])
```

iii. The underlying `spk_pos_interp` in utils.py also has a per-neuron loop that is inherently inefficient but was not modified by the AI.

## 12-c. What processing does the code repeat multiple times?

i. Behavior data is loaded twice for each session: once in `collect_stimulus_names()` to gather all unique stimulus names, and again in the main conversion loop. This is functionally necessary but doubles I/O for behavior files.

ii.
```python
# First load: stimulus name collection
def collect_stimulus_names(catalog):
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))

# Second load: main conversion loop
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
```

iii. The full spike data is loaded, used for d' computation, and then the selected subset is re-extracted and interpolated. The full spike matrix is needed for d' but only selected neurons are interpolated, so this is somewhat necessary.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code interpolates neural data onto all 60 position bins but only keeps the first 40 (`:N_POSITION_BINS_CORRIDOR`). The last 20 bins (grey space between corridors) are computed and then discarded. Similarly, running speed is interpolated onto all 60 bins before being truncated to 40. The `training_day_rank` (ordinal session index) is computed for each session but never used in the final output.

ii.
```python
# Full 60-bin interpolation, then truncation to 40:
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]  # discard 20 bins

# training_day_rank computed but unused:
session["training_day_rank"] = int(index)
```

iii. The 60-bin interpolation is inherited from the reference code's `get_interpPos_spk` which uses `n_bins=60`. The reference code sometimes uses the grey-space bins (e.g., for normalization in `Get_coding_direction`), but the converter does not need them.
