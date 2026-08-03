# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from `Imaging_Exp_info.npy`, then iterates over all `sup_*` group entries (7 supervised groups). For each session, it loads behavior data from `Beh_<group>.npy` files and neural data via `utils.load_spk()` (which loads `{mname}_{datexp}_{blk}_neural_data.npy` and concatenates the `spks` list). Retinotopy data is loaded from `{mname}_{datexp}_trans.npz`. Only the rewarded task cohort (`sup_*` groups) is included; naive and unsupervised cohorts are excluded.

ii.
```python
def load_exp_info():
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def load_behavior(group_name):
    path = os.path.join(BEH_DIR, f"Beh_{group_name}.npy")
    return np.load(path, allow_pickle=True).item()

# In convert_dataset():
exp_info = load_exp_info()
catalog = attach_session_days(build_session_catalog(exp_info))
# ...
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
```

iii. The AI states in CONVERSION_NOTES.md: "I restricted the export to the rewarded task cohort because the decoder input specification requires a meaningful per-trial `reward_availability` variable. The unsupervised and naive cohorts do not have rewarded corridors."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `mname` field in the experiment metadata entries. A sorted list of unique subject names is built, and each session is mapped to its subject via `subject_to_index`.

ii.
```python
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
# ...
subject_idx.append(subject_to_index[session["subject"]])
```

iii. The AI's CONVERSION_NOTES says: "This yields 28 unique imaging sessions, 5 mice." The subject identity comes directly from the `mname` field in the experiment info metadata.

## 1-c. How are the data split into sessions?

i. The AI builds a session catalog by iterating over all `sup_*` groups, creating a unique session ID from `{mname}_{datexp}_{blk}`. When duplicate recordings appear across multiple groups (e.g., the same mouse/date/block in both `sup_train1_before_learning` and `sup_test1`), the AI deduplicates by keeping only the entry from the highest-priority group (earliest in `SUP_GROUP_PRIORITY`). This yields 28 unique sessions from 33 metadata rows.

ii.
```python
SUP_GROUP_PRIORITY = [
    "sup_train1_before_learning", "sup_train1_after_learning", "sup_test1",
    "sup_train2_before_learning", "sup_train2_after_learning", "sup_test2", "sup_test3",
]

def build_session_catalog(exp_info):
    session_candidates = defaultdict(list)
    for group_name in SUP_GROUP_PRIORITY:
        for entry in exp_info[group_name]:
            base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            behavior_key = base_session_id
            if "stimtype" in entry:
                behavior_key = f"{behavior_key}_{entry['stimtype']}"
            session_candidates[base_session_id].append({
                "group_name": group_name, "behavior_key": behavior_key, "entry": entry,
            })
    catalog = []
    for base_session_id in sorted(session_candidates):
        candidates = sorted(
            session_candidates[base_session_id],
            key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
        )
        canonical = candidates[0]
        # ...
```

iii. From CONVERSION_NOTES: "The raw metadata contains 33 supervised analysis rows but only 28 unique task recordings. Duplicates such as `sup_test1` and `sup_train2_before_learning` were deduplicated by recording id (`mouse_date_block`)."

## 1-d. How are the data split into trials?

i. The number of trials per session comes from `beh["ntrials"]`. The AI iterates over `range(ntrials)` for each session. Neural data is split into trials via the `utils.spk_pos_interp()` interpolation function, which reshapes the continuous recording into `(n_neurons, n_trials, n_bins)`.

ii.
```python
ntrials = int(beh["ntrials"])
# ...
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
# ...
for trial_idx in range(ntrials):
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. The AI relies on the behavior dict's `ntrials` field and the reference `spk_pos_interp` interpolation to split continuous data into per-trial format.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All trials from each session are included. The only filtering occurs at the neuron level (d-prime threshold and visual cortex region membership) and at the session/cohort level (only supervised/rewarded task groups).

ii.
```python
for trial_idx in range(ntrials):
    # All trials are included, no filtering
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. The AI does not mention any trial-level filtering in CONVERSION_NOTES. The reference code also does not appear to filter individual trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the deconvolved Suite2p spike traces stored in `{mname}_{datexp}_{blk}_neural_data.npy`, loaded via `utils.load_spk()`. This function concatenates the `spks` list from the saved numpy dict.

ii.
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
```

iii. From CONVERSION_NOTES: "Uses the deconvolved Suite2p traces in `data/spk/*_neural_data.npy`."

## 2-b. How is the `neural` data processed?

i. The neural data is processed in several steps: (1) Only running frames are kept (`ft_move > 0`), (2) the running-frame spike data is interpolated onto a uniform spatial grid using `utils.spk_pos_interp()` with the accumulated position, (3) only the first 40 bins (out of 60, corresponding to the 4m corridor) are kept, (4) data is cast to float16.

ii.
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. From CONVERSION_NOTES: "Uses running-only frames: `beh['ft_move'][:nframes] > 0`" and "Uses accumulated-position interpolation exactly in the style of `utils.spk_pos_interp(...)`."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: (1) Only neurons in visual cortex regions (V1, mHV, lHV, aHV) are kept, excluding `outside_visual_cortex`. (2) A d-prime selectivity score is computed between familiar rewarded and familiar unrewarded corridor frames during running, and neurons with `|d'| >= 0.3` are preferred. The top neurons are selected with balanced sampling across regions, capped at 512 per session.

ii.
```python
DP_THRESHOLD = 0.3
MAX_NEURONS = 512

def select_neurons(dp, region_idx, max_neurons):
    candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
    selected = []
    # ... balanced selection across V1, mHV, lHV, aHV ...
    # fallback: if not enough neurons above threshold, also include below-threshold visual cortex neurons
```

iii. From CONVERSION_NOTES: "Compute d' between familiar rewarded and familiar unrewarded corridors using running-only corridor frames. Prefer neurons with `|d'| >= 0.3`, matching the paper's selective-neuron threshold."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to corridor entry (trial start). The `utils.spk_pos_interp()` function interpolates spike data onto a spatial grid based on accumulated position. Position bin 0 corresponds to corridor entry. The first 40 bins (4m corridor) are kept.

ii.
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. From CONVERSION_NOTES: "Alignment event: corridor entry / trial start." and "Exported analysis window: first 40 bins only, corresponding to the 4 m corridor."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is spatially binned at 10 cm resolution (0.1 m per bin), giving an implied time bin of 10 cm / 60 cm/s = 166.667 ms. No additional temporal rebinning is applied. The spatial binning is inherited from the reference code's interpolation grid.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0  # = 166.667 ms
```

iii. From CONVERSION_NOTES: "Bin width: 10 cm. Implied time bin: 10 cm / 60 cm s^-1 = 0.1666667 s = 166.6667 ms."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh["SoundPos"]`, which gives the position (in spatial bins) at which the sound cue occurs for each trial. The current time bin position is derived from the bin index.

ii.
```python
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
cue_position = float(sound_positions[trial_idx])
```

iii. The AI uses `SoundPos` from the behavior data, which gives the spatial position of the sound cue per trial.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `cue_position / 6.0 - time_since_start`, where `time_since_start` is the time at each spatial bin (bin_index / 6.0 seconds). Both the cue position and current position are converted to time by dividing by 6.0 (the VR speed in bins/second, since 60 bins / 10 seconds). This gives a signed value: positive before the cue, negative after.

ii.
```python
time_since_start = position_units / 6.0  # position_units = np.arange(40)
# ...
cue_position = float(sound_positions[trial_idx])
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    # ...
])
```

iii. The AI converts the cue position to time by dividing by 6.0 (spatial bins per second at VR speed), then subtracts the current time to get time-to-cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are on the same 40-bin spatial grid. Each bin index corresponds to a spatial position and an implied time from trial start. The time-to-cue is computed at each bin position, so it is naturally aligned with the neural data bins.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)  # 0..39
time_since_start = position_units / 6.0
# time_to_cue at each bin = cue_time - bin_time
```

iii. Alignment is implicit through the shared spatial grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session date (`entry['datexp']`) in the experiment info metadata. The AI computes elapsed days since the first session date for each subject.

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

iii. From CONVERSION_NOTES: "`day_of_training`: elapsed days since the first rewarded task imaging session for that mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, sessions are sorted by date. The first session's date is the reference (day 0). For each subsequent session, the number of elapsed calendar days from the first session is computed. This value is constant across all time bins within a trial.

ii.
```python
session["training_day_index"] = float((session_date - first_date).days)
# In the trial loop:
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32)
```

iii. The AI uses calendar day difference, not ordinal session number.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the spatial bin index. Each bin corresponds to a 10 cm position step, and at the VR speed of 0.6 m/s, each bin is 1/6 second apart.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. This is computed directly from the bin grid, not from any raw data variable. It represents the implicit time since corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Simple computation: `bin_index / 6.0` gives time in seconds from corridor entry. Ranges from 0.0 (bin 0) to 6.5 seconds (bin 39).

ii.
```python
time_since_start = position_units / 6.0  # [0.0, 0.167, 0.333, ..., 6.5]
```

iii. The value 6.0 comes from the VR speed: 60 bins / 10 sec, or equivalently 0.1 m / (0.6 m/s) per bin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Both share the same 40-bin spatial grid, so alignment is inherent. Bin index i in the neural data corresponds to time i/6.0 seconds.

ii.
```python
# Same 40-bin grid for both neural and input
time_since_start = position_units / 6.0
```

iii. Alignment is by construction on the shared spatial grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh["isRew"]`, which is a per-trial boolean array indicating whether the corridor is rewarded.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
# ...
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. The AI uses the `isRew` field from the behavior data directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value is cast to float (1.0 or 0.0) and broadcast across all 40 time bins for the trial. It is per-trial constant.

ii.
```python
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. From CONVERSION_NOTES: "`reward_availability`: 1 for rewarded familiar corridors, 0 otherwise."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh["WallName"]` (per-trial stimulus name) and `beh["UniqWalls"]` (list of unique stimulus names across all sessions).

ii.
```python
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
stimulus_names = collect_stimulus_names(catalog)  # sorted unique names across all sessions
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
# ...
trial_stimulus = str(trial_wall_names[trial_idx])
np.full(N_POSITION_BINS_CORRIDOR, stimulus_to_index[trial_stimulus], dtype=np.int16)
```

iii. Stimulus names are collected globally and sorted with natural sort, then mapped to integer indices.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique stimulus names are collected across all sessions from `beh["UniqWalls"]`, sorted with natural sort order, and assigned integer indices. Each trial's `WallName` is mapped to its integer index. The value is constant across all 40 bins.

ii.
```python
def collect_stimulus_names(catalog):
    stimulus_names = set()
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
    return sorted(stimulus_names, key=natural_sort_key)
```

iii. This produces 14 unique stimulus categories across the dataset.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh["LickPos"]` (lick positions in spatial bin units) and `beh["LickTrind"]` (trial index for each lick event).

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. The AI uses the lick position and trial assignment data from the behavior dict.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick positions belonging to that trial are selected using `LickTrind`. Lick positions are binned into the 40-bin corridor grid. Any bin containing at least one lick gets a value of 1; otherwise 0. Licks outside the corridor range [0, 40) are excluded.

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
# ...
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. This produces a binary time-varying signal aligned to the spatial grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick positions are in the same spatial bin coordinate system as the interpolated neural data (0 to 39 for the 4m corridor). Flooring the lick position gives the corresponding bin index.

ii.
```python
indices = np.floor(valid_positions).astype(int)
lick_bins[np.clip(indices, 0, nbins - 1)] = 1
```

iii. Both share the same 40-bin spatial grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Not derived from raw data variables per se. Position bins are deterministic: each of the 40 spatial bins is assigned to one of 4 equal 1-meter position categories (10 bins per meter).

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
# [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3]
```

iii. Position is constructed from the grid itself. Bins 0-9 = 0-1m, 10-19 = 1-2m, 20-29 = 2-3m, 30-39 = 3-4m.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A fixed mapping from spatial bin index to position category: `np.repeat(np.arange(4), 10)`. Every 10 bins maps to one 1-meter corridor segment.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. This is a direct construction, not derived from behavioral data.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The corridor is divided into 4 equal 1-meter bins. Each 10-cm spatial bin is assigned to one of the 4 categories: 0 (0-1m), 1 (1-2m), 2 (2-3m), 3 (3-4m).

ii.
```python
position_values = [f"{start}-{start + 1}m" for start in range(4)]
# ["0-1m", "1-2m", "2-3m", "3-4m"]
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Matches the instruction: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Both are on the same 40-bin spatial grid. Bin index directly determines position category.

ii.
```python
# Same 40-bin grid
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Alignment is inherent by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh["ft_RunSpeed"]`, which is a per-frame running speed signal.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
```

iii. The raw frame-level running speed is filtered to running-only frames and then interpolated onto the spatial grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. (1) Running speed is filtered to running-only frames (`ft_move > 0`), (2) interpolated onto the 60-bin spatial grid using `spk_pos_interp()`, (3) trimmed to the first 40 corridor bins, (4) discretized into quartiles computed globally across all sessions and trials.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
all_speed_values.append(interp_speed.reshape(-1))
# After all sessions:
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. From CONVERSION_NOTES: "running_speed_bin: quartiles computed over all interpolated corridor time bins in the full export."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all speed values across all sessions/trials/bins. Then `np.digitize` assigns each speed value to one of 4 bins (0-3). The quartile edges ensure each bin contains approximately 25% of the data globally.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. From CONVERSION_NOTES: "Running-speed quartile edges: 15.0726, 24.9147, 37.9999." Matches the instruction: "Running speed discretized into 4 bins, each corresponding to 25% of the data."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The running speed is interpolated onto the same spatial grid as the neural data using the same `spk_pos_interp()` function and the same accumulated position values. Both are on the 40-bin corridor grid.

ii.
```python
# Same interpolation approach as neural data:
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. Speed uses the same spatial interpolation pipeline as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI checks for non-finite `SoundPos` values and raises an error if found. The corridor length is verified to be exactly 60 bins. Lick positions outside the corridor range are excluded. The AI adds a small epsilon (1e-12) to the d-prime denominator to avoid division by zero. No other explicit handling of missing data (NaN values in neural data, etc.) is implemented.

ii.
```python
if not np.isfinite(cue_position):
    raise ValueError(f"... non-finite SoundPos on trial {trial_idx}")
if corridor_length != N_POSITION_BINS_TOTAL:
    raise ValueError(f"... expected corridor length {N_POSITION_BINS_TOTAL}, found {corridor_length}")
# d-prime: epsilon to avoid div-by-zero
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
# Licks: filter out of range
valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
```

iii. The AI's approach is conservative: raise errors for unexpected conditions rather than silently handling them.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading large neural data files (47k-90k neurons per session), (2) the `spk_pos_interp()` interpolation which processes each neuron individually via `interp_value()`, (3) computing d-prime across all neurons using nanmean/nanstd on large arrays.

ii.
```python
# Loading large neural arrays
spk = utils.load_spk(...)  # 47k-90k neurons x many frames
# Interpolation
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
# D-prime computation
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
```

iii. The agent trajectory mentions waiting for interpolation: "The interpolation step is the dominant cost."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for constructing input/output arrays could be partially vectorized. The `bin_licks` function is called once per trial in a loop. The `select_neurons` function uses Python loops over region candidates.

ii.
```python
# Per-trial loop
for trial_idx in range(ntrials):
    # Each trial constructs input_trial, output_trial individually
    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. However, the dominant cost is the `spk_pos_interp` call which is already delegated to the reference utils module.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data is loaded twice for each session: once in `collect_stimulus_names()` to gather all unique wall names, and again in the main conversion loop. The `natural_sort_key` function is called repeatedly for sorting.

ii.
```python
def collect_stimulus_names(catalog):
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]  # first load
        stimulus_names.update(map(str, beh["UniqWalls"]))

# Then in convert_dataset main loop:
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]  # second load
```

iii. The duplicate loading is a minor inefficiency since behavior files are small compared to neural data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The full 60-bin interpolation is computed but only the first 40 bins are kept (`[:N_POSITION_BINS_CORRIDOR]`). The last 20 bins (reward zone) are discarded. (2) Neurons outside visual cortex are loaded and used for d-prime computation but never selected. (3) The `session_info` metadata stores per-session stimulus counts and detailed provenance info that the decoder does not use.

ii.
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]  # discard last 20 bins
```

iii. Computing all 60 bins before trimming is consistent with the reference approach but wastes computation on bins that are ultimately discarded.
