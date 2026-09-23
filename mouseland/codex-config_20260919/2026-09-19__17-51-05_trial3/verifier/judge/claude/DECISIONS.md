# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It first reads the master index `Imaging_Exp_info.npy` and builds a deduplicated list of 89 physical sessions. Behavior files (`Beh_<group>.npy`) are loaded once per experiment group, and neural/retinotopy files are loaded per session.

ii.
```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
# ...
behavior_dict = np.load(BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True).item()
# ...
raw = np.load(spk_path, allow_pickle=True).item()
planes = raw["spks"]
# ...
with np.load(ret_path, allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The AI documented its loading strategy in CONVERSION_NOTES.md, noting behavior files are loaded once per group to avoid redundant I/O, and neural files are loaded and released one session at a time to manage memory.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment descriptor. The AI builds a sorted list of unique subjects and creates a lookup index for each session.

ii.
```python
subjects = sorted({r["db"]["mname"] for r in records})
subject_lookup = {name: i for i, name in enumerate(subjects)}
# ...
subject_idx.append(subject_lookup[record["db"]["mname"]])
```

iii. The experiment index already provides the mouse name per session, so no additional splitting logic is needed.

## 1-c. How are the data split into sessions?

i. A session is identified by the composite key `mname_datexp_blk`. Duplicate entries across experiment groups are deduplicated by tracking seen session IDs, resulting in 89 unique sessions.

ii.
```python
def session_id(db: dict) -> str:
    return f"{db['mname']}_{db['datexp']}_{db['blk']}"

# ...
if sid in seen:
    continue
seen.add(sid)
```

iii. The AI noted that the 142 experiment-group entries refer to only 89 unique physical recordings, and correctly deduplicates them.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` (frame-level trial index), `ft_CorrSpc` (corridor space mask), and additionally `ft_move > 0` (movement mask). For each trial, the AI selects only frames where the animal is in the textured corridor AND the VR is advancing.

ii.
```python
def selected_frames_by_trial(beh, sid):
    trial_id = np.asarray(beh["ft_trInd"])
    valid = (
        np.isfinite(trial_id)
        & np.asarray(beh["ft_CorrSpc"], dtype=bool)
        & (np.asarray(beh["ft_move"]) > 0)
    )
    selected = np.flatnonzero(valid)
    selected_trial = trial_id[selected].astype(np.int64)
    # ...
    frames = [selected[selected_trial == trial] for trial in range(ntrials)]
```

iii. The AI justified including the movement filter by noting that the reference code's neural analyses use `ft_CorrSpc & (ft_move > 0)` and that removing stationary frames prevents reward-consumption pauses from dominating the data.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter trials based on length. It keeps all 38,110 trials across all sessions. The only validation is that every trial must have at least 2 valid (moving, corridor) frames, which is enforced by raising an error otherwise.

ii.
```python
if np.any(lengths < 2):
    bad = np.flatnonzero(lengths < 2).tolist()
    raise ValueError(f"{sid}: trials with fewer than two valid frames: {bad}")
```

iii. The AI argued that because stationary frames are already removed, the movement filter naturally handles the problem of animals that stop for long periods, eliminating the need for trial-length-based filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`spk/<session_id>_neural_data.npy`), which contains a list of arrays per imaging plane. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
raw = np.load(spk_path, allow_pickle=True).item()
planes = raw["spks"]
# ...
with np.load(ret_path, allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The AI noted these are the released Suite2p deconvolved traces used by all analyses.

## 2-b. How is the `neural` data processed?

i. The AI concatenates imaging planes (filtering out non-visual-area neurons during concatenation for memory efficiency), then extracts the selected frames per trial and stores as float32.

ii.
```python
activity = np.empty((int(keep.sum()), nframes), dtype=np.float32)
raw_offset = 0
kept_offset = 0
for plane in planes:
    nplane = int(plane.shape[0])
    plane_keep = keep[raw_offset : raw_offset + nplane]
    nkeep = int(plane_keep.sum())
    activity[kept_offset : kept_offset + nkeep] = plane[plane_keep]
    # ...
# ...
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
```

iii. The AI stores neural data as float32 (not float16 as in the reference) and performs plane-by-plane filtering to avoid a full concatenation of all neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area: only neurons with `iarea` codes mapping to V1 (8), mHV (0,1,2,9), lHV (5,6), or aHV (3,4) are retained. Codes -1 and 7 are excluded as outside visual cortex. This retains 4,105,393 of 4,691,034 neuron-session units.

ii.
```python
region = np.full(len(iarea), -1, dtype=np.int16)
region[iarea == 8] = 0            # V1
region[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
region[np.isin(iarea, [5, 6])] = 2        # lHV
region[np.isin(iarea, [3, 4])] = 3        # aHV
keep = region >= 0
```

iii. The AI noted that Suite2p already performed cell classification upstream and that analysis-specific d-prime thresholds would leak the stimulus target, so only area filtering is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). Each trial's data starts at its first selected frame (moving, in-corridor) and ends at its last. Trials are variable length.

ii.
```python
for trial, frames in enumerate(frames_by_trial):
    ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
```

iii. The AI set `off_start=0.0` and `off_end=None` because trials start at corridor entry but have variable endpoints depending on running speed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of 3.17 Hz is preserved, giving a bin size of ~315.46 ms. However, because stationary frames are removed, timepoints within a trial may have temporal gaps.

ii.
```python
IMAGING_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / IMAGING_RATE_HZ
# ...
"time_bin_size": float(TIME_BIN_MS),
```

iii. The AI noted that the imaging frame is the finest resolution available and behavior streams are already aligned to it.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute timestamp of the sound cue for each trial) and `ft` (the absolute timestamp of each imaging frame).

ii.
```python
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
```

iii. The AI uses trial-level timestamps directly rather than frame-number-based interpolation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference between the cue timestamp and each frame's timestamp is computed, multiplied by 86400 to convert from MATLAB datenum (days) to seconds. The sign convention is positive before the cue, negative after.

ii.
```python
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
```

iii. The AI noted this gives "time to" the cue (positive before, negative after), matching the variable name.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the `ft` timestamps of the same selected frames used for neural data.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
```

iii. Frame-level alignment ensures temporal consistency.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `days` field in the experiment descriptor, falling back to `sess#` if `days` is not available.

ii.
```python
if "days" in db:
    day_value = float(db["days"])
    day_source = "days"
elif "sess#" in db:
    day_value = float(db["sess#"])
    day_source = "sess#"
```

iii. The AI used the released metadata fields rather than computing session order from dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The numeric value from the descriptor is used directly as a float, broadcast across all frames of each trial in the session.

ii.
```python
np.full(T, record["day_value"]),
```

iii. The AI noted this is the only released continuous per-session training-stage field.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (absolute timestamp of trial start) and `ft` (absolute timestamp of each imaging frame).

ii.
```python
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. Uses direct timestamps rather than frame-number interpolation.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The difference between each frame's timestamp and the trial start timestamp, multiplied by 86400 to convert from days to seconds. Values start near zero and increase.

ii.
```python
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. Straightforward time difference calculation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same selected frame timestamps used for neural data.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. Frame-level alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` (boolean per trial indicating reward delivery) and `WallName` (wall texture name per trial). The AI identifies the rewarded wall from trials where `isRew` is True, then marks ALL trials with that wall as reward-available.

ii.
```python
def rewarded_wall(beh, sid):
    walls = np.asarray(beh["WallName"])
    reward_walls = np.unique(walls[np.asarray(beh["isRew"], dtype=bool)])
    # ...
    return None if len(reward_walls) == 0 else str(reward_walls[0])

rw_wall = rewarded_wall(beh, sid)
reward_available = (
    np.zeros(len(walls), dtype=np.float32)
    if rw_wall is None
    else (walls == rw_wall).astype(np.float32)
)
```

iii. The AI argued that "reward availability" means whether reward is available in the corridor (not whether it was actually delivered), so all trials with the rewarded wall texture should be marked as 1. This gives 4,446 positive trials vs 4,336 actual deliveries.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The unique rewarded wall name is identified from any `isRew=True` trial. All trials sharing that wall name are marked 1; others are 0. Sessions with no delivered rewards are all 0. The value is broadcast across frames.

ii.
```python
reward_available = (
    np.zeros(len(walls), dtype=np.float32)
    if rw_wall is None
    else (walls == rw_wall).astype(np.float32)
)
# ...
np.full(T, reward_available[trial]),
```

iii. The AI distinguished between reward delivery (isRew) and reward availability (being in the rewarded corridor).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on each trial's corridor walls.

ii.
```python
STIMULUS_PREFIX_TO_CLASS = {"circle": 0, "leaf": 1, "rock": 2, "wood": 3}

def stimulus_class(wall_name: str) -> int:
    for prefix, value in STIMULUS_PREFIX_TO_CLASS.items():
        if str(wall_name).startswith(prefix):
            return value
```

iii. The AI maps wall names to four categories by prefix matching.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 unique wall names are mapped to 4 categories (circle=0, leaf=1, rock=2, wood=3) via prefix matching. The per-trial category is broadcast across all frames.

ii.
```python
stim = stimulus_class(str(walls[trial]))
# ...
np.full(T, stim, dtype=np.int16),
```

iii. The AI uses prefix matching rather than an explicit lookup dictionary, but achieves the same mapping.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame number of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
lick_fr = np.asarray(beh["LickFr"])
lick_tr = np.asarray(beh["LickTrind"])
finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
lick_fr_int = lick_fr[finite_lick].astype(np.int64)
lick_tr_int = lick_tr[finite_lick].astype(np.int64)
```

iii. The AI uses both lick frame and lick trial index to assign licks to specific trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI finds lick frames assigned to that trial (via `LickTrind`), then checks if any selected frames match using `np.isin`. A frame is 1 if it contains a lick, 0 otherwise.

ii.
```python
trial_lick_frames = lick_fr_int[lick_tr_int == trial]
lick = np.isin(frames, trial_lick_frames).astype(np.int16)
```

iii. The AI filters licks by both frame number and trial assignment.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick events are matched to the same selected frame indices used for neural data via `np.isin`.

ii.
```python
lick = np.isin(frames, trial_lick_frames).astype(np.int16)
```

iii. Frame-level alignment through shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame (0-40 in the textured corridor).

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
position = np.clip(position, 0, 3).astype(np.int16)
```

iii. Uses the native frame-aligned position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 to get meters, then floored to get integer bin indices. Values are clipped to 0-3.

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
position = np.clip(position, 0, 3).astype(np.int16)
```

iii. Straightforward binning into four 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-m bins: 0-1m (bin 0), 1-2m (bin 1), 2-3m (bin 2), 3-4m (bin 3). Derived by floor-dividing decimeter position by 10.

ii.
```python
OUTPUT_VALUES = [
    # ...
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    # ...
]
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
position = np.clip(position, 0, 3).astype(np.int16)
```

iii. Matches the instruction's request for 4 equal-length 1-m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read from `ft_Pos` at the same selected frame indices used for neural data.

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
```

iii. Frame-level alignment through shared indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"])[frames]
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. Uses the native frame-aligned running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile thresholds (25th, 50th, 75th percentiles) across ALL selected timepoints from ALL sessions in a prepass. Individual frame speeds are then binned using `np.digitize` with these global thresholds.

ii.
```python
def speed_quartile_prepass(records):
    # collects speed from all sessions...
    speeds = np.concatenate(speed_chunks).astype(np.float64, copy=False)
    thresholds = np.quantile(speeds, [0.25, 0.50, 0.75])
    return thresholds, summary

# Per trial:
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The AI argued that global quartile thresholds give consistent speed labels across sessions for the shared decoder.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins using global quartile thresholds. `np.digitize` maps speeds to bins 0-3. The thresholds are approximately [12.42, 25.35, 40.85] in native speed units.

ii.
```python
speed_edges = [12.422419514874129, 25.352614545028338, 40.854577405174]
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The AI noted that the global approach ensures each bin represents approximately 25% of all data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read from `ft_RunSpeed` at the same selected frame indices used for neural data.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"])[frames]
```

iii. Frame-level alignment through shared indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI validates that selected frames don't exceed the neural frame count, filters non-finite lick events, validates behavior shapes, checks for NaN/Inf in neural data, and raises errors for unexpected conditions. Behavior is truncated to neural frame length implicitly by only selecting frames within the neural data range.

ii.
```python
if max_selected >= nframes_neural:
    raise ValueError(...)
finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
if not np.all(np.isfinite(activity)):
    raise ValueError(...)
```

iii. The AI characterized this as "a very clean dataset" and focused on defensive validation rather than recovery.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files, which total ~405 GB. The full conversion took about 25 minutes.

ii.
```python
raw = np.load(spk_path, allow_pickle=True).item()
```

iii. The AI noted I/O cost dominates and cannot be further optimized.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in the main conversion could potentially be partially vectorized, but the variable trial lengths make this difficult. The behavior prepass iterates over all sessions twice (once for speed thresholds, once for conversion).

ii.
```python
for trial, frames in enumerate(frames_by_trial):
    # per-trial processing...
```

iii. The AI noted the I/O cost dominates, making loop optimization less impactful.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded twice: once in the speed quartile prepass and once during the main conversion. Frame selection logic (`selected_frames_by_trial`) is also called twice per session.

ii.
```python
def speed_quartile_prepass(records):
    # loads all behavior files...
    frames = selected_frames_by_trial(beh, sid)

def convert(args):
    # loads all behavior files again...
    frames_by_trial = selected_frames_by_trial(beh, sid)
```

iii. The AI designed this as a two-pass approach: first compute global speed thresholds, then do the full conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI collects extensive metadata, validation statistics, and performs comprehensive shape/value checking that goes beyond what is needed for the final pickle. The prepass also computes some summary statistics (reward counts, trial length stats) stored only in metadata.

ii.
```python
summary = {
    "n_trials": int(total_trials),
    "n_timepoints": int(len(speeds)),
    "trial_length_min": ...,
    # ...
}
```

iii. The AI designed for defensive correctness with extensive validation, which adds some overhead.
