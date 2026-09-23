# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, deduplicates recordings by `mname/datexp/blk`, groups behavior requests by experiment type, loads each `Beh_<experiment_type>.npy` once into a compact per-session dict, and then loads spikes and retinotopy per session during conversion.

ii. ```python
exp_info = np.load(
    data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
).item()
sources = make_source_map(exp_info, data_root / "spk")
behaviors = load_behaviors(data_root, sources)
```
```python
behavior_file = np.load(path, allow_pickle=True).item()
for sid, key in requested:
    behaviors[sid] = compact_behavior(behavior_file[key])
```
```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
```

iii. In the trajectory, the agent inspected the dataset layout first and described the source as separate neural, behavior, and retinotopy streams. It then said the converter handled all 89 unique recordings and treated repeated behavior copies as duplicate listings of the same physical recording.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the `mname` field from the experiment table. The final `subjects` list is sorted unique mouse names, and each session gets a `subject_idx` entry via `subject_to_id`.

ii. ```python
subjects = sorted({sources[sid]["record"]["mname"] for sid in session_ids})
subject_to_id = {name: idx for idx, name in enumerate(subjects)}
```
```python
subject_idx.append(subject_to_id[str(record["mname"])])
```

iii. In the trajectory, the agent examined `Imaging_Exp_info.npy` and confirmed the subject identity lives in `mname`, with 89 sessions across 19 mice.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` triple. The AI builds a `session_id` from those fields and uses the first occurrence in the experiment table as a canonical copy when the same recording appears in multiple behavior files.

ii. ```python
def session_id(record: dict) -> str:
    return f"{record['mname']}_{record['datexp']}_{record['blk']}"
```
```python
for experiment_type, records in exp_info.items():
    for record in records:
        sid = session_id(record)
        sources.setdefault(
            sid,
            {
                "experiment_type": experiment_type,
                "behavior_key": behavior_key(record),
                "record": record,
            },
        )
```

iii. The agent inspected the experiment table and found 142 listings but only 89 unique recordings, then justified using the first occurrence as a deterministic canonical copy for duplicated sessions.

## 1-d. How are the data split into trials?

i. Trials are not taken as full corridor traversals. Instead, the AI first builds a valid-frame mask requiring a valid trial id, `ft_CorrSpc`, and `ft_move > 0`, then groups the surviving frames by `ft_trInd`. Each trial therefore consists only of the retained running frames from the textured corridor.

ii. ```python
def valid_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    trial = beh["ft_trInd"][:nframes]
    valid_trial = (
        np.isfinite(trial)
        & (trial >= 0)
        & (trial < beh["ntrials"])
    )
    return (
        valid_trial
        & beh["ft_CorrSpc"][:nframes].astype(bool)
        & (beh["ft_move"][:nframes] > 0)
    )
```
```python
frame_indices = np.flatnonzero(frame_mask)
trial_ids = beh["ft_trInd"][: len(frame_mask)][frame_mask].astype(np.int64)
kept_trials = np.unique(trial_ids)
for trial in kept_trials:
    columns = np.flatnonzero(trial_ids == trial)
    frames = frame_indices[columns]
```

iii. In the trajectory, the agent explicitly said it would follow the paper code’s “running-only” mask and “preserve those original imaging frames,” rather than reconstruct full trial spans.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s long-trial outlier filter. Trials are filtered only implicitly: a trial disappears if it has no frames surviving the running-texture mask, and a whole session is rejected if fewer than two usable trials remain.

ii. ```python
frame_indices = np.flatnonzero(frame_mask)
trial_ids = beh["ft_trInd"][: len(frame_mask)][frame_mask].astype(np.int64)
if len(frame_indices) == 0:
    raise RuntimeError(f"No valid running corridor frames in {sid}")
```
```python
kept_trials = np.unique(trial_ids)
...
if len(session_neural) < 2:
    raise RuntimeError(f"Fewer than two usable trials in {sid}")
```

iii. The agent stated in the trajectory that it intended to retain the “full trial set” of 38,110 trials while filtering only non-running frames and out-of-area neurons.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data comes from `spks` in each session’s spike file, with neuron inclusion and region labels determined by `iarea` from the matching retinotopy file.

ii. ```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
...
iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
```

iii. In the trajectory, the agent inspected both the spike-file structure and retinotopy files and said the converter would keep the paper’s four retinotopic visual-area groups.

## 2-b. How is the `neural` data processed?

i. The AI does not smooth, normalize, deconvolve, or spatially interpolate the traces. It filters neurons by area, filters frames by the running-texture mask, extracts the selected columns plane by plane, concatenates planes, and stores per-trial contiguous copies of those retained frames.

ii. ```python
selected_planes = []
offset = 0
for plane in raw_planes:
    local_keep = np.flatnonzero(keep_neuron[offset : offset + plane.shape[0]])
    selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
    offset += plane.shape[0]
selected_neural = np.concatenate(selected_planes, axis=0)
```
```python
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. The trajectory says the agent chose to “preserve those original imaging frames” and to avoid spatial interpolation because the requested decoder alignment was temporal rather than position-aligned.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered solely by retinotopic area labels. The AI keeps neurons whose `iarea` codes fall into four grouped visual regions and drops atlas-unassigned or other-area neurons.

ii. ```python
AREA_CODES = {
    "V1": (8,),
    "medial": (0, 1, 2, 9),
    "anterior": (3, 4),
    "lateral": (5, 6),
}
```
```python
region = np.full(len(iarea), -1, dtype=np.int8)
for idx, name in enumerate(names):
    region[np.isin(iarea, AREA_CODES[name])] = idx
keep = region >= 0
return keep, region[keep].astype(np.int64)
```

iii. In the trajectory, the agent said it was following `code/utils.py::neu_area_ID` and repeatedly reported retained-neuron counts consistent with those four visual-area groups.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats corridor entry as the nominal alignment event, but the actual per-trial neural arrays start at the first retained running frame inside the corridor and omit non-running frames within the trial. The output is therefore aligned by trial identity and original timestamps, not as a contiguous corridor-entry-to-exit time series.

ii. ```python
frame_mask = valid_frame_mask(beh, nframes)
frame_indices = np.flatnonzero(frame_mask)
trial_ids = beh["ft_trInd"][: len(frame_mask)][frame_mask].astype(np.int64)
```
```python
for trial in kept_trials:
    columns = np.flatnonzero(trial_ids == trial)
    frames = frame_indices[columns]
    times = beh["ft"][frames]
    trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. The agent explicitly justified this in the trajectory by saying it would keep the original running frames and let timestamps retain gaps, rather than interpolate or pad to a fixed corridor-entry-aligned window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal bin is one imaging frame at 3.17 Hz, i.e. about 315 ms. No temporal rebinning or resampling is applied.

ii. ```python
FRAME_RATE_HZ = 3.17
```
```python
"time_bin_size": 1000.0 / FRAME_RATE_HZ,
```

iii. In the trajectory, the agent described the data as keeping original imaging frames and not spatially or temporally interpolating them.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` for each trial and `ft` for the timestamps of the retained imaging frames.

ii. ```python
times = beh["ft"][frames]
time_to_sound = (
    (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The agent said in the trajectory that it would use exact recorded frame timestamps for cue timing rather than reconstructing them from frame indices.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI subtracts each retained frame’s timestamp from the trial’s `SoundTime` and converts the MATLAB-day units to seconds. The value is positive before the cue and negative after the cue.

ii. ```python
time_to_sound = (
    (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
).astype(np.float32)
trial_input[0] = time_to_sound
```

iii. The trajectory justification was that the converter should use exact recorded timestamps for cue timing and preserve the original temporal gaps rather than interpolating onto a reconstructed uniform trial axis.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained frame timestamps that index the per-trial neural columns, so it has the same length and sample positions as the neural data for that trial.

ii. ```python
frames = frame_indices[columns]
times = beh["ft"][frames]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
time_to_sound = (
    (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The agent’s trajectory repeatedly emphasized that all trial-wise signals were tied to the original retained imaging frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session’s `mname` and `datexp` fields in the experiment table.

ii. ```python
record = sources[sid]["record"]
date = datetime.strptime(record["datexp"], "%Y_%m_%d").date()
subject = str(record["mname"])
```

iii. In the trajectory, the agent inspected `Imaging_Exp_info.npy` and noted that many sessions lacked a populated `days` field, so it based the variable on the universally available recording dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes elapsed calendar days since each mouse’s earliest imaging session, not ordinal training-session number. It broadcasts that scalar across all time bins in the trial.

ii. ```python
def elapsed_days_by_session(session_ids: list[str], sources: dict) -> dict[str, float]:
    parsed = {}
    first_date = {}
    for sid in session_ids:
        record = sources[sid]["record"]
        date = datetime.strptime(record["datexp"], "%Y_%m_%d").date()
        subject = str(record["mname"])
        parsed[sid] = (subject, date)
        first_date[subject] = min(date, first_date.get(subject, date))
    return {
        sid: float((date - first_date[subject]).days)
        for sid, (subject, date) in parsed.items()
    }
```
```python
trial_input[1] = day_by_session[sid]
```

iii. The code comment and trajectory justification are the same: because the experiment table does not provide a complete training-day counter, the agent chose elapsed calendar day as “the only uniform, non-imputed continuous measure.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` for each trial and `ft` for the timestamps of retained imaging frames.

ii. ```python
times = beh["ft"][frames]
time_from_start = (
    (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. In the trajectory, the agent said it would use exact recorded frame timestamps for trial timing rather than reconstruct trial start from frame numbers.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI subtracts the trial’s `Trial_start_time` from each retained frame timestamp and converts the difference from days to seconds. The resulting value is stored as a time-varying input.

ii. ```python
time_from_start = (
    (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
).astype(np.float32)
trial_input[2] = time_from_start
```

iii. The agent justified this as part of using exact recorded timestamps for cue and trial timing while keeping original frame gaps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame timestamps as the neural data, so it is aligned sample-by-sample to the neural columns. Because the AI drops non-running frames, the first neural sample in a trial can occur after time zero.

ii. ```python
frames = frame_indices[columns]
times = beh["ft"][frames]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
time_from_start = (
    (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The trajectory explicitly says the agent wanted timestamps to retain gaps caused by pauses instead of filling them in.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial `isRew` field.

ii. ```python
trial_input[3] = float(beh["isRew"][trial])
```

iii. The trajectory does not show extra deliberation here; the agent treated reward availability as a direct per-trial field from behavior.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No additional processing is applied beyond casting the per-trial Boolean to float and broadcasting it across all time bins of that trial.

ii. ```python
trial_input[3] = float(beh["isRew"][trial])
```

iii. In the trajectory, the agent did not indicate any transformation beyond using the behavior field directly.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial `WallName` field.

ii. ```python
def visual_vocabulary(behaviors: dict) -> list[str]:
    return sorted({str(wall) for beh in behaviors.values() for wall in beh["WallName"]})
```
```python
trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. In the trajectory, the agent inspected swap-session behavior files and saw that `WallName` carried labels such as `leaf1_swap1` and `leaf1_swap2`, then chose to use the observed vocabulary directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse textures to four base categories. It builds a sorted vocabulary of all distinct `WallName` strings in the dataset and encodes each trial using that 15-class label space, then broadcasts the label across every retained time bin of the trial.

ii. ```python
stimulus_values = visual_vocabulary(behaviors)
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
```
```python
trial_output = np.empty((4, ntime), dtype=np.int64)
trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. The trajectory shows the agent exploring the raw `WallName` vocabulary in swap sessions and then later reporting 15 visual-stimulus classes in verification output, indicating that it intentionally kept the fine-grained labels.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the lick frame indices in the session.

ii. ```python
lick_frame = beh["LickFr"]
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
...
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. In the trajectory, the agent inspected the behavior schema and identified `LickFr` as the relevant lick-alignment variable.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI drops non-finite lick-frame entries, casts the remaining lick-frame indices to integers, and marks a retained frame as 1 if its frame index appears in `LickFr`, otherwise 0.

ii. ```python
lick_frame = beh["LickFr"]
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
...
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. The trajectory indicates the agent wanted all trial-wise variables to stay on the retained neural-frame grid, so licking was turned into a binary indicator on those frame indices.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by using the exact same retained frame indices as the neural trial array. Each lick output vector therefore has the same length as the per-trial neural matrix.

ii. ```python
frames = frame_indices[columns]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. The agent’s trajectory repeatedly described alignment in terms of preserving the original retained imaging frames across signals.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-wise position variable `ft_Pos`, together with the recorded corridor length `Texture_Length`.

ii. ```python
out["texture_length_dm"] = float(beh["Texture_Length"])
```
```python
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
```

iii. The trajectory shows the agent inspected position ranges inside the valid corridor frames and chose to keep the position signal on those same frames.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI divides position by one quarter of the recorded corridor length, floors to an integer bin index, and clips the result into four categories.

ii. ```python
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
position_bin = np.clip(position_bin, 0, 3)
```

iii. The code comment says `Texture_Length` is 40 decimeters in this dataset, and the agent chose to use that recorded setting so the four bins remained exactly 1 m if a future source file changed.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four equal-length bins by dividing the corridor length into quarters and clipping the resulting bin index to the range `0..3`.

ii. ```python
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
position_bin = np.clip(position_bin, 0, 3)
```

iii. The trajectory justification is the same as in the code comment: use the recorded corridor length so the categories stay tied to the actual 4 m corridor geometry.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained frame indices used for the trial’s neural data, so the position vector is sample-aligned to the neural columns.

ii. ```python
frames = frame_indices[columns]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
```

iii. The trajectory repeatedly emphasizes retaining the original selected imaging frames across all modalities.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-wise `ft_RunSpeed` values.

ii. ```python
speed_edges = global_speed_edges(behaviors)
...
speed_bin = np.digitize(
    beh["ft_RunSpeed"][frames], speed_edges, right=False
).astype(np.int64)
```

iii. The trajectory shows the agent inspected the running-speed distribution over the retained frames and then computed quartiles from that retained sample set.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI pools all retained running-corridor frames across the entire dataset, computes global value-based quartile thresholds with `np.quantile`, and then bins each trial’s retained frames using those shared edges.

ii. ```python
def global_speed_edges(behaviors: dict) -> np.ndarray:
    speeds = []
    for beh in behaviors.values():
        mask = valid_frame_mask(beh, len(beh["ft"]))
        speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
    all_speeds = np.concatenate(speeds)
    edges = np.quantile(all_speeds, (0.25, 0.50, 0.75))
```
```python
speed_bin = np.digitize(
    beh["ft_RunSpeed"][frames], speed_edges, right=False
).astype(np.int64)
```

iii. In the trajectory, the agent explicitly said it “computes running-speed quartiles globally over the retained samples.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the three global quartile edges from `np.quantile`, and `np.digitize` maps each retained frame to category 0, 1, 2, or 3.

ii. ```python
edges = np.quantile(all_speeds, (0.25, 0.50, 0.75))
```
```python
speed_bin = np.digitize(
    beh["ft_RunSpeed"][frames], speed_edges, right=False
).astype(np.int64)
```

iii. The trajectory justification was that the quartiles should be computed once from the retained sample population, and the verification output later showed exactly balanced global speed bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running-speed labels are taken from the same retained frame indices used for the neural trial array, so they are aligned sample-by-sample with the neural data.

ii. ```python
frames = frame_indices[columns]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
speed_bin = np.digitize(
    beh["ft_RunSpeed"][frames], speed_edges, right=False
).astype(np.int64)
```

iii. The trajectory consistently describes all trial-wise signals as staying on the same retained imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior-derived streams to a common valid length, ignores non-finite trial indices and lick frames, raises errors on spike/retinotopy mismatches or missing behavior keys, and rejects sessions with no valid frames or fewer than two usable trials.

ii. ```python
nframes = min(
    nframes,
    len(beh["ft"]),
    len(beh["ft_trInd"]),
    len(beh["ft_CorrSpc"]),
    len(beh["ft_move"]),
)
```
```python
valid_trial = (
    np.isfinite(trial)
    & (trial >= 0)
    & (trial < beh["ntrials"])
)
```
```python
lick_frame = beh["LickFr"]
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
```

iii. In the trajectory, the agent spent time checking data consistency up front, then described the conversion as passing shape and retinotopy consistency checks session by session.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading very large spike files and selecting retained neurons and frames plane by plane for every session. A secondary cost is the full-dataset scan used to compute global running-speed quartiles.

ii. ```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
...
for plane in raw_planes:
    local_keep = np.flatnonzero(keep_neuron[offset : offset + plane.shape[0]])
    selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
```
```python
for beh in behaviors.values():
    mask = valid_frame_mask(beh, len(beh["ft"]))
    speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
```

iii. The trajectory repeatedly comments on the dataset’s size and on spike-file I/O dominating runtime and output size.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code repeatedly groups retained frames back into trials with `np.flatnonzero(trial_ids == trial)` inside a loop over trials, and it recomputes per-trial lick membership with `np.isin`. Those per-trial scans could be replaced with one-pass grouping or precomputed trial slices.

ii. ```python
kept_trials = np.unique(trial_ids)
for trial in kept_trials:
    columns = np.flatnonzero(trial_ids == trial)
    frames = frame_indices[columns]
```
```python
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. The trajectory does not show the agent discussing this as a problem; the focus there was on correctness and memory safety rather than squeezing out more vectorization.

## 12-c. What processing does the code repeat multiple times?

i. The AI scans behavior twice with nearly the same validity logic: once globally in `global_speed_edges()` and again per session in `convert()`. It also re-applies the same valid-frame mask logic to every session after already having used it to build the global speed thresholds.

ii. ```python
def global_speed_edges(behaviors: dict) -> np.ndarray:
    for beh in behaviors.values():
        mask = valid_frame_mask(beh, len(beh["ft"]))
```
```python
for session_number, sid in enumerate(session_ids, start=1):
    ...
    frame_mask = valid_frame_mask(beh, nframes)
    frame_indices = np.flatnonzero(frame_mask)
```

iii. In the trajectory, the agent explicitly described computing running-speed quartiles globally over retained samples before starting the full session-by-session conversion, so this repeated pass was deliberate.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra bookkeeping and validation that the decoder does not use: it stores rich `session_info` metadata, copies behavior into compact side dicts, checks behavior/spike-session set equality, stores reward-mode and source-experiment annotations, and saves global speed edges in metadata. These are not used by the decoder itself.

ii. ```python
out["texture_length_dm"] = float(beh["Texture_Length"])
out["reward_mode"] = str(beh["Reward_Mode"])
```
```python
if set(sources) != spk_sessions:
    ...
    raise RuntimeError(...)
```
```python
"running_speed_quartile_edges": speed_edges.tolist(),
"session_info": session_info,
```

iii. The trajectory shows the agent intentionally adding consistency checks and descriptive metadata so the conversion choices were explicit, even though those fields are not needed for downstream decoder training.
