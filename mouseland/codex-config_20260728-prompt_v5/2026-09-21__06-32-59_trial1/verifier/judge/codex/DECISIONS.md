# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as a subject/session index, scans nearly all `Beh_*.npy` behavior files, groups all behavior entries by a raw 5-part session key, and then processes one raw imaging recording at a time. For each processed session it loads spike planes from `spk/<raw_key>_neural_data.npy` and retinotopy labels from `retinotopy/<mouse>_<date>_trans.npz`.

ii. 
```python
def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

for beh_path in iter_behavior_files():
    beh_all = np.load(beh_path, allow_pickle=True).item()
    for full_key, beh in beh_all.items():
        raw_key = parse_raw_session_key(full_key)
        raw_to_views[raw_key].append(...)
```

```python
def load_spike_planes(raw_key: str) -> list[np.ndarray]:
    path = SPK_DIR / f"{raw_key}_neural_data.npy"
    obj = np.load(path, allow_pickle=True).item()
    return obj["spks"]

def load_retinotopy(raw_key: str) -> np.ndarray:
    mouse_date = "_".join(raw_key.split("_")[:4])
    path = RETINO_DIR / f"{mouse_date}_trans.npz"
    return np.load(path, allow_pickle=True)["iarea"]
```

iii. In `CONVERSION_NOTES.md` Step 4-5, the AI justified this as building the dataset around the 89 unique raw imaging recordings, because many behavior files are duplicated analysis views of the same recording.

## 1-b. How are the data split into subjects?

i. Subjects are defined by mouse name from `Imaging_Exp_info.npy`. Each raw session key is mapped to `rec["mname"]`, and final `subjects`/`subject_idx` are built from the sorted unique subject names.

ii. 
```python
for rec in records:
    raw_key = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
    raw_to_subject[raw_key] = rec["mname"]
```

```python
subjects = sorted({spec.subject for spec in selected_specs})
subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
"subject_idx": np.array([subject_to_idx[spec.subject] for spec in selected_specs], dtype=np.int8),
```

iii. The notes say the mouse identity is already explicit in the metadata, so no inferred grouping was needed.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique raw imaging recording identified by the first five underscore-delimited fields of a behavior key: mouse, year, month, day, and block. Multiple behavior “views” with the same raw key are merged into one session spec instead of becoming separate sessions.

ii. 
```python
def parse_raw_session_key(full_key: str) -> str:
    parts = full_key.split("_")
    return "_".join(parts[:5])
```

```python
for full_key, beh in beh_all.items():
    raw_key = parse_raw_session_key(full_key)
    raw_to_views[raw_key].append(...)
```

iii. In Step 4 the AI wrote that the paper/data support 89 raw recordings, while behavior files contain duplicate figure-specific views, so it merged duplicates by raw session key.

## 1-d. How are the data split into trials?

i. Trials are defined by `beh["ntrials"]`, but frames are assigned to each trial only when three conditions hold: the frame belongs to that trial (`ft_trInd == trial`), is inside the textured corridor (`ft_CorrSpc`), and is a running frame (`ft_move > 0`). This means each converted trial is a variable-length sequence of running corridor frames, not all corridor frames.

ii. 
```python
def get_trial_frame_indices(beh: dict[str, Any], nframes: int) -> list[np.ndarray]:
    ft_trind = np.asarray(beh["ft_trInd"][:nframes])
    ft_corr = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
    ft_move = np.asarray(beh["ft_move"][:nframes]) > 0

    for trial in range(ntrials):
        idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
        frame_sets.append(idx.astype(np.int64, copy=False))
```

iii. The AI justified this in Step 5 and Step 10 by saying the reference analyses use a valid-frame mask based on running within corridor and that `StartFr` is slightly offset from the first usable frame.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference 99th-percentile long-trial filter. Instead, a trial is kept if it has at least one selected running-corridor frame. Sessions with fewer than 2 surviving trials are rejected.

ii. 
```python
for trial, frame_idx in enumerate(trial_frame_sets):
    if frame_idx.size == 0:
        continue
    ...

if len(neural_trials) < 2:
    raise ValueError(f"Session {spec.raw_key} has fewer than 2 valid trials after processing")
```

iii. The notes argue that using only running corridor frames already removes long idle pauses, so the AI did not add a separate whole-trial outlier-length filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `spks` in each session’s spike file, with one array per imaging plane. Brain-region metadata are derived from `iarea` in the session’s retinotopy file.

ii. 
```python
def load_spike_planes(raw_key: str) -> list[np.ndarray]:
    obj = np.load(path, allow_pickle=True).item()
    return obj["spks"]

def load_retinotopy(raw_key: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)["iarea"]
```

iii. The notes explicitly say the public data already contain deconvolved activity and retinotopy labels, so no upstream neural derivation was needed.

## 2-b. How is the `neural` data processed?

i. For each kept trial, the AI slices the spike planes at the selected frame indices, concatenates planes along the neuron axis, and stores the result as `float16`. It does not pad or resample trials.

ii. 
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16, copy=False)
neural_trials.append(neural_trial)
```

iii. In Step 6 the AI said it avoided full-session concatenation to reduce memory, and stored `float16` because the decoder later converts data back to `float32` internally.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not drop any neurons. It maps retinotopy codes into five region labels (`V1`, `mHV`, `lHV`, `aHV`, `other`) and keeps all neurons, including those outside the four main visual-area groups.

ii. 
```python
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "other"]

def build_brain_region_idx(iarea: np.ndarray) -> np.ndarray:
    idx = np.full(iarea.shape, 4, dtype=np.int8)
    idx[iarea == 8] = 0
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    idx[np.isin(iarea, [5, 6])] = 2
    idx[np.isin(iarea, [3, 4])] = 3
    return idx
```

iii. Step 5 says the AI chose to “retain all neurons” because it did not find a public global neuron-QC rule beyond Suite2p preprocessing and treated region as metadata rather than a filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says alignment is to corridor entry, but operationally each trial starts at the first frame assigned to that trial that is both in corridor and moving. Internal non-running frames are removed, so the neural data are aligned to a running-only trial trajectory rather than a contiguous trial from `StartFr`.

ii. 
```python
idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
...
"temporal_alignment_event": "first imaging frame assigned to the current corridor trial (corridor entry)",
"frame_selection": "frames where ft_trInd == trial, ft_CorrSpc is True, and ft_move > 0",
```

iii. In Step 10 the AI justified this by noting `StartFr` was offset by 1-4 frames in spot checks, and that the reference imaging analyses used a running-frame mask.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal time bin size is one imaging frame, `1000 / 3.17` ms. No temporal rebinning is applied; the AI keeps frame-level samples after its frame-selection mask.

ii. 
```python
NOMINAL_FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / NOMINAL_FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
```

iii. The notes say the script keeps framewise samples because the decoder task needs time-varying licking and speed outputs, unlike the paper’s position-interpolated analyses.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from `SoundPos` and framewise corridor position `ft_Pos`, not from `SoundFr` and frame timestamps.

ii. 
```python
sound_pos = np.asarray(beh["SoundPos"][: int(beh["ntrials"])], dtype=np.float32)
ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
...
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. Step 7 and Step 10 say the AI switched from wall-clock timestamps to position-derived time so that excluded pause periods would not inflate cue-relative times.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the AI computes time to cue as `(SoundPos - current_position) / 6 dm/s`, using a fixed virtual corridor speed of 60 cm/s. The value is therefore positive before the cue and negative after it.

ii. 
```python
CORRIDOR_SPEED_DM_PER_S = 6.0
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. The AI’s justification was that frame timestamps become misleading after dropping stationary frames, while position divided by nominal VR speed gives “motion time” along the corridor.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same `frame_idx` positions used to slice the neural data for that trial, so it has the same per-trial length as the neural matrix.

ii. 
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. The notes repeatedly state that all converted streams should share the same selected frame set for each trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives training day from the raw session key: subject identity, calendar date, and block number. It finds each subject’s first recording date and measures later sessions relative to it.

ii. 
```python
date, block = parse_date_and_block(raw_key)
subject_dates[raw_to_subject[raw_key]].append((date, block))
subject_first_date = {
    subject: min(date for date, _ in date_blocks)
    for subject, date_blocks in subject_dates.items()
}
```

iii. Step 5 says the AI wanted a “mouse-relative continuous session day” rather than a simple session count.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes `training_day` as the number of elapsed calendar days since the subject’s first imaging date, plus a small `0.01 * (block - 1)` offset to distinguish same-day blocks. That scalar is then broadcast across all timepoints of the trial.

ii. 
```python
training_day = float((date - subject_first_date[subject]).days) + 0.01 * (block - 1)
...
day_trial = np.full(frame_idx.size, spec.training_day, dtype=np.float16)
```

iii. The notes justify this as preserving within-mouse chronology while distinguishing multiple blocks on the same date.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives this from framewise position `ft_Pos` after trial masking, not from `StartFr` or frame timestamps.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
...
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. In Step 10 the AI explicitly says it moved away from raw timestamps because removing pause frames made wall-clock “time since start” unrealistically large.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI converts retained frame positions to seconds by dividing corridor position in decimeters by the nominal VR speed of `6 dm/s`. Because it uses position directly, time starts at the first retained running frame’s position rather than an interpolated `StartFr` timestamp.

ii. 
```python
CORRIDOR_SPEED_DM_PER_S = 6.0
trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. The stated rationale was to represent motion time through the corridor, not wall-clock time that includes stationary periods omitted from the converted trial.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same `frame_idx` samples as the neural data, so its time axis is identical in length and ordering to each neural trial.

ii. 
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
input_trial = np.vstack([... , t_sec.astype(np.float16, copy=False), ...])
```

iii. The AI’s general alignment rule was to derive every time-varying input/output from the same per-trial selected frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is taken directly from the trial-level `isRew` array.

ii. 
```python
is_rew = np.asarray(beh["isRew"][: int(beh["ntrials"])], dtype=np.int8)
```

iii. The notes say the converter should use direct task variables from the behavior data instead of reconstructing reward status.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value is cast to float and broadcast to every timepoint in the kept trial.

ii. 
```python
reward_trial = np.full(frame_idx.size, float(is_rew[trial]), dtype=np.float16)
...
input_trial = np.vstack([... , reward_trial])
```

iii. The AI’s justification in Step 4 was that `isRew` already directly encodes whether reward is available in that corridor.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives stimulus labels from per-trial `WallName`, but it first builds a session-specific mapping from `UniqWalls` and `stim_id` gathered across all behavior views for the same raw session.

ii. 
```python
uniq_walls = list(map(str, np.asarray(view.beh["UniqWalls"])))
stim_ids = np.asarray(view.beh["stim_id"])
...
label_map[wall] = canonical
```

```python
wall_name = np.asarray(beh["WallName"][: int(beh["ntrials"])], dtype=object)
literal_wall = str(wall_name[trial])
canonical_wall = spec.wall_to_stimulus[literal_wall]
```

iii. The trajectory and Step 5 say the AI found that literal wall names can map differently across mice/sessions, so it preferred a session-specific merged mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each trial’s `WallName` through a session-specific canonical label map and encodes the result as one of 8 classes: `circle1`, `circle2`, `circle3`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`. The category is then broadcast across every timepoint in the trial.

ii. 
```python
OUTPUT_STIMULI = [
    "circle1", "circle2", "circle3", "leaf1",
    "leaf2", "leaf3", "leaf1_swap1", "leaf1_swap2",
]
...
stimulus_idx = OUTPUT_STIM_TO_IDX[canonical_wall]
...
np.full(frame_idx.size, stimulus_idx, dtype=np.uint8)
```

iii. In Step 5 the AI justified this as preserving session-specific stimulus identity and preserving `circle3` instead of collapsing everything into 4 coarse categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`: the AI uses lick frame numbers together with lick trial IDs to assign licks to converted trial frames.

ii. 
```python
lick_frames = np.asarray(beh["LickFr"], dtype=int)
lick_trials = np.asarray(beh["LickTrind"], dtype=int)
...
lick_trial_mask = lick_trials == trial
licking = binarize_licks(lick_frames, lick_trial_mask, frame_idx)
```

iii. The implied rationale is that `LickFr` alone gives imaging-frame timing, while `LickTrind` makes the per-trial restriction explicit.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI creates a zero vector over the kept frame indices, intersects that trial’s lick frames with `frame_idx`, and sets matching positions to `1`. Multiple licks in the same kept frame still produce a binary `1`.

ii. 
```python
out = np.zeros(frame_idx.size, dtype=np.uint8)
frames = np.asarray(lick_frames[trial_lick_mask], dtype=np.int64)
kept = np.intersect1d(frames, frame_idx, assume_unique=False)
offsets = np.searchsorted(frame_idx, kept)
out[offsets[valid]] = 1
```

iii. No separate prose justification was given beyond preserving frame alignment and using a binary time-varying lick signal.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by using the exact same per-trial `frame_idx` used to slice the neural matrices.

ii. 
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
licking = binarize_licks(lick_frames, lick_trial_mask, frame_idx)
```

iii. The notes treat shared per-trial frame indices as the common alignment mechanism for all time-varying streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position comes directly from framewise `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
...
pos_bins = position_to_bins(ft_pos[frame_idx]).astype(np.uint8, copy=False)
```

iii. Step 5 says the position output should come from the paper’s direct framewise position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI floors position in decimeters to 1 m bins by dividing by `10`, then clips the result to bins `0..3`.

ii. 
```python
def position_to_bins(ft_pos_dm: np.ndarray) -> np.ndarray:
    pos_dm = np.asarray(ft_pos_dm, dtype=np.float32)
    bins = np.floor(pos_dm / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. In Step 5 the AI justified this as implementing the requested four equal-length 1 m bins across the textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It uses four hard spatial thresholds corresponding to `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` meters, encoded as integer bins `0-3`.

ii. 
```python
return np.clip(np.floor(pos_dm / 10.0).astype(np.int16), 0, 3)
```

iii. The notes explicitly say these are “4 equal 1 m bins.”

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled at the same `frame_idx` values used for the neural trial.

ii. 
```python
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
pos_bins = position_to_bins(ft_pos[frame_idx]).astype(np.uint8, copy=False)
```

iii. The AI’s alignment rule was again shared frame indices across streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived directly from framewise `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float32)
...
speed_trial = ft_speed[frame_idx].astype(np.float32, copy=False)
```

iii. The notes describe speed as a direct framewise behavior variable that only needed discretization.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first stores continuous speed values for every kept frame of every trial, concatenates all of them across the whole converted dataset, computes global quartile edges with `np.quantile`, and then converts each trial’s speed values to 4 categories by thresholding against those edges.

ii. 
```python
all_speeds = np.concatenate(
    [speed for session in processed_sessions for speed in session["speed_trials"]],
    axis=0,
)
quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
```

```python
bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
session["output"][trial_idx][3] = bins
```

iii. Step 5 says the AI preferred global quartiles so the speed categories would be consistent across sessions for a cross-session decoder.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded by the three dataset-wide quartile boundaries returned by `np.quantile(all_speeds, [0.25, 0.5, 0.75])`, and `np.digitize` assigns bins `0-3`.

ii. 
```python
quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
...
bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
```

iii. The notes justify this as giving four bins that each represent 25% of the included data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sampled on the same kept frame indices as the neural data, then binned without changing that indexing.

ii. 
```python
speed_trial = ft_speed[frame_idx].astype(np.float32, copy=False)
...
session["output"][trial_idx][3] = bins
```

iii. The AI uses `frame_idx` as the common alignment reference before and after discretization.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI trims framewise behavior arrays to the neural frame count, verifies duplicate behavior views agree on core arrays, verifies spike-plane frame counts and retinotopy neuron counts, skips empty trials, and errors out on missing stimulus mappings or sessions with fewer than 2 valid trials.

ii. 
```python
if any(int(arr.shape[1]) != nframes for arr in planes):
    raise ValueError(...)
if total_neurons != int(iarea.shape[0]):
    raise ValueError(...)
```

```python
ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
...
if frame_idx.size == 0:
    continue
...
if missing:
    raise ValueError(f"Missing label mapping for session {raw_key}: {missing}")
```

iii. Compared with the reference, the AI added many explicit validation checks. The notes describe these as safeguards against inconsistent duplicate behavior views and mismatched session files.

## 12-a. What are the most time-consuming steps of the code?

i. The AI’s notes identify large-session spike I/O, per-trial neural slicing/concatenation, and full-dataset serialization as the main time and memory costs.

ii. 
```python
planes = load_spike_planes(spec.raw_key)
...
neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16, copy=False)
...
with out_path.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 and Step 9 say a first full run became too large and too slow to serialize, which drove later dtype and frame-mask changes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI still loops over all trials to find frame indices, loops over trials again to build each trial, and loops again to bin licks and apply speed bins. `get_trial_frame_indices` in particular rescans frame labels once per trial.

ii. 
```python
for trial in range(ntrials):
    idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
```

```python
for trial, frame_idx in enumerate(trial_frame_sets):
    ...
for session in processed_sessions:
    for trial_idx, speed_trial in enumerate(session["speed_trials"]):
        bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
```

iii. The notes mention some speedups, but the final code still leaves these per-trial Python loops in place.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats trial-frame extraction in both sample selection and session processing, loads retinotopy once in sample selection and again during conversion, and performs a second full pass over stored speed arrays to apply bins after trial construction.

ii. 
```python
def count_trials_with_corridor_licks(beh: dict[str, Any]) -> int:
    frame_sets = get_trial_frame_indices(beh, len(beh["ft"]))
```

```python
def get_session_neuron_count(raw_key: str) -> int:
    return int(load_retinotopy(raw_key).shape[0])
...
iarea = load_retinotopy(spec.raw_key)
```

```python
speed_trials.append(speed_trial)
...
for session in processed_sessions:
    for trial_idx, speed_trial in enumerate(session["speed_trials"]):
        ...
```

iii. The notes justify some of this as a tradeoff for lower peak memory and easier sample-session selection, but it is still repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script carries extra machinery for plotting/sample diagnostics, stores continuous `speed_trials` only to overwrite the final categorical speed output later, and computes/stores session-specific label maps and reward-mode metadata that are not used by downstream decoder training.

ii. 
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
...
def make_processing_plot(...):
    ...
```

```python
"speed_trials": speed_trials,
...
session["output"][trial_idx][3] = bins
```

```python
"session_label_maps": {
    spec.raw_key: dict(processed_sessions[i]["wall_to_stimulus"])
    for i, spec in enumerate(selected_specs)
},
"reward_modes_present": sorted({session["reward_mode"] for session in processed_sessions}),
```

iii. The notes frame these as useful for debugging and documentation, but they are not required by the decoder’s final input format.
