# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master experiment index from `beh/Imaging_Exp_info.npy`, builds one canonical session per unique raw recording id `<mname>_<datexp>_<blk>`, and collapses duplicate behavior aliases across experiment types. Behavior files are loaded from `Beh_<exp_type>.npy` through a small LRU cache, while spike data and retinotopy are loaded per session during conversion.

ii. 
```python
def load_experiment_index() -> dict[str, list[dict[str, Any]]]:
    return np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()

@lru_cache(maxsize=2)
def load_behavior_file(exp_type: str) -> dict[str, dict[str, Any]]:
    return np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```

```python
for exp_type, entries in exp_info.items():
    beh_file = load_behavior_file(exp_type)
    for entry in entries:
        raw_id = canonical_raw_id(entry)
        beh_key = behavior_key_for_entry(entry)
        rec = per_raw.setdefault(raw_id, {...})
```

```python
beh = get_behavior(session)
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset contains `142` experiment-index entries and `99` unique behavior keys but only `89` real neural recordings, matching the paper’s recording count. The notes also say the script should do a deterministic two-pass conversion and cache behavior loads to avoid unnecessary rereads.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined from `entry["mname"]` in the experiment index. The final `subjects` list is the sorted set of unique mouse names across kept sessions, and `subject_idx` maps each converted session to that list.

ii. 
```python
rec = per_raw.setdefault(
    raw_id,
    {
        "subject": entry["mname"],
        "dateexp": entry["datexp"],
        "block": entry["blk"],
        ...
    },
)
```

```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.subject])
```

iii. The notes treat `mname` as the authoritative subject id and repeatedly reference the expected paper-level total of `19` mice.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique raw recording id `<mouse>_<date>_<block>`. If that same raw recording appears under multiple experiment types or behavior aliases, the AI collapses those entries into one `SessionInfo` record while preserving aliases only as metadata.

ii. 
```python
def canonical_raw_id(entry: dict[str, Any]) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
```

```python
rec = per_raw.setdefault(
    raw_id,
    {
        "subject": entry["mname"],
        "dateexp": entry["datexp"],
        "block": entry["blk"],
        "behavior_exp_type": exp_type,
        "behavior_key": beh_key,
        "aliases": set(),
        "exp_types": set(),
    },
)
rec["aliases"].add(beh_key)
rec["exp_types"].add(exp_type)
```

iii. The notes explicitly justify this as a consistency fix: the paper says `89 recordings in 19 mice`, so the conversion should use `89` canonical raw sessions rather than duplicating behavior aliases.

## 1-d. How are the data split into trials?

i. Trials are split by grouping frame indices according to `ft_trInd`, but only after applying a corridor-running mask. That means a converted trial contains the subset of frames satisfying `ft_trInd == trial`, `ft_CorrSpc`, and `ft_move > 0`; non-running frames inside the raw trial are dropped instead of being kept as part of a contiguous corridor traversal.

ii. 
```python
def trial_frame_groups(beh: dict[str, Any]) -> list[np.ndarray]:
    ft_tr = np.asarray(beh["ft_trInd"])
    ft_corr = np.asarray(beh["ft_CorrSpc"], dtype=bool)
    ft_move = np.asarray(beh["ft_move"]) > 0
    ...
    keep = finite & ft_corr & ft_move
    frame_idx = np.flatnonzero(keep)
    trial_ids = tr_int[keep]
```

```python
changes = np.flatnonzero(np.diff(trial_ids)) + 1
split_frames = np.split(frame_idx, changes)
split_trials = np.split(trial_ids, changes)
for frames, tids in zip(split_frames, split_trials):
    groups[int(tids[0])] = frames.astype(np.int32, copy=False)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as preserving the paper’s running-only analysis rule while still aligning the dataset to corridor entry for decoder use.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s long-trial outlier filter. It drops only trials with zero retained corridor-running frames, and later drops sessions that end up with fewer than two converted trials.

ii. 
```python
for trial, frames in enumerate(groups):
    if frames.size == 0:
        continue
```

```python
if len(neural_trials) < 2:
    print(
        f"[convert] skipping {session.raw_id}: only {len(neural_trials)} converted trial(s)",
        flush=True,
    )
    continue
```

iii. The notes say the session/trial curation rule is to drop trials with zero retained frames and sessions with fewer than two trials. Later notes also show that very long sparse-movement trials were inspected and deliberately retained as “source-faithful” rather than filtered out.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the per-session `spks` arrays in `<session_id>_neural_data.npy`. Brain-region annotations come from `iarea` in the corresponding retinotopy file.

ii. 
```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)

iarea = np.asarray(retino["iarea"], dtype=np.float32)
brain_region_idx = map_iarea_to_region_idx(iarea)
```

iii. The notes repeatedly justify using released deconvolved `spks` directly because both the paper and reference code analyze those traces rather than recomputing `dF/F`.

## 2-b. How is the `neural` data processed?

i. For each trial, the AI slices the selected frame indices out of each raw spike block, concatenates the plane-wise slices into one neuron-by-time matrix, and casts the result to `float32`. It does not compute `dF/F`, deconvolve, pad, or temporally resample the traces.

ii. 
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
session_neural.append(neural_trial)
```

iii. The notes say this was chosen to keep the released deconvolved signal intact and to avoid the extra memory cost of concatenating a whole session before trial slicing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not drop neurons outside the four main visual areas. Instead, it maps `iarea` into six labels: `V1`, `mHV`, `lHV`, `aHV`, `unassigned_7`, and `outside_visual`, and keeps all rows unless an unexpected `iarea` code appears.

ii. 
```python
BRAIN_REGIONS = [
    "V1",
    "mHV",
    "lHV",
    "aHV",
    "unassigned_7",
    "outside_visual",
]
```

```python
out[iarea == 8] = 0
out[np.isin(iarea, [0, 1, 2, 9])] = 1
out[np.isin(iarea, [5, 6])] = 2
out[np.isin(iarea, [3, 4])] = 3
out[iarea == 7] = 4
out[iarea == -1] = 5
if np.any(out < 0):
    raise ValueError(...)
```

iii. The notes justify this as a “brain-region policy” meant to preserve all neural rows and paper-level neuron counts while still exposing the main visual-area grouping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata declares alignment to corridor entry / trial start, but the actual neural samples are the retained corridor-running imaging frames inside each trial. So the trial is indexed relative to trial start conceptually, but the first retained neural bin can occur after corridor entry if the animal was stationary at first.

ii. 
```python
keep = finite & ft_corr & ft_move
...
groups[int(tids[0])] = frames.astype(np.int32, copy=False)
```

```python
"temporal_alignment_event": "corridor entry / trial start",
"frame_selection": "native imaging frames within corridor while running (ft_CorrSpc & ft_move > 0)",
```

iii. The notes explicitly frame this as a compromise: use corridor entry as the common event required by the task, but preserve the running-only frame selection used in the paper’s analyses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one native imaging frame at `3.17 Hz`, so `1000 / 3.17` ms per bin. No temporal rebinning or resampling is applied.

ii. 
```python
FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

```python
"time_bin_size": float(TIME_BIN_MS),
"source_frame_rate_hz": float(FRAME_RATE_HZ),
```

iii. The notes justify this by pointing back to the reference code and paper, which operate on native imaging-frame samples.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundTime` and `ft`, both treated as raw timestamps.

ii. 
```python
ft = np.asarray(beh["ft"], dtype=np.float64)
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
```

```python
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. The notes describe this as one of the decoder-specific inputs built directly from native timing variables.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every retained frame of a trial, the AI subtracts the frame timestamp from the trial’s cue timestamp and converts the difference from MATLAB-day units to seconds. It does not interpolate cue frame numbers onto a frame-time axis; it uses `SoundTime` directly.

ii. 
```python
frame_times = ft[frames]
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. The notes defend the time inputs as “source-faithful timing,” using raw timestamps tied to corridor entry and cue time rather than redefining them after masking.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same `frame_times = ft[frames]` array that defines the neural bins for that converted trial, so it is exactly aligned to the retained neural columns.

ii. 
```python
frame_times = ft[frames]
...
input_trial = np.stack(
    [time_to_cue, day_of_training, time_since_start, reward_availability],
    axis=0,
)
```

iii. The notes treat all decoder inputs as being built on the same retained frame grid as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the calendar date string `dateexp` in the session metadata, after parsing it into a Python `date`.

ii. 
```python
@property
def date_obj(self):
    return datetime.strptime(self.dateexp, "%Y_%m_%d").date()
```

```python
return {
    session.raw_id: float((session.date_obj - subject_first_date[session.subject]).days)
    for session in sessions
}
```

iii. The notes justify this by saying `sess#` and `days` metadata are inconsistent across experiment branches, so calendar-day offset is a continuous value defined for every session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI finds the earliest recording date for each subject, computes elapsed calendar days from that date to each session date, casts the value to `float32`, and broadcasts it across every retained frame in each trial.

ii. 
```python
def compute_day_offsets(sessions: list[SessionInfo]) -> dict[str, float]:
    subject_first_date = {}
    for session in sessions:
        subject_first_date[session.subject] = min(
            subject_first_date.get(session.subject, session.date_obj),
            session.date_obj,
        )
```

```python
day_value = np.float32(day_offsets[session.raw_id])
day_of_training = np.full(frames.size, day_value, dtype=np.float32)
```

iii. The notes explicitly call this “subject-specific calendar-day offset from first recording” and justify it as more robust than using inconsistent session-count metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` and `ft`.

ii. 
```python
trial_start_time = np.asarray(beh["Trial_start_time"], dtype=np.float64)
ft = np.asarray(beh["ft"], dtype=np.float64)
```

```python
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. The notes describe trial start as corridor entry and use the raw timestamps directly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the AI subtracts the trial’s start timestamp from the frame timestamp and converts the result from days to seconds.

ii. 
```python
frame_times = ft[frames]
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. The notes describe this as a direct raw-timestamp difference, intentionally kept in wall-clock time even after the running-only mask.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same retained frame indices as the neural data, so each value is aligned one-to-one with a neural column in the converted trial.

ii. 
```python
frame_times = ft[frames]
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. The notes consistently describe all per-trial streams as being built on the retained imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, the per-trial rewarded-corridor flag.

ii. 
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. The notes treat this as the native trial-level reward context and do not describe any extra transformation.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No nontrivial processing is applied. The per-trial boolean is converted to `0.0/1.0` and broadcast across all retained frames of the trial.

ii. 
```python
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. The notes justify this as using the native rewarded-versus-unrewarded corridor identity directly.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, but the AI uses the raw wall labels directly rather than collapsing them to four coarse categories.

ii. 
```python
def gather_stimulus_vocabulary(sessions: list[SessionInfo]) -> list[str]:
    vocab = set()
    for session in sessions:
        beh = get_behavior(session)
        vocab.update(map(str, np.unique(np.asarray(beh["WallName"]))))
    return sorted(vocab)
```

```python
wall_name = np.asarray(beh["WallName"])
stimulus_idx = stim_to_idx[str(wall_name[trial])]
```

iii. The notes justify this as a “stimulus label policy” that preserves raw names, including `wood*` and swap variants, to avoid ambiguous relabeling.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI constructs a global vocabulary of all raw `WallName` strings across sessions, maps each raw label to an integer id, and repeats that integer across all retained frames of the trial.

ii. 
```python
stim_vocab = gather_stimulus_vocabulary(sessions)
stim_to_idx = {stim: idx for idx, stim in enumerate(stim_vocab)}
```

```python
stimulus_idx = stim_to_idx[str(wall_name[trial])]
stimulus_out = np.full(frames.size, stimulus_idx, dtype=np.int16)
```

iii. The notes say this preserves the raw dataset labels and keeps swap variants and `wood*` names explicit.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the imaging-frame indices of licks.

ii. 
```python
def build_lick_binary(beh: dict[str, Any], nframes: int) -> np.ndarray:
    lick_binary = np.zeros(nframes, dtype=np.int8)
    lick_frames = np.asarray(beh["LickFr"])
```

```python
lick_out = lick_binary[frames].astype(np.int16, copy=False)
```

iii. The notes describe the licking output as a direct frame-based variable, matching the source convention used elsewhere in the codebase.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI builds a binary session-length lick vector, filters out non-finite and out-of-range lick frames, converts the remaining frame numbers to integer indices, deduplicates them with `np.unique`, and marks those frames as `1`.

ii. 
```python
lick_binary = np.zeros(nframes, dtype=np.int8)
lick_frames = np.asarray(beh["LickFr"])
if lick_frames.size == 0:
    return lick_binary
finite = np.isfinite(lick_frames)
lick_idx = lick_frames[finite].astype(np.int64, copy=False)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
if lick_idx.size:
    lick_binary[np.unique(lick_idx)] = 1
```

iii. The notes justify this as using the frame-index convention already present in the data and reference analyses, while being slightly defensive about invalid or repeated entries.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is sliced with the same retained frame indices as the neural trial, so it is aligned one-to-one with the neural columns.

ii. 
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
lick_out = lick_binary[frames].astype(np.int16, copy=False)
```

iii. The notes consistently describe all time-varying outputs as being placed on the retained imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=np.float32)
pos_out = position_to_bin(ft_pos[frames])
```

iii. The notes justify this by treating the native position unit as `0.1 m`, consistent with the 4 m texture corridor stored as `0` to `40`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips the native position to the texture corridor range, divides by `10` native units, floors the result, and returns a 0-to-3 bin id.

ii. 
```python
def position_to_bin(pos: np.ndarray) -> np.ndarray:
    bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. The notes justify this as converting the 4 m corridor into the required four equal 1 m spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are fixed corridor edges at `0`, `10`, `20`, `30`, and `40` native units, corresponding to `0-1m`, `1-2m`, `2-3m`, and `3-4m`.

ii. 
```python
"output_values": [
    stim_vocab,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ["q1", "q2", "q3", "q4"],
],
```

```python
bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
return np.clip(bins, 0, 3)
```

iii. The notes describe these as the four 1 m bins required by the decoder task.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is evaluated on the same retained frame indices `frames` that define the neural trial.

ii. 
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
pos_out = position_to_bin(ft_pos[frames])
```

iii. The notes describe position as a per-frame behavioral stream aligned to the same selected imaging frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)
...
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
```

iii. The notes identify running speed as the raw frame-level speed variable used for the decoder output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first performs a dataset-wide behavior-only scan over retained corridor-running frames from all sessions, concatenates all finite running-speed samples, computes quartile edges with `np.quantile`, stabilizes ties by nudging non-increasing edges upward, and then applies those global thresholds to each frame during conversion.

ii. 
```python
def stabilized_quantile_edges(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(np.float32)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf, dtype=np.float32)
    return edges
```

```python
all_speeds = np.concatenate(speeds)
edges = stabilized_quantile_edges(all_speeds)
...
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
```

iii. The notes explicitly justify this as computing global quartiles over the canonical full dataset after the chosen frame mask, then applying them consistently to every session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are defined by three global quantile thresholds at the 25th, 50th, and 75th percentiles of retained running-speed samples, and each frame is assigned by `np.digitize`.

ii. 
```python
def speed_to_bin(speed: np.ndarray, speed_edges: np.ndarray) -> np.ndarray:
    return np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

```python
"speed_bin_edges": [float(x) for x in speed_edges.tolist()],
...
["q1", "q2", "q3", "q4"],
```

iii. The notes justify this by the decoder-task requirement that the four bins each correspond to 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained frame indices used to build the neural trial.

ii. 
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
```

iii. The notes treat running speed like the other frame-level streams: aligned on the retained imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some irregularities defensively: it masks out non-finite `ft_trInd` values, ignores non-finite and out-of-range lick frames, and raises an error on unexpected `iarea` codes. Trials with no retained frames are skipped, and sessions with fewer than two converted trials are skipped. It does not otherwise correct long outlier trials or trim behavior arrays explicitly to neural frame count.

ii. 
```python
finite = np.isfinite(ft_tr)
tr_int = np.full(ft_tr.shape, -1, dtype=np.int32)
tr_int[finite] = ft_tr[finite].astype(np.int32)
```

```python
finite = np.isfinite(lick_frames)
lick_idx = lick_frames[finite].astype(np.int64, copy=False)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
```

```python
if np.any(out < 0):
    raise ValueError(f"Found unexpected iarea labels: {sorted(np.unique(iarea[out < 0]).tolist())}")
```

iii. The notes describe the dataset as mostly clean and treat the remaining long-trial oddities as source-data edge cases that should be documented rather than filtered away.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive stages are the dataset-wide behavior-only speed scan and the per-session loading and slicing of the very large spike files. Optional processing plots also add overhead when enabled.

ii. 
```python
def gather_speed_edges(sessions: list[SessionInfo]) -> np.ndarray:
    ...
    for idx, session in enumerate(sessions, start=1):
        beh = get_behavior(session)
        groups = trial_frame_groups(beh)
        ...
    all_speeds = np.concatenate(speeds)
```

```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
...
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
```

iii. The notes explicitly discuss runtime estimates and state that the remaining cost is dominated by the full-data conversion path after the behavior-only first pass and trial-wise slicing optimizations.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the main frame-grouping step in `trial_frame_groups`, so it avoided the human solution’s per-trial full-frame scan. The remaining loops are mainly over sessions and over ragged per-trial frame groups in `gather_speed_edges` and `convert_session`; those could only be partially vectorized because the converted trials have variable lengths.

ii. 
```python
keep = finite & ft_corr & ft_move
frame_idx = np.flatnonzero(keep)
trial_ids = tr_int[keep]
changes = np.flatnonzero(np.diff(trial_ids)) + 1
split_frames = np.split(frame_idx, changes)
```

```python
for frames in groups:
    if frames.size == 0:
        continue
    vals = ft_run_speed[frames]
```

```python
for trial, frames in enumerate(groups):
    if frames.size == 0:
        continue
    ...
```

iii. The notes say the code was deliberately written to avoid unnecessary whole-session concatenation and to use a safe behavior-only first pass; they do not claim an obvious faster hot-path replacement beyond that.

## 12-c. What processing does the code repeat multiple times?

i. The code revisits behavior multiple times: once to gather the global stimulus vocabulary, again to gather global speed edges, again during actual conversion, and again for optional plotting. `trial_frame_groups(beh)` is recomputed in several of those passes.

ii. 
```python
stim_vocab = gather_stimulus_vocabulary(sessions)
speed_edges = gather_speed_edges(sessions)
...
neural_trials, input_trials, output_trials, region_idx, meta = convert_session(...)
...
plot_processing_summary(session, meta, beh, neural_trials, input_trials, output_trials, speed_edges)
```

```python
groups = trial_frame_groups(beh)
```

iii. The notes justify this as a deliberate two-pass design: one behavior-only pass for global setup quantities, then a conversion pass, with optional diagnostic plotting on top.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The optional plotting path is diagnostic-only and is not used by decoder training or verification. The code also computes and stores detailed per-session metadata fields such as aliases, experiment types, reward mode, and stimulus inventories that are helpful for review but are not consumed by downstream decoder analyses.

ii. 
```python
if plots_remaining > 0:
    beh = get_behavior(session)
    plot_processing_summary(session, meta, beh, neural_trials, input_trials, output_trials, speed_edges)
    plots_remaining -= 1
```

```python
session_meta = {
    "raw_id": session.raw_id,
    "subject": session.subject,
    "dateexp": session.dateexp,
    "block": session.block,
    "behavior_exp_type": session.behavior_exp_type,
    "behavior_key": session.behavior_key,
    "aliases": list(session.aliases),
    "exp_types": list(session.exp_types),
    ...
    "reward_mode": str(beh["Reward_Mode"]),
    "stimuli_present": sorted(map(str, np.unique(wall_name))),
}
```

iii. The notes explicitly position these pieces as validation, inspection, and documentation support rather than core decoder inputs.
