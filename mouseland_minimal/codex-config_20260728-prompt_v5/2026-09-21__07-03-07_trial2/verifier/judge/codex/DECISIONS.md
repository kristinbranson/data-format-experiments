# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use `Imaging_Exp_info.npy` as the master index. Instead, it scanned every `Beh_*.npy` file under `beh/`, merged records that collapse to the same base session key, then iterated the merged session keys. For each merged session it loaded one spike file from `spk/` and one retinotopy file from `retinotopy/`.

ii. 
```python
def merge_behavior_records() -> dict[str, dict]:
    merged = {}
    beh_files = sorted(
        fname for fname in os.listdir(BEH_ROOT) if fname.startswith("Beh_") and fname.endswith(".npy")
    )
    for fname in beh_files:
        beh = np.load(os.path.join(BEH_ROOT, fname), allow_pickle=True).item()
        for key, record in beh.items():
            base_key = parse_base_session_key(key)
            if base_key not in merged:
                merged[base_key] = {k: v for k, v in record.items()}
```
```python
beh = behaviors[base_key]
spk_path = os.path.join(SPK_ROOT, f"{base_key}_neural_data.npy")
retino_path = os.path.join(RETINO_ROOT, f"{subject}_{base_key.split('_', 1)[1][:10]}_trans.npz")

spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```

iii. In the trajectory, the AI said the dataset was split across behavior metadata, deconvolved neural recordings, and retinotopy files, and later said it would “collapse duplicate behavior annotations onto the 89 unique recordings” before iterating sessions. It justified this as a way to make one tractable export from the raw files.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the first underscore-delimited token of each merged session key, then sorted to build `subjects` and `subject_idx`.

ii. 
```python
def parse_subject_and_date(base_key: str) -> tuple[str, datetime]:
    parts = base_key.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    return subject, date
```
```python
subjects = sorted(per_subject_dates)
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The trajectory does not give a separate argument for subject parsing; it treats the session key format as authoritative and uses it to recover mouse identity.

## 1-c. How are the data split into sessions?

i. Sessions are defined by “base session keys.” If a behavior key ends in a `swap...` suffix, that suffix is removed and the recordings are merged into one session. The script then processes the sorted merged base keys as the session list.

ii. 
```python
def parse_base_session_key(key: str) -> str:
    parts = key.split("_")
    if parts[-1].startswith("swap"):
        return "_".join(parts[:-1])
    return key
```
```python
for key, record in beh.items():
    base_key = parse_base_session_key(key)
    if base_key not in merged:
        merged[base_key] = {k: v for k, v in record.items()}
        continue
```

iii. In the trajectory, the AI said “Some recordings are reused across multiple figure-analysis files” and later concluded that duplicated entries were “the same recordings with alternate stimulus annotations,” so it would collapse them “to one recording apiece.”

## 1-d. How are the data split into trials?

i. Trials are defined by the framewise trial index `ft_trInd`, but only frames that are both inside the corridor (`ft_CorrSpc`) and moving (`ft_move > 0`) are kept. A trial is therefore represented by its moving corridor frames only, not by all frames from corridor entry through the corridor traversal.

ii. 
```python
trial_idx = np.asarray(beh["ft_trInd"][:nfr])
move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```
```python
for tr in range(int(beh["ntrials"])):
    mask = (trial_idx == tr) & move_corr_mask
    if not np.any(mask):
        continue
```

iii. The trajectory repeatedly says the AI wanted to keep “running periods the paper analyzed,” and its final summary describes the export as using “running frames within the 4 m corridor.”

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has no retained frames after the moving-corridor mask. There is no trial-length outlier filter. At the session level, the script raises an error if fewer than two trials survive.

ii. 
```python
for tr in range(int(beh["ntrials"])):
    mask = (trial_idx == tr) & move_corr_mask
    if not np.any(mask):
        continue
```
```python
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {base_key} has fewer than two valid trials after filtering.")
```

iii. The trajectory justifies this through the running-only decision: trials with no retained running corridor frames are treated as unusable, and the session must still satisfy the decoder’s minimum-trial requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the `spks` arrays in the session’s spike file. The neuron region labels come from `iarea` in the matching retinotopy file.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```

iii. In the trajectory, the AI explicitly said it was using deconvolved neural recordings plus retinotopy files and keeping “deconvolved traces.”

## 2-b. How is the `neural` data processed?

i. The AI concatenates a selected subset of neurons across planes, keeps only moving corridor frames within each trial, and stores each trial as `float16`. It does not pad trials.

ii. 
```python
selected_spk = np.concatenate(selected_chunks, axis=0)
```
```python
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The trajectory says the converter would “keep the paper’s deconvolved traces and movement restriction,” and later summarizes the output as “deconvolved neural activity on running frames within the 4 m corridor.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons assigned to the four broad retinotopic regions (`V1`, `mHV`, `lHV`, `aHV`). Within each region and plane it further keeps only “responsive” neurons whose mean activity in moving corridor frames exceeds mean activity in moving gray-space frames. It then ranks those neurons by corridor-frame variance and caps retention at 128 neurons per region per session.

ii. 
```python
corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
gray_mask = np.asarray(beh["ft_GraySpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```
```python
corr_mean = plane[:, corr_mask].mean(axis=1)
gray_mean = plane[:, gray_mask].mean(axis=1)
corr_var = plane[:, corr_mask].var(axis=1)
responsive = corr_mean > gray_mean
```
```python
order = np.lexsort((row_arr, plane_arr, -score_arr))
top = order[:MAX_NEURONS_PER_AREA]
```

iii. The trajectory says the AI wanted a “paper-motivated subset of retinotopically assigned corridor-responsive neurons” and repeatedly justified the 128-per-area cap as necessary to keep the dataset “trainable” and “tractable.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata says the alignment event is corridor entry, but the actual per-trial neural arrays are aligned to the subset of frames in that trial where the mouse is both in the corridor and moving. The first retained sample is therefore the first retained running corridor frame, not necessarily the first frame after trial start.

ii. 
```python
"temporal_alignment_event": "corridor entry (trial start)",
"off_start": 0.0,
"off_end": None,
```
```python
mask = (trial_idx == tr) & move_corr_mask
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. In the trajectory, the AI framed this as keeping “running-only analyses” while still calling the trials “corridor-aligned trial segments.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame interval estimated from the median `ft` difference and reports that as `time_bin_size` in milliseconds. It does not rebin or resample the data.

ii. 
```python
def compute_time_bin_size_ms(behaviors: dict[str, dict]) -> float:
    dts = []
    for beh in behaviors.values():
        ft = np.asarray(beh["ft"])
        if ft.size > 1:
            dts.append(np.median(np.diff(ft)) * MS_PER_DAY)
    return float(np.median(dts))
```
```python
"time_bin_size": time_bin_size,
```

iii. The trajectory does not justify this separately, but it consistently treats the imaging frames as the native aligned bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and from the imaging-frame duration inferred from `ft`.

ii. 
```python
dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
```
```python
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. The trajectory says the AI switched timing inputs to frame-index timing using `StartFr` and `SoundFr` because that was “the correct alignment space for the neural data.”

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in the trial, the AI subtracts the retained frame index from the trial’s `SoundFr` and multiplies by `dt_seconds`, producing a continuous time-to-cue signal in seconds. Positive values are before the cue and negative values after the cue.

ii. 
```python
selected_frames = frame_numbers[mask]
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. The trajectory explicitly says the original attempt mixed absolute behavior timestamps with neural alignment, so the AI changed to frame-index timing for this variable.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same retained moving-corridor frames that are used for that trial’s neural matrix.

ii. 
```python
mask = (trial_idx == tr) & move_corr_mask
selected_frames = frame_numbers[mask]
...
neural_trials.append(selected_spk[:, mask].astype(np.float16))
input_trials.append(
    np.vstack([time_to_sound, day_arr, time_since_start, reward_available]).astype(np.float32)
)
```

iii. The trajectory justification is the same as above: it explicitly switched to frame-index timing to stay in the same alignment space as the neural frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the subject name and recording date encoded in each merged session key.

ii. 
```python
def parse_subject_and_date(base_key: str) -> tuple[str, datetime]:
    parts = base_key.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    return subject, date
```

iii. The trajectory says the source metadata did not provide “a consistent explicit training-day count,” so the AI based this variable on session dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the AI finds the first recording date and then assigns each session the number of calendar days since that first date. That scalar is broadcast across all retained time bins in the trial.

ii. 
```python
first_dates = {subject: min(dates) for subject, dates in per_subject_dates.items()}
...
day_since_first[key] = float((date - first_dates[subject]).days)
```
```python
day_arr = np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. The trajectory and metadata both justify this as “calendar days since the first imaging session for each subject” because the AI believed no consistent explicit training-day count was available.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the imaging-frame duration inferred from `ft`.

ii. 
```python
dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. The trajectory says the AI changed timing variables to the `StartFr`/`SoundFr` frame-index space to match neural alignment.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame in the trial, the AI subtracts the trial’s `StartFr` from that frame index and multiplies by `dt_seconds`. The result is a continuous time-since-start signal in seconds.

ii. 
```python
selected_frames = frame_numbers[mask]
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. The trajectory says this was a correction to avoid “mixing absolute behavior timestamps with imaging-frame alignment.”

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained moving-corridor frames as the neural trial matrix.

ii. 
```python
mask = (trial_idx == tr) & move_corr_mask
selected_frames = frame_numbers[mask]
...
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The trajectory justification is again that `StartFr`-based frame indexing was the alignment space the AI wanted for the neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew` for each trial.

ii. 
```python
reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. The trajectory does not discuss this variable separately; it is taken directly from the behavior record.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No substantive processing is applied beyond converting the trial-level value to `0.0` or `1.0` and broadcasting it across the retained bins of that trial.

ii. 
```python
reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. The trajectory does not give a separate justification for this; it follows directly from the per-trial reward flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` for each trial.

ii. 
```python
wall_name = str(beh["WallName"][tr])
stim_category = category_to_idx[wall_to_category(wall_name)]
```

iii. The trajectory says the converter would “map wall names to stimulus families (`circle`, `leaf`, `rock`, `wood`).”

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI parses the alphabetic prefix of the wall name, lowercases it, checks that it is one of the four expected families, maps that family to an integer category, and broadcasts the result across all retained bins of the trial.

ii. 
```python
def wall_to_category(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(name))
    ...
    category = match.group(1).lower()
    if category not in {"circle", "leaf", "rock", "wood"}:
        raise ValueError(...)
```
```python
visual_category = np.full(mask.sum(), stim_category, dtype=np.int16)
```

iii. The trajectory explicitly describes this as mapping wall names to the four stimulus families for the decoder output.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. 
```python
lick_fr = np.asarray(beh["LickFr"], dtype=int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
```

iii. The trajectory does not justify this separately; it treats licking as a framewise behavioral stream already available in the behavior files.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI casts lick frame indices to integers, removes indices outside the imaged frame range, sets those frames to 1 in a binary framewise vector, then samples that vector on the retained trial frames.

ii. 
```python
lick_binary = np.zeros(nfr, dtype=np.int8)
lick_fr = np.asarray(beh["LickFr"], dtype=int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
if lick_fr.size:
    lick_binary[np.unique(lick_fr)] = 1
```
```python
lick = lick_binary[mask].astype(np.int16)
```

iii. The trajectory does not discuss this variable separately; it is a direct framewise binarization.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by applying the same retained moving-corridor frame mask used for the neural data.

ii. 
```python
mask = (trial_idx == tr) & move_corr_mask
...
lick = lick_binary[mask].astype(np.int16)
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The trajectory’s general justification is the running-only alignment rule: all time-varying variables are sampled on the same retained frame subset as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. The trajectory does not discuss this variable separately; it is treated as one of the framewise behavior streams.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips position to the corridor range, divides by 10 decimeters, floors the result, and casts it to integer category codes.

ii. 
```python
pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. The trajectory does not justify this separately; it follows the task’s request for four equal-length 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is thresholded into four bins: `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4)` meters, implemented as decimeter bins `0-9`, `10-19`, `20-29`, and `30-39`, then clipped to `0..3`.

ii. 
```python
pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. There is no separate trajectory discussion; this thresholding is implied by the decoder specification and the final code.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained moving-corridor frame mask used for the neural data.

ii. 
```python
mask = (trial_idx == tr) & move_corr_mask
...
pos_bin = ... ft_pos[mask] ...
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The trajectory’s general justification is that all streams should follow the same running-only frame subset.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. The trajectory does not discuss the raw source separately; it treats running speed as one of the framewise behavior streams.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes three global quartile edges from all sessions using frames where `ft_CorrSpc` is true and `ft_move > 0`. For each retained frame in a trial it then bins `ft_RunSpeed` by those global edges with `np.digitize`.

ii. 
```python
def compute_speed_edges(behaviors: dict[str, dict]) -> np.ndarray:
    speeds = []
    for beh in behaviors.values():
        mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)
        if np.any(mask):
            speeds.append(np.asarray(beh["ft_RunSpeed"])[mask])
    all_speeds = np.concatenate(speeds)
    return np.quantile(all_speeds, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The trajectory does not give a detailed separate justification beyond the broader running-only export plan and the desire for a tractable decoder dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded with three global quantile cutpoints computed from all retained moving-corridor frames across the dataset, producing four bins via `np.digitize`.

ii. 
```python
speed_edges = compute_speed_edges(behaviors)
...
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The trajectory does not justify the thresholding rule separately.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained moving-corridor frame mask used for the neural data.

ii. 
```python
mask = (trial_idx == tr) & move_corr_mask
...
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The trajectory’s general justification is that all time-varying variables should be aligned to the same running-only frame subset as neural activity.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates framewise behavior arrays to the number of imaged frames, drops lick indices outside that range, checks for spike/retinotopy neuron-count mismatches, skips trials with no retained moving-corridor frames, and errors on sessions with fewer than two valid trials.

ii. 
```python
trial_idx = np.asarray(beh["ft_trInd"][:nfr])
move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```
```python
lick_fr = np.asarray(beh["LickFr"], dtype=int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
```
```python
if nneu_total != iarea.shape[0]:
    raise RuntimeError(...)
```

iii. In the trajectory, the AI mostly discussed tractability and trainability rather than data-cleaning edge cases, but it did explicitly justify the timing fix as a correction for an alignment mistake found during verification.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each session’s large spike file, computing neuron statistics for corridor-versus-gray filtering, and processing all sessions twice at the dataset level for global metadata such as speed edges and time-bin size.

ii. 
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
selected_spk, region_idx = select_neurons_for_session(spk_planes, iarea, beh)
```
```python
speed_edges = compute_speed_edges(behaviors)
time_bin_size = compute_time_bin_size_ms(behaviors)
```

iii. The trajectory explicitly called the full export “the expensive step” because it had to stream all 89 recordings, and separately emphasized raw dataset scale and the need to keep the result trainable.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trial_data`, the per-plane/per-region loop nest in `select_neurons_for_session`, and the full-session loop used to accumulate speed edges are the main candidates.

ii. 
```python
for tr in range(int(beh["ntrials"])):
    mask = (trial_idx == tr) & move_corr_mask
```
```python
for plane_idx, plane in enumerate(spk_planes):
    ...
    for region in BRAIN_REGIONS:
```
```python
for beh in behaviors.values():
    mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)
```

iii. The trajectory does not discuss vectorization explicitly, but it does note that export cost is dominated by iterating large recordings.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly computes moving-corridor masks in separate functions, repeatedly parses wall names into categories, and performs a separate full-dataset pass to compute speed edges after it has already loaded and merged all behavior records.

ii. 
```python
mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)
```
```python
corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
gray_mask = np.asarray(beh["ft_GraySpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```
```python
all_categories = sorted(
    {
        wall_to_category(wall_name)
        for beh in behaviors.values()
        for wall_name in np.asarray(beh["WallName"]).tolist()
    }
)
```

iii. The trajectory does not call out this repetition explicitly; its focus was instead on keeping the export tractable.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the `stim_id` merge logic, because the merged `stim_id` values are never used downstream. The script also computes descriptive speed-bin labels and metadata fields that are not used by the decoder itself.

ii. 
```python
stim_old = np.asarray(merged[base_key]["stim_id"], dtype=float)
stim_new = np.asarray(record["stim_id"], dtype=float)
if stim_old.shape == stim_new.shape:
    fill = np.isnan(stim_old) & ~np.isnan(stim_new)
    stim_old[fill] = stim_new[fill]
    merged[base_key]["stim_id"] = stim_old
```
```python
"output_values": [
    all_categories,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    describe_speed_bins(speed_edges),
],
```

iii. The trajectory justified the merge as preserving the “richest mapping,” but the final converter never uses `stim_id`, so that specific merge work is not consumed by the downstream analysis.
