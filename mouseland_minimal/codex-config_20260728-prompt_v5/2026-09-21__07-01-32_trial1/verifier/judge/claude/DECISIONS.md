# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all spike files in `/app/data/spk/` (89 `*_neural_data.npy` files) as the canonical session list, rather than using `Imaging_Exp_info.npy` as the master index. For each spike file, it extracts the raw session key from the filename, then selects the best matching behavior file from the `beh/` directory using a scoring heuristic. Retinotopy is loaded per session from `retinotopy/`. Behavior files are loaded via `np.load(..., allow_pickle=True).item()`.

ii.
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
for spk_file in spk_files:
    raw_key = spk_file.name.replace("_neural_data.npy", "")
    ...
    beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
    beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
```

iii. The agent discovered that `Imaging_Exp_info.npy` has 142 entries but only 89 unique recordings, and chose to use spike files as the canonical list since each spike file corresponds to one unique recording. A behavior selection heuristic scores candidates by number of unique wall names, finite stim_ids, and absence of placeholder entries.

## 1-b. How are the data split into subjects?

i. The subject (mouse name) is extracted from the first part of the session key (e.g., `TX60` from `TX60_2022_05_30_1`). Unique subjects are sorted to form the subjects list, and `subject_idx` maps each session to its index.

ii.
```python
subject, _, block = parse_session_key(raw_key)
# ...
subjects = sorted({plan.subject for plan in session_plans})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The subject name is directly encoded in the session key/spike filename. No derivation needed.

## 1-c. How are the data split into sessions?

i. A session corresponds to one spike file. The agent uses all 89 spike files in the `spk/` directory. For each, the best behavior file/key is selected via a scoring heuristic that handles cases where a session appears in multiple behavior files.

ii.
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
for spk_file in spk_files:
    raw_key = spk_file.name.replace("_neural_data.npy", "")
    beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
```

iii. The agent found 34 sessions appearing in multiple behavior files and implemented a scoring function to select the best candidate per session, preferring files with more unique wall names and fewer placeholder entries.

## 1-d. How are the data split into trials?

i. Trials are identified by iterating from 0 to `ntrials - 1` using the behavior data. For each trial, frames are selected where `ft_trInd == trial` and `ft_CorrSpc == True` (inside the corridor texture). Trials are variable-length.

ii.
```python
ntrials = int(beh["ntrials"])
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```

iii. Same approach as the reference code -- uses the data's own trial labeling and corridor space flags.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) trial duration must be <= 40 seconds (`TRIAL_DURATION_MAX_S = 40.0`), computed as `(Gray_space_time - Trial_start_time) * 86400`; (2) maximum frame gap within a trial must be <= 1 second (`MAX_FRAME_GAP_S = 1.0`); (3) minimum 5 corridor frames per trial (`MIN_FRAMES_PER_TRIAL = 5`). Sessions with fewer than 2 valid trials are rejected.

ii.
```python
TRIAL_DURATION_MAX_S = 40.0
MAX_FRAME_GAP_S = 1.0
MIN_FRAMES_PER_TRIAL = 5
# ...
duration_s = float((gray_time[tr] - trial_start[tr]) * 24.0 * 3600.0)
max_gap_s = float(np.max(np.diff(ft[idx]) * 24.0 * 3600.0))
if duration_s > TRIAL_DURATION_MAX_S or max_gap_s > MAX_FRAME_GAP_S:
    continue
```

iii. The agent empirically explored trial durations and found pathological trials lasting 400-1770 seconds. At threshold 40s, 97.9% of trials are kept and 0 sessions are lost. The frame gap filter (99.94% pass) removes trials with anomalous temporal discontinuities. The min-frames filter is a safety floor (actual minimum was 11 frames).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike files (`spk/<session_id>_neural_data.npy`), which contains a list of neurons-by-frames arrays per imaging plane, concatenated across planes. Retinotopy (`iarea`) is loaded from `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
spk_chunks = spk_obj["spks"]
# concatenated across planes via memmap writing
```

iii. The agent traced the authors' loading code in `utils.py` which concatenates `spks` chunks.

## 2-b. How is the `neural` data processed?

i. The traces are not further processed (no dF/F or deconvolution). Frames from valid trials' corridor segments are extracted and stored as float16 in memory-mapped files. A custom `MappedTrialArray` class is used to enable efficient pickling by storing only the memmap path and row range.

ii.
```python
base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)
# ...
block = chunk[:, keep_idx[c0:c1]].T
base[c0:c1, col_start:col_start + n_chunk] = block
```

iii. The agent noted the data already holds deconvolved traces and chose memmap-backed storage to handle the large dataset size efficiently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **All neurons are kept**, including those outside the four visual areas (V1, mHV, aHV, lHV). Neurons with iarea values not matching any of the four known areas (including iarea=-1 and iarea=7) are assigned to an "unknown" region category, which is the 5th entry in the brain regions list.

ii.
```python
REGION_NAMES = ["V1", "mHV", "aHV", "lHV", "unknown"]
# ...
def region_index_from_iarea(iarea):
    out = np.full(iarea.shape, REGION_TO_INDEX["unknown"], dtype=np.int16)
    out[iarea == 8] = REGION_TO_INDEX["V1"]
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = REGION_TO_INDEX["mHV"]
    out[(iarea == 3) | (iarea == 4)] = REGION_TO_INDEX["aHV"]
    out[(iarea == 5) | (iarea == 6)] = REGION_TO_INDEX["lHV"]
    return out
```

iii. The agent defaulted to including all neurons with a fallback "unknown" label rather than dropping non-visual-area neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). For each trial, only frames where `ft_trInd == trial` and `ft_CorrSpc == True` are selected. Trials are variable length, starting at corridor entry and ending when the mouse exits the texture.

ii.
```python
idx = np.flatnonzero((ft_tr == tr) & corr)
# ...
neural_trials.append(
    MappedTrialArray(memmap_path, base_shape, ..., tp.row_start, tp.row_stop)
)
```

iii. Alignment is inherent in the frame selection -- corridor frames begin at corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed as the median inter-frame interval across sessions, yielding approximately 314.6 ms. This is stored in the metadata.

ii.
```python
dt = np.diff(ft) * 24.0 * 3600.0
median_dt_s = float(np.median(dt)) if dt.size else np.nan
# ...
median_time_bin_ms = 1000.0 * float(np.median(np.asarray(time_bin_values)))
```

iii. The agent computed the median dt from timestamps (MATLAB datenums converted to seconds), getting ~314.6ms, close to the reference's 1000/3.17 = 315.5ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the timestamp of the sound cue for each trial (in MATLAB datenum format), and `ft`, the timestamp of every imaging frame.

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
# ...
((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32)
```

iii. The agent uses `SoundTime` (absolute timestamp) rather than `SoundFr` (frame number).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the time to sound cue is `(SoundTime[trial] - ft[frame_idx]) * 86400` seconds. This is positive before the cue and negative after, consistent with "time TO sound cue".

ii.
```python
t_frame = ft[idx]
input_trial = np.vstack([
    ((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32),
    ...
])
```

iii. Direct time difference in MATLAB datenum units, converted to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame indices (`idx`) as the neural data for each trial, ensuring temporal alignment.

ii.
```python
t_frame = ft[idx]
((sound_time[tr] - t_frame) * 24.0 * 3600.0)
```

iii. All streams use the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date, extracted from the session key (e.g., `2022_05_30`), and the first session date for each subject.

ii.
```python
date = session_date(raw_key)  # parses date from session key
first_date = {
    subject: min(plan.date for plan in session_plans if plan.subject == subject)
    for subject in subjects
}
```

iii. The date is encoded in the session key filename.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Day of training is computed as the number of **calendar days** since the subject's first session: `(session_date - first_date_for_subject).days`. This yields a float (e.g., 0.0, 1.0, 7.0, ...) broadcast across all time bins of a trial.

ii.
```python
plan.day_of_training = float((plan.date - first_date[plan.subject]).days)
# ...
np.full((T,), plan.day_of_training, dtype=np.float32)
```

iii. Calendar days rather than ordinal session count. The first session is day 0.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the timestamp of trial start for each trial (in MATLAB datenum), and `ft`, the timestamp of each frame.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
# ...
((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The agent uses `Trial_start_time` (absolute timestamp) rather than `StartFr` (frame number).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, time since trial start is `(ft[frame_idx] - Trial_start_time[trial]) * 86400` seconds. Positive after trial start.

ii.
```python
((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32)
```

iii. Direct time difference, converted from MATLAB datenum to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame indices (`idx`) as neural data for each trial.

ii.
```python
t_frame = ft[idx]
```

iii. All streams share the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which flags whether each trial is in a rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
# ...
np.full((T,), float(is_rew[tr]), dtype=np.float32)
```

iii. Directly available in the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0 or 1.0) and broadcast across all time bins of the trial.

ii.
```python
np.full((T,), float(is_rew[tr]), dtype=np.float32)
```

iii. No special processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial's corridor walls.

ii.
```python
stim_idx = canonical_stimulus_index(str(beh["WallName"][tr]))
```

iii. The agent used `WallName` as the primary source, noting that `TrialStim` had placeholder values in some sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are mapped to "canonical" stimulus names via `WALL_TO_CANONICAL`, which maps rock/wood variants to their circle/leaf equivalents (e.g., `rock1` -> `circle1`, `wood1` -> `leaf1`). This yields **8 distinct categories**: `circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2`. The value is per-trial, broadcast across all time bins.

ii.
```python
STIMULUS_CATEGORIES = [
    "circle1", "circle2", "circle3", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
WALL_TO_CANONICAL = {
    "circle1": "circle1", "rock1": "circle1",
    "circle2": "circle2", "rock2": "circle2",
    ...
}
# ...
np.full((T,), stim_idx, dtype=np.int16)
```

iii. The agent discovered 15 unique wall names across all sessions. It paired rock/wood with circle/leaf as functionally equivalent, but preserved the numbered variants (1, 2, 3) and swap variants as separate categories, yielding 8 categories rather than the 4 broad categories (circle, leaf, rock, wood) used by the reference.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame indices of licks) and `LickTrind` (trial index of each lick).

ii.
```python
lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
sel = lick_trials == int(tr)
```

iii. The agent uses both `LickFr` and `LickTrind` to assign licks to specific trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices are rounded to nearest integer, clipped to valid range, and filtered to the current trial using `LickTrind`. A mapping from absolute frame indices to relative within-trial indices is created, and the binary lick array is set to 1 at those positions.

ii.
```python
lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
sel = lick_trials == int(tr)
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
frame_to_rel = np.full((nfr,), -1, dtype=np.int32)
frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
rel = frame_to_rel[lick_frames]
rel = rel[rel >= 0]
lick[np.unique(rel)] = 1
```

iii. The agent verified that nearest-frame assignment had median timing errors of ~0.017s, well within the ~0.31s frame interval.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are mapped to the same within-trial frame indices used for neural data, so alignment is inherent.

ii.
```python
frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
```

iii. All streams share the same frame indices per trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. Directly from the behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to [0, 39.999] decimeters and divided by 10 to get meters, then floored to give 4 bins (0-1m, 1-2m, 2-3m, 3-4m).

ii.
```python
def position_series_for_trial(pos_trial):
    pos_clipped = np.clip(pos_trial, 0.0, 39.999)
    return np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. Straightforward conversion from decimeters to 1-meter bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m. Position is clipped to < 40 dm and divided by 10, floored.

ii.
```python
pos_clipped = np.clip(pos_trial, 0.0, 39.999)
return np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. Matches the task instruction of 4 equal-length 1-m-long spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same frame indices per trial as neural data.

ii.
```python
pos_bins = position_series_for_trial(pos[idx])
```

iii. All streams share frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
```

iii. Directly from behavior data; negative speeds are clipped to 0.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is clipped to non-negative values. **Global** quantiles (25th, 50th, 75th percentiles) are computed across all corridor frames from all sessions. Per-trial speed is binned using `np.digitize` with these global quantile boundaries, yielding 4 bins (q1-q4).

ii.
```python
all_speed = np.concatenate(speed_values).astype(np.float32)
speed_quantiles = np.quantile(all_speed, [0.25, 0.50, 0.75]).astype(np.float32)
# ...
def speed_series_for_trial(speed_trial, speed_quantiles):
    return np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)
```

iii. Global quantiles ensure consistent bin boundaries across sessions, each bin holding approximately 25% of the data globally.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins based on global quartile thresholds. `np.digitize` assigns each speed value to one of 4 bins.

ii.
```python
speed_quantiles = np.quantile(all_speed, [0.25, 0.50, 0.75])
np.digitize(speed_trial, speed_quantiles, right=True)
```

iii. Quartile-based binning ensures each bin contains approximately 25% of data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same frame indices per trial as neural data.

ii.
```python
speed_bins = speed_series_for_trial(speed[idx], speed_quantiles)
```

iii. All streams share frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior array length is set to `len(beh["ft"]) - 1` to approximately match spike frame count. If the actual spike frame count differs by up to 2 frames, trial plans are recomputed using the spike frame count. Lick frames are clipped to valid range. Trials with fewer than 5 frames are dropped.

ii.
```python
nfr = len(beh["ft"]) - 1
# ...
if abs(spk_nfr - plan.nframes_behavior) > 2:
    raise RuntimeError(...)
trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)
# ...
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
```

iii. The agent noted that behavior arrays sometimes have 1-2 more frames than spike arrays and handled this with a tolerance check and re-planning.

## 12-a. What are the most time-consuming steps of the code?

i. Reading and writing the spike data. Each spike file must be loaded, frames extracted, and written to memmap files. The memmap approach distributes I/O across both reading and writing.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
# ...
base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)
```

iii. I/O dominates due to the large spike file sizes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial frame selection loop scans the full frame index array for each trial independently (`np.flatnonzero((ft_tr == tr) & corr)` for each trial), rather than grouping all frames by trial in one pass. The lick assignment also loops per trial.

ii.
```python
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```

iii. These loops are minor compared to I/O cost.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded multiple times: once during `build_session_plans()` (planning phase) and again during `assemble_dataset()` (conversion phase) via `load_behavior(plan)`. Trial plans may also be recomputed if spike frame counts don't match.

ii.
```python
# In build_session_plans():
beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
# In assemble_dataset():
beh = load_behavior(plan)  # loads again
```

iii. The two-pass approach (plan then convert) necessitates reloading behavior data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The behavior file index (`build_behavior_index()`) scans all behavior files and computes metadata (unique walls, finite stim_ids, placeholder counts) for a scoring heuristic, when the reference approach of simply using `Imaging_Exp_info.npy` to map sessions to behavior files is simpler and direct. The memmap infrastructure (custom `MappedTrialArray` class, file writing) adds complexity that is unnecessary if the pickle file is to be loaded in full by the decoder.

ii.
```python
def build_behavior_index():
    candidates = defaultdict(list)
    behavior_files = sorted(...)
    for fp in behavior_files:
        beh = np.load(fp, allow_pickle=True).item()
        for key, dat in beh.items():
            # computes scoring metrics for each candidate
```

iii. The behavior selection heuristic and memmap infrastructure are over-engineered for this use case.
