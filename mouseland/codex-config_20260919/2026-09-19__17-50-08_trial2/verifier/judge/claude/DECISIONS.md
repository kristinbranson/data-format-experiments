# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavior data by globbing all `Beh_*.npy` files in `data/beh/`, loading each as a pickled dictionary. It deduplicates physical sessions by collapsing swap aliases (removing `_swap1`/`_swap2` suffixes). It also loads the experiment registry from `Imaging_Exp_info.npy` and cross-validates that all neural (`spk/`), behavior, and retinotopy files match. For each session, it loads plane-wise spks from `spk/<base>_neural_data.npy` and retinotopy from `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
def behavior_catalog() -> tuple[...]:
    for path in sorted((DATA_ROOT / "beh").glob("Beh_*.npy")):
        loaded = np.load(path, allow_pickle=True).item()
        for key, beh in loaded.items():
            base = physical_base(key)
            if base in sessions:
                # ... validate alias consistency
                continue
            sessions[base] = beh
```

```python
def registry_catalog() -> tuple[...]:
    registry = np.load(DATA_ROOT / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
    for experiment_type, entries in registry.items():
        for entry in entries:
            base = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            aliases[base].append(plain)
```

iii. From CONVERSION_NOTES.md: "Collapsing by physical `<mouse>_<date>_<block>` yields exactly 89 behavior sessions, matching all neural files, with 38,110 unique trials." The AI validates that all data sources are consistent before processing.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the first component of the session base string (e.g., `TX60` from `TX60_2021_05_04_1`), producing 19 unique sorted subjects.

ii.
```python
subjects = sorted({base.split("_")[0] for base in bases})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[base.split("_")[0]] for base in bases], dtype=np.int16)
```

iii. The session base format `<mouse>_<date>_<block>` allows direct extraction of the mouse name as the first underscore-delimited token.

## 1-c. How are the data split into sessions?

i. Each physical recording is identified by the `<mouse>_<date>_<block>` base string. Behavior aliases (swap1/swap2) are collapsed to the physical base. The 89 unique neural files define the session set.

ii.
```python
spk_bases = {p.name.removesuffix("_neural_data.npy") for p in (DATA_ROOT / "spk").glob("*_neural_data.npy")}
if spk_bases != set(behaviors) or spk_bases != set(aliases):
    raise ValueError(...)
```

iii. From CONVERSION_NOTES.md: "Physical-session scope: Include all 89 paper recordings exactly once. Behavior aliases are alternate analyses of identical frames, not additional sessions."

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` (trial index per frame) from the behavior data. For each trial (0 to `ntrials-1`), frames are selected where the trial index matches AND the frame passes the curation mask (`ft_CorrSpc & ft_move > 0`). Trials with no surviving frames are excluded.

ii.
```python
valid = np.isfinite(trial_stamp) & corridor & movement
# ...
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    if frames.size == 0:
        excluded_trials.append(trial)
        continue
```

iii. From CONVERSION_NOTES.md: "Only trials lacking at least one common curated neural/behavior frame will be excluded."

## 1-e. How are trials filtered based on quality controls?

i. The AI drops only trials with zero curated frames (i.e., no frames satisfying `ft_CorrSpc & ft_move > 0` within the common neural/behavior length). No trial length filter is applied. The output shows 38,110/38,110 trials retained across all 89 sessions.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
if frames.size == 0:
    excluded_trials.append(trial)
    continue
```

iii. From CONVERSION_NOTES.md Step 5: "Only trials lacking at least one common curated neural/behavior frame will be excluded. Fail loudly on non-finite neural/input/output values." No length-based filtering is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<base>_neural_data.npy` (a list of plane-wise neuron-by-frame arrays) and `iarea` from `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
def load_selected_spikes(base, mapped):
    path = DATA_ROOT / "spk" / f"{base}_neural_data.npy"
    neural_dict = np.load(path, allow_pickle=True).item()
    planes = neural_dict["spks"]
    # ...
    selected = np.empty((int(mapped.sum()), nframes), dtype=np.float32)
```

iii. "Neural files already contain plane-wise spks; the reference loader concatenates these without computing delta-F/F."

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed (no dF/F, normalization, or smoothing). Plane-wise arrays are concatenated over neurons, filtered by mapped visual areas, and stored as float32. Frames are selected per trial based on the `ft_CorrSpc & ft_move > 0` mask.

ii.
```python
selected = np.empty((int(mapped.sum()), nframes), dtype=np.float32)
# ...
for plane in planes:
    local_mask = mapped[src0:src1]
    count = int(local_mask.sum())
    selected[dst0 : dst0 + count] = plane[local_mask]
# ...
neural = spk[:, frames].copy()
```

iii. From CONVERSION_NOTES.md: "Calcium processing: Load supplied spks directly... No dF/F or further neural normalization for the generic decoder."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their `iarea` maps to V1, mHV, lHV, or aHV. Neurons with `iarea == -1` or `iarea == 7` are excluded. This retains 4,105,393 of 4,691,034 ROIs.

ii.
```python
def area_codes(iarea):
    mapped = np.isin(iarea, [0, 1, 2, 3, 4, 5, 6, 8, 9])
    codes = np.full(iarea.shape, -1, dtype=np.int8)
    codes[iarea == 8] = 0      # V1
    codes[np.isin(iarea, [0, 1, 2, 9])] = 1   # mHV
    codes[np.isin(iarea, [5, 6])] = 2          # lHV
    codes[np.isin(iarea, [3, 4])] = 3          # aHV
    return mapped, codes[mapped]
```

iii. "Retain only mapped V1/mHV/lHV/aHV neurons (4,105,393); exclude -1/7 as outside/unknown visual cortex."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Frames are selected per trial from `ft_trInd` filtered by the `ft_CorrSpc & ft_move > 0` mask. Trials have variable length. No fixed window or padding is applied; `off_start = 0.0` and `off_end = None`.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
```
```python
"temporal_alignment_event": "entry into the 0-4 m textured visual corridor",
"off_start": 0.0,
"off_end": None,
```

iii. "Samples are native 3.17-Hz imaging frames; stationary frames are omitted per the reference analysis."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate of 3.17 Hz is used (~315.46 ms per frame). No temporal rebinning is applied. However, stationary frames (`ft_move <= 0`) are removed, creating temporal gaps within trials.

ii.
```python
FS_HZ = 3.17
MS_PER_FRAME = 1000.0 / FS_HZ
# ...
"time_bin_size": MS_PER_FRAME,
```

iii. "Native imaging bins are nominally 315.46 ms. Selected samples remain native frames but stationary intervals are omitted."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of each imaging frame).

ii.
```python
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. The AI uses `SoundTime` (a timestamp) rather than `SoundFr` (a frame index), computing the time difference directly in seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the time to sound cue is computed as `(SoundTime[trial] - ft[frame]) * 86400` in seconds. The sign convention is positive before the cue and negative after.

ii.
```python
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. From metadata: "time_to_sound_sign: positive before cue, zero at cue, negative after cue."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same `frames` array (frame indices) as the neural data for each trial, ensuring exact temporal alignment.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. All data streams use the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `days` field in the experiment registry entries, or the minimum `sess#` field across aliases if `days` is not present.

ii.
```python
def registry_catalog():
    # ...
    for base, entries in aliases.items():
        explicit_days = [e["days"] for e in entries if "days" in e]
        session_indices = [e["sess#"] for e in entries if "sess#" in e]
        if explicit_days:
            training_day[base] = float(min(explicit_days))
        elif session_indices:
            training_day[base] = float(min(session_indices))
```

iii. From CONVERSION_NOTES.md: "Use explicit `days` when present; otherwise the minimum `sess#` among aliases (avoids treating swap label 2 or naive reuse as another physical day)."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The `days` field value is used directly as a float. It is broadcast (repeated) across all frames of every trial in that session. The range is [0, 15].

ii.
```python
np.full(frames.size, day, dtype=np.float64),
```

iii. The value comes from the experiment registry metadata rather than being computed from session ordering.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (per-trial timestamp of trial start) and `ft` (per-frame timestamps).

ii.
```python
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. The AI uses `Trial_start_time` (a timestamp) rather than `StartFr` (a frame index).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, the elapsed time is `(ft[frame] - Trial_start_time[trial]) * 86400` seconds. This preserves actual elapsed time including stationary intervals (which are removed from the neural data but reflected in the timestamps).

ii.
```python
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. From CONVERSION_NOTES.md: "Temporal inputs use the original timestamps, so elapsed-time gaps are not hidden by curation."

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Uses the same `frames` array as the neural data for each trial.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. Same frame-index alignment as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial boolean/integer flag.

ii.
```python
np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64),
```

iii. Uses `isRew` directly, not the session mode string.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[trial]` is cast to float (0.0 or 1.0) and broadcast across all frames of the trial.

ii.
```python
np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64),
```

iii. "Use per-trial `isRew`, never the session mode string."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial string naming the corridor wall texture.

ii.
```python
category = stimulus_category(str(beh["WallName"][trial]))
```

iii. "Derive physical category from alphabetic prefix of `WallName`."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The leading alphabetic prefix of each `WallName` is extracted via regex and mapped to one of four categories: circle (0), leaf (1), rock (2), wood (3). The value is per-trial, broadcast across all frames.

ii.
```python
def stimulus_category(name: str) -> int:
    match = re.match(r"([A-Za-z]+)", str(name))
    return CATEGORY_TO_CODE[match.group(1).lower()]
```
```python
CATEGORY_VALUES = ["circle", "leaf", "rock", "wood"]
decoder_output[0] = category
```

iii. Uses regex to extract the category prefix rather than a hard-coded lookup table, which is more flexible but functionally equivalent.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the imaging frame number of each lick event.

ii.
```python
lick_float = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
lick_global[lick_frames] = 1
```

iii. Uses `LickFr` converted to integer frame indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array of zeros is created. Lick frame indices from `LickFr` are filtered for validity (finite, non-negative, within range) and used to set corresponding elements to 1. Per-trial licking is extracted from this global array using the trial's frame indices.

ii.
```python
lick_global = np.zeros(common_nframes, dtype=np.int8)
lick_float = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
lick_global[lick_frames] = 1
# ...
decoder_output[1] = lick_global[frames]
```

iii. Binary frame-level licking signal.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Uses the same `frames` array as the neural data for each trial.

ii.
```python
decoder_output[1] = lick_global[frames]
```

iii. Same frame-index alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
```

iii. Uses `ft_Pos` directly.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (converting to meters), floored, and clipped to [0, 3], producing four 1-m bins: [0-1), [1-2), [2-3), [3-4) m.

ii.
```python
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
```

iii. "Four equal 1-m bins spanning textured 0-4 m corridor."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(position_dm / 10)` naturally creates four bins at 0, 1, 2, 3 meter boundaries. Values are clipped to [0, 3] to handle any edge cases.

ii.
```python
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
```

iii. Matches the instruction for "4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same `frames` array as the neural data for each trial.

ii.
```python
position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
```

iii. Same frame-index alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
```

iii. Uses `ft_RunSpeed` directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into four quartile bins using global thresholds computed across all curated frames from all sessions. The thresholds are the 25th, 50th, and 75th percentiles. `np.searchsorted` is used to assign bins.

ii.
```python
def fill_speed_classes(outputs, speeds):
    all_speed = np.concatenate([trial for session in speeds for trial in session]).astype(np.float64)
    thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
    for out_session, speed_session in zip(outputs, speeds, strict=True):
        for out_trial, speed_trial in zip(out_session, speed_session, strict=True):
            labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
            out_trial[3] = labels
```

iii. "Compute global quartiles over all included running frames... Global running-speed quartiles: 12.42242, 25.352615, 40.854578."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quantile thresholds at [0.25, 0.50, 0.75] are computed from all curated frames. `np.searchsorted(..., side='right')` assigns each speed to one of four bins (Q1-Q4). Global counts are approximately equal: [205395, 205394, 205395, 205395].

ii.
```python
thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
```

iii. "Global thresholds so class meanings are comparable across recordings."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same `frames` array as the neural data for each trial.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
```

iii. Same frame-index alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior frame arrays can differ by 1-2 frames; the AI truncates all to the common minimum length. Lick frames are filtered for finiteness and range validity. Non-finite neural or input values raise errors. Sessions with fewer than 2 usable trials raise errors. Trials with zero curated frames are excluded.

ii.
```python
common_nframes = min([neural_nframes] + [len(beh[k]) for k in frame_fields])
# ...
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
# ...
if not np.all(np.isfinite(neural)):
    raise ValueError(...)
```

iii. "Truncate all frame streams to common neural/behavior length; omit only trials with no remaining curated frame."

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files from disk, which total ~404 GiB. The full conversion takes about 12.4 minutes, dominated by neural file I/O.

ii.
```python
spk, neural_nframes, _ = load_selected_spikes(base, mapped)
```

iii. From conversion output: "total 742.95s" for 89 sessions, with individual session neural loads ranging from ~1s to ~20s.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within each session iterates over all trials and selects frames. The speed quartile filling also loops over all sessions and trials. However, these are minor compared to I/O costs.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    # ...
```

iii. The trial loop computes `flatnonzero` for each trial, which rescans the valid mask. A grouped approach (e.g., using `np.unique` on `trial_stamp`) could avoid repeated scans.

## 12-c. What processing does the code repeat multiple times?

i. The behavior catalog loading iterates over all behavior files and validates alias consistency. When swap aliases exist, duplicate consistency checks are performed. The speed class filling requires a second pass over all output arrays.

ii.
```python
# First pass: convert sessions
for index, base in enumerate(bases):
    result = convert_session(...)
# Second pass: fill speed classes
thresholds, speed_counts = fill_speed_classes(output_all, speed_all)
```

iii. The two-pass approach for speed (convert then fill) is necessary since global thresholds require all data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores per-session metadata including detailed registry aliases, behavior file/key provenance, and timing information that the decoder does not use. The `validate_converted` function performs shape/range checks on the full dataset before writing.

ii.
```python
info["registry_aliases"] = aliases[base]
# ...
validate_converted(data)
```

iii. The detailed metadata is documentation rather than computation, and the validation is a correctness check rather than unnecessary processing. The `ft_move > 0` filtering removes ~40% of frames that the reference solution would have kept, which is a significant difference in data volume.
