# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavior data by scanning the `beh/` directory for all files matching `Beh_*.npy`, loading each one, and merging entries by their base session key (stripping swap suffixes). Spike data is loaded per-session from `spk/<base_key>_neural_data.npy`, and retinotopy from `retinotopy/<subject>_<date>_trans.npz`. Unlike the reference, the AI does NOT use `Imaging_Exp_info.npy` as a master index; instead it directly iterates over behavior file keys.

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
                continue
            # merge stim_id fields
            ...
```

```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```

iii. The agent stated (step 61): "I've confirmed the duplicated entries are the same recordings with alternate stimulus annotations... I'm now collapsing those to one recording apiece." The agent chose to merge directly from behavior files rather than using the experiment info index.

## 1-b. How are the data split into subjects?

i. The subject (mouse name) is parsed from the first component of each session key string. Subjects are collected as sorted unique names, and `subject_to_idx` maps each to an integer index.

ii.
```python
def parse_subject_and_date(base_key: str) -> tuple[str, datetime]:
    parts = base_key.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    return subject, date

subjects = sorted(per_subject_dates)
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent did not provide specific justification; the subject name is directly available in the session key.

## 1-c. How are the data split into sessions?

i. A session is identified by the base key (mouse_date_block) after stripping swap suffixes. Duplicate entries across behavior files are merged into a single record per physical recording. The sorted unique base keys define the session list.

ii.
```python
def parse_base_session_key(key: str) -> str:
    parts = key.split("_")
    if parts[-1].startswith("swap"):
        return "_".join(parts[:-1])
    return key

session_keys = sorted(behaviors)
```

iii. The agent stated (step 38, 61): "Some recordings are reused across multiple figure-analysis files... I've confirmed the duplicated entries are the same recordings with alternate stimulus annotations, mainly in the newer naive/grating cohorts."

## 1-d. How are the data split into trials?

i. For each session, the AI iterates over trials 0 through `ntrials-1`. A trial's frames are those where `ft_trInd == trial` AND `ft_CorrSpc` is true AND `ft_move > 0` (the running filter). If no frames survive these filters for a trial, it is skipped.

ii.
```python
for tr in range(int(beh["ntrials"])):
    mask = (trial_idx == tr) & move_corr_mask
    if not np.any(mask):
        continue
```
where:
```python
move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```

iii. The agent stated (step 31): "The remaining gap is whether the conversion should keep all frames per trial or only the running periods the paper analyzed." and (step 79): "keep corridor-aligned trial segments." The agent chose to apply the paper's running-only filter (`ft_move > 0`).

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped only if they have zero frames after the corridor + running filter. There is no trial length outlier filter (unlike the reference which drops trials beyond the 99th percentile in length).

ii.
```python
if not np.any(mask):
    continue
```
and sessions with fewer than 2 valid trials raise an error:
```python
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {base_key} has fewer than two valid trials after filtering.")
```

iii. The agent did not explicitly discuss trial length filtering. By applying the running-only filter, extremely long stationary periods are already excluded frame-by-frame rather than by dropping whole trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (a list of per-plane neuron-by-frame arrays) and `iarea` from `retinotopy/<subject>_<date>_trans.npz`.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```

iii. Same as the reference: the deconvolved calcium traces are used directly.

## 2-b. How is the `neural` data processed?

i. The AI applies several processing steps beyond simply extracting frames: (1) Only neurons in the four visual areas (V1, mHV, lHV, aHV) are considered. (2) Among those, only "responsive" neurons (corridor mean activity > gray-space mean activity, both restricted to running frames) are kept. (3) These are ranked by corridor-frame variance and capped at 128 per area (MAX_NEURONS_PER_AREA), giving up to 512 neurons per session. The selected neural data is stored as float16.

ii.
```python
corr_mean = plane[:, corr_mask].mean(axis=1)
gray_mean = plane[:, gray_mask].mean(axis=1)
responsive = corr_mean > gray_mean

# ... rank by variance, cap at MAX_NEURONS_PER_AREA per region
order = np.lexsort((row_arr, plane_arr, -score_arr))
top = order[:MAX_NEURONS_PER_AREA]
```

```python
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The agent stated (step 71): "One practical constraint matters here: these raw recordings are large enough that an unfiltered trial-by-trial export could be unmanageable." and (step 79): "retain a deterministic, paper-motivated subset of retinotopically assigned corridor-responsive neurons per session so the dataset stays trainable."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters are applied: (1) Neurons outside the four visual areas are dropped. (2) Among visual-area neurons, only those with higher mean activity during corridor running than gray-space running are kept ("responsive" filter). (3) A cap of 128 neurons per area is enforced, selecting by highest corridor-frame variance.

ii.
```python
region_mask = np.isin(plane_areas, AREA_CODES[region])
keep = responsive & region_mask
# ...
top = order[:MAX_NEURONS_PER_AREA]
```

iii. The agent justified the neuron cap as necessary for tractability (step 71, 87). The responsiveness filter was motivated by the paper's analyses which focused on corridor-responsive neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Only running frames within the corridor are kept per trial. The first retained frame of a trial corresponds to the first running frame after corridor entry. Trials have variable length.

ii.
```python
mask = (trial_idx == tr) & move_corr_mask
selected_frames = frame_numbers[mask]
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. The agent stated the alignment is to corridor entry, consistent with the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed empirically as the median inter-frame interval across all sessions, converted to milliseconds. This gives approximately 315 ms (matching the ~3.17 Hz imaging rate).

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

iii. The agent did not specifically discuss this; it's computed from data rather than hard-coded.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame indices of retained frames, combined with a per-session `dt_seconds` (median inter-frame interval).

ii.
```python
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. The agent initially used absolute timestamps but corrected to frame-index-based timing after the verifier flagged implausible values (step 125): "I'm switching those inputs to frame-index timing (StartFr/SoundFr)."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundFr - frame_index) * dt_seconds`, where `dt_seconds` is the median inter-frame interval. This gives positive values before the cue and negative after, matching the "time TO sound cue" semantics.

ii.
```python
dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. The agent corrected this after initial verification issues (step 125, 127).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame mask (`move_corr_mask` intersected with trial index) as the neural data, so it is aligned by construction.

ii.
```python
selected_frames = frame_numbers[mask]
time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. All data streams share the same frame mask.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session keys, which contain the date. The AI parses the date from each session key and computes calendar days since the subject's first imaging session.

ii.
```python
def collect_subject_info(session_keys):
    per_subject_dates = defaultdict(list)
    for key in session_keys:
        subject, date = parse_subject_and_date(key)
        per_subject_dates[subject].append(date)
    first_dates = {subject: min(dates) for subject, dates in per_subject_dates.items()}
    day_since_first = {}
    for key in session_keys:
        subject, date = parse_subject_and_date(key)
        day_since_first[key] = float((date - first_dates[subject]).days)
    return subjects, subject_to_idx, day_since_first
```

iii. The agent stated (step 138): "day_of_training is defined as calendar days since each subject's first imaging session because the source metadata does not provide a consistent explicit training-day count for every recording."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day is computed as the number of calendar days between the session date and the subject's earliest session date. This differs from the reference which counts the ordinal number of recording sessions (0, 1, 2, ...). The value is broadcast as a constant across all time bins of each trial.

ii.
```python
day_since_first[key] = float((date - first_dates[subject]).days)
# ...
day_arr = np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. The agent justified this as: "a consistent explicit training-day count was not available for every recording."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the frame number of corridor entry for each trial) and the frame indices of retained frames, combined with `dt_seconds`.

ii.
```python
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. Corrected after initial verification issues (step 125).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(frame_index - StartFr) * dt_seconds`, giving seconds since corridor entry. Positive after trial start, near zero at the start.

ii.
```python
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. The agent uses frame-index arithmetic with a constant dt rather than interpolating actual frame timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame mask as neural data, aligned by construction.

ii.
```python
selected_frames = frame_numbers[mask]
time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. Same frame selection for all data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in a rewarded corridor.

ii.
```python
reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. No specific justification needed; directly available from the data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is cast to float (0.0 or 1.0) and broadcast across all time bins of the trial.

ii.
```python
reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. No processing beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
wall_name = str(beh["WallName"][tr])
stim_category = category_to_idx[wall_to_category(wall_name)]
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name is parsed with a regex to extract the alphabetic prefix (e.g., "circle" from "circle1"), lowercased, and mapped to one of four categories: circle, leaf, rock, wood. The category index is broadcast across all time bins.

ii.
```python
def wall_to_category(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(name))
    category = match.group(1).lower()
    return category

all_categories = sorted({wall_to_category(wall_name) for beh in behaviors.values() for wall_name in ...})
category_to_idx = {category: idx for idx, category in enumerate(all_categories)}
visual_category = np.full(mask.sum(), stim_category, dtype=np.int16)
```

iii. The regex-based approach derives the same four categories as the reference's hardcoded lookup table.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of each lick in the session.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
if lick_fr.size:
    lick_binary[np.unique(lick_fr)] = 1
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array is created (0 = no lick, 1 = lick). Lick frames are truncated to int, filtered to valid range, deduplicated with `np.unique`, and set to 1.

ii.
```python
lick_binary = np.zeros(nfr, dtype=np.int8)
lick_fr = np.asarray(beh["LickFr"], dtype=int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
if lick_fr.size:
    lick_binary[np.unique(lick_fr)] = 1
```

iii. Essentially the same as the reference approach.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is indexed with the same frame mask used for neural data.

ii.
```python
lick = lick_binary[mask].astype(np.int16)
```

iii. Aligned by shared frame mask.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to [0, 39.999], divided by 10 (converting decimeters to meters), floored, and clipped to [0, 3] to produce four 1-meter bins.

ii.
```python
pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. Functionally equivalent to the reference's `np.clip(ft_Pos // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m, matching the instructions.

ii.
```python
["0-1m", "1-2m", "2-3m", "3-4m"]
pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. Same binning as the reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed with the same frame mask used for neural data.

ii.
```python
pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. Aligned by shared frame mask.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile edges across ALL sessions (restricted to running frames in corridor), using `np.quantile` at [0.25, 0.5, 0.75]. Then `np.digitize` assigns each frame's speed to one of four bins. This differs from the reference which computes rank-based quartiles per session.

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

speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
```

iii. The agent used global quantile edges rather than per-session rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by global quantile edges at 25th, 50th, and 75th percentiles of running speeds across the entire dataset.

ii.
```python
def describe_speed_bins(edges: np.ndarray) -> list[str]:
    return [
        f"<= {edges[0]:.2f} cm/s",
        f"{edges[0]:.2f}-{edges[1]:.2f} cm/s",
        f"{edges[1]:.2f}-{edges[2]:.2f} cm/s",
        f"> {edges[2]:.2f} cm/s",
    ]
```

iii. The instructions say "each corresponding to 25% of the data," which could be interpreted globally or per-session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed with the same frame mask used for neural data.

ii.
```python
speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
```

iii. Aligned by shared frame mask.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior is cut to the number of neural frames. Lick frames outside the valid range or negative are dropped. Trials with zero valid frames after filtering are skipped. Sessions with fewer than 2 valid trials raise an error (halting, not skipping). Neuron count mismatches between spikes and retinotopy also raise errors.

ii.
```python
nfr = selected_spk.shape[1]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
if not np.any(mask):
    continue
if len(neural_trials) < 2:
    raise RuntimeError(...)
if nneu_total != iarea.shape[0]:
    raise RuntimeError(...)
```

iii. The agent handles edge cases but raises errors rather than gracefully skipping problematic sessions, which could cause the entire conversion to fail if any session has issues.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (large .npy files) and the neuron selection process (computing means and variances over corridor/gray frames per plane).

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
# ... per-plane mean/variance computation for neuron selection
corr_mean = plane[:, corr_mask].mean(axis=1)
gray_mean = plane[:, gray_mask].mean(axis=1)
corr_var = plane[:, corr_mask].var(axis=1)
```

iii. The agent noted throughput of ~8-9 sessions per 30 seconds (step 115).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron selection iterates over planes and brain regions with nested Python loops rather than vectorizing across all neurons at once. The trial loop also processes one trial at a time.

ii.
```python
for plane_idx, plane in enumerate(spk_planes):
    # ...
    for region in BRAIN_REGIONS:
        region_mask = np.isin(plane_areas, AREA_CODES[region])
        # ...
```

iii. No specific justification from the agent.

## 12-c. What processing does the code repeat multiple times?

i. The behavior merge step loads all behavior files upfront, but then during session processing `ft_move` masking is recomputed. The `parse_subject_and_date` function is called multiple times for the same key in different functions (`collect_subject_info` calls it twice per key).

ii.
```python
# Called in collect_subject_info
for key in session_keys:
    subject, date = parse_subject_and_date(key)
    per_subject_dates[subject].append(date)
# ... and again:
for key in session_keys:
    subject, date = parse_subject_and_date(key)
    day_since_first[key] = float((date - first_dates[subject]).days)
```

iii. No specific justification from the agent.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `stim_id` merging in `merge_behavior_records` fills NaN values in `stim_id` arrays, but `stim_id` is never used in the conversion -- only `WallName` is used for stimulus category. The computation of `compute_speed_edges` iterates over all sessions twice (once for edges, once during session processing). The `describe_speed_bins` function creates descriptive labels that are stored but not used by the decoder.

ii.
```python
# stim_id merging - never used downstream
stim_old = np.asarray(merged[base_key]["stim_id"], dtype=float)
stim_new = np.asarray(record["stim_id"], dtype=float)
fill = np.isnan(stim_old) & ~np.isnan(stim_new)
stim_old[fill] = stim_new[fill]
merged[base_key]["stim_id"] = stim_old
```

iii. No specific justification from the agent.
