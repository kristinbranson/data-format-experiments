# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from three directories under `/app/data`: `beh/` (behavior), `spk/` (deconvolved calcium traces), and `retinotopy/` (visual area labels). `Imaging_Exp_info.npy` is loaded first as a master index. The AI collects all experiment entries into "session views" grouped by the unique (mouse, date, block) triplet, then selects one "canonical" view per triplet using a priority scoring function. Behavior is loaded via `np.load` of `Beh_<exp_type>.npy`, spikes from `<session_id>_neural_data.npy`, and retinotopy from `<mouse>_<date>_trans.npz`.

ii.
```python
def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def collect_session_views(exp_info):
    views = defaultdict(list)
    for exp_type, records in exp_info.items():
        beh_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for rec in records:
            triplet = (rec["mname"], rec["datexp"], rec["blk"])
            key = "_".join(triplet)
            ...
            views[triplet].append(SessionView(...))
    return views
```

iii. The AI documented in CONVERSION_NOTES.md that `Imaging_Exp_info.npy` contains 142 experiment entries referencing 89 unique recordings across 19 mice, and that some recordings appear under multiple experiment types. The canonical view selection chooses the "best" behavior view for each unique recording.

## 1-b. How are the data split into subjects?

i. Mouse name is extracted from the triplet's first element (mname). Unique subjects are collected and sorted, then each session gets an index into this list.

ii.
```python
subjects = sorted({triplet[0] for triplet, _view, _exp_types in sessions})
subject_lookup = {name: idx for idx, name in enumerate(subjects)}
...
data["subject_idx"].append(subject_lookup[mouse])
```

iii. The AI noted 19 unique mice in `Imaging_Exp_info.npy`, consistent with the paper's statement of 19 mice.

## 1-c. How are the data split into sessions?

i. A session is one unique (mouse, date, block) triplet. When the same recording appears under multiple experiment types, the AI selects one canonical view via a priority scoring function (`behavior_view_priority`) that ranks by number of finite `stim_id` entries, number of unique walls, and whether it has a `stimtype` field. This yields 89 sessions.

ii.
```python
def choose_canonical_view(views):
    out = []
    for triplet in sorted(views.keys(), ...):
        candidates = []
        for view in views[triplet]:
            beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]
            candidates.append((behavior_view_priority(beh, view.has_stimtype), view))
        candidates.sort(..., reverse=True)
        out.append((triplet, candidates[0][1], sorted(set(exp_types))))
    return out
```

iii. The AI explained that using 89 unique recordings as sessions matches the paper's statement and avoids double-counting from `stimtype` variants.

## 1-d. How are the data split into trials?

i. Trials are defined using `ft_trInd` and `ft_CorrSpc`. For each trial index from 0 to `ntrials-1`, the AI creates a boolean mask of frames matching that trial index AND being inside the corridor space. Only corridor frames are included.

ii.
```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
    if mask.sum() < 2:
        skipped_trials += 1
        continue
```

iii. The AI documented that the corridor-only approach matches the paper's focus on the 0-4 m texture corridor and makes the 4 position bins well-defined.

## 1-e. How are trials filtered based on quality controls?

i. A trial is skipped if: (1) it has fewer than 2 corridor frames, (2) its wall name doesn't map to one of the 4 known families, (3) `SoundTime` or `Trial_start_time` is non-finite, or (4) neural/input/output contain non-finite values. There is NO filtering based on trial length.

ii.
```python
if mask.sum() < 2:
    skipped_trials += 1
    continue
if family not in family_lookup:
    skipped_trials += 1
    continue
if not np.isfinite(cue_time) or not np.isfinite(start_time):
    skipped_trials += 1
    continue
if not (np.all(np.isfinite(neural_trial)) and ...):
    skipped_trials += 1
    continue
```

iii. The conversion output shows 38,110 trials kept with 0 skipped, suggesting all trials pass these filters. Unlike the reference, no trial-length percentile filter is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the `*_neural_data.npy` files (a list of arrays, one per imaging plane) and `iarea` in the `*_trans.npz` retinotopy files.

ii.
```python
def load_spike_planes(triplet):
    obj = np.load(path, allow_pickle=True).item()
    return [np.asarray(x) for x in obj["spks"]]

def load_iarea(triplet):
    return np.load(path, allow_pickle=True)["iarea"]
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. The AI performs substantial additional processing not done by the reference: it computes d-prime selectivity between primary stimulus pair responses, filters for "corridor-responsive" neurons (mean activity for stimulus A or B exceeding grey-space mean), and selects up to 64 top-|d'| neurons per brain region (V1, mHV, lHV, aHV), capped at 256 neurons per session. Neural data is stored as float16.

ii.
```python
def select_neurons(planes, beh, iarea):
    ...
    dp = compute_dprime(plane[:, stim_a_fr], plane[:, stim_b_fr])
    corr_neu = (plane[:, stim_a_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1)) | (
        plane[:, stim_b_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1))
    ...
    strong_pool = region_pool & (np.abs(dp) >= 0.3)
    chosen_pool = strong_pool if strong_pool.any() else region_pool
    ...
    chosen = candidates[:N_PER_REGION]  # N_PER_REGION = 64
```

iii. The AI justified this as keeping the dataset "tractable" and staying close to the paper's selective-neuron logic. However, the reference code does not perform any neuron selection beyond area filtering.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies multiple neuron quality filters: (1) only neurons in V1, mHV, lHV, aHV regions are considered, (2) neurons must be "corridor-responsive" (activity higher during corridor stimuli than gray space), (3) neurons must have finite d-prime, (4) preference for |d'| >= 0.3, and (5) capped at 64 per region. This results in ~249 neurons per session (mean), compared to ~46,000+ in the reference.

ii. See code in 2-b above.

iii. The AI noted this was intentional for tractability. The reference keeps all neurons in the 4 visual areas without any selectivity-based filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). Each trial's neural data is the subset of imaging frames where `ft_trInd == trial_idx` and `ft_CorrSpc == True`. Trials have variable length.

ii.
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays], axis=0
).astype(np.float16, copy=False)
```

iii. Consistent with corridor-entry alignment specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate (~3.17 Hz, ~314.7 ms) is the native temporal resolution. The AI computes the exact median frame interval across sessions rather than using a fixed constant.

ii.
```python
def compute_frame_timing(sessions):
    dts = []
    for triplet, view, _exp_types in sessions:
        beh = load_behavior(view)
        ft = np.asarray(beh["ft"], dtype=float)
        dts.append(float(np.nanmedian(np.diff(ft)) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(dts)), dts
```

iii. The AI reported median frame interval of 314.694 ms, consistent with the reference's 1000/3.17 = 315.46 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum of the sound cue for each trial) and `ft` (the MATLAB datenum of each imaging frame).

ii.
```python
cue_time = float(beh["SoundTime"][trial_idx])
trial_times = ft[mask]
(cue_time - trial_times) * SECONDS_PER_DAY
```

iii. The AI uses `SoundTime` rather than `SoundFr` (the frame number used by the reference). Both represent the same event but in different units (absolute time vs. frame index).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Computed as `(SoundTime[trial] - ft[frame]) * 86400` seconds. This gives positive values before the cue and negative after, consistent with the name "time TO sound cue."

ii.
```python
(cue_time - trial_times) * SECONDS_PER_DAY
```

iii. The reference uses `SoundFr` interpolated onto frame times and computes `cue_time - frame_time`. Both yield the same sign convention and similar values, though the interpolation approach differs slightly.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Uses the same frame mask (`mask`) as the neural data for the trial, so alignment is automatic.

ii.
```python
trial_times = ft[mask]
# same mask used for neural_trial
neural_trial = np.concatenate([selected_plane[:, mask] ...])
```

iii. Frame-based alignment ensures consistency.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date (`datexp`) parsed into a Python `date` object for each session, computed per-subject.

ii.
```python
def subject_day_map(sessions):
    per_subject = defaultdict(list)
    for triplet, _view, _exp_types in sessions:
        mouse, datexp, blk = triplet
        per_subject[mouse].append((parse_date(datexp), int(blk), triplet))
    for mouse, entries in per_subject.items():
        entries.sort()
        first_date = entries[0][0]
        for dt, blk, triplet in entries:
            out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
```

iii. The AI uses calendar days since first recording (e.g., 0, 7, 9, 16, ... up to 92) plus a small block offset (0.01 per block). The reference instead counts the ordinal session number (0, 1, 2, ... up to 7).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days elapsed since the subject's first recording date, with a 0.01 offset per block number. Broadcast across all frames of the trial.

ii.
```python
out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
...
np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. This produces values ranging from 0 to 92, reflecting actual elapsed days rather than session count. The reference produces values 0-7 reflecting session ordinal.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (MATLAB datenum of each trial's start) and `ft` (MATLAB datenum of each imaging frame).

ii.
```python
start_time = float(beh["Trial_start_time"][trial_idx])
(trial_times - start_time) * SECONDS_PER_DAY
```

iii. The reference uses `StartFr` (frame number) interpolated onto frame times instead.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(ft[frame] - Trial_start_time[trial]) * 86400` seconds. Positive values increase with time since trial start.

ii.
```python
(trial_times - start_time) * SECONDS_PER_DAY
```

iii. Both AI and reference produce time in seconds since trial start, with similar values.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame mask as neural data.

ii.
```python
trial_times = ft[mask]
```

iii. Frame-based alignment ensures consistency.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial reward availability flag.

ii.
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. Same variable as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0 or 1.0) via `float(bool(...))`, broadcast across all frames of the trial.

ii.
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. Functionally equivalent to the reference's `beh['isRew'].astype(int)`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial's corridor wall.

ii.
```python
wall_name = str(beh["WallName"][trial_idx])
family = family_from_wall_name(wall_name)
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses a regex-based prefix extraction function (`family_from_wall_name`) to map wall names to categories (circle, leaf, rock, wood). The value is per-trial, broadcast across frames.

ii.
```python
def family_from_wall_name(name: str) -> str:
    lowered = name.lower()
    for prefix in ("circle", "leaf", "rock", "wood", "brick"):
        if lowered.startswith(prefix):
            return prefix
    ...

family_values = ["circle", "leaf", "rock", "wood"]
family_lookup = {name: idx for idx, name in enumerate(family_values)}
stim_out = np.full(mask.sum(), family_lookup[family], dtype=np.int16)
```

iii. The reference uses a hardcoded dictionary (`TEXTURE`) mapping all 15 wall names to 4 categories. Both approaches produce the same mapping for all observed wall names.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of each lick event in the session.

ii.
```python
def build_lick_frame_vector(beh, nfr):
    lick_vec = np.zeros(nfr, dtype=np.int8)
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_idx = np.asarray(np.floor(lick_fr), dtype=int)
    valid = (lick_idx >= 0) & (lick_idx < nfr)
    lick_vec[lick_idx[valid]] = 1
    return lick_vec
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are floored to integers, then used to set a binary vector to 1 at those frame positions. Frames with at least one lick are 1, others 0.

ii. See code in 8-a above.

iii. The reference uses `.astype(int)` which truncates toward zero (equivalent to floor for positive values). Both produce a binary per-frame licking vector.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is computed for all frames, then indexed with the same boolean mask used for neural data.

ii.
```python
lick_out = lick_vec[mask].astype(np.int16)
```

iii. Frame-based alignment consistent with neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)[:nfr]
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 (converting decimeters to meters), floored, and clipped to [0, 3], producing 4 bins of 1 m each.

ii.
```python
def position_to_bin(pos):
    bins = np.floor(np.asarray(pos, dtype=float) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. Functionally equivalent to the reference's `np.clip(beh['ft_Pos'][:n_frames] // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1-m bins: [0-1m), [1-2m), [2-3m), [3-4m].

ii. See code in 9-b.

iii. Matches the reference and instruction specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed with the same boolean mask as neural data.

ii.
```python
pos_out = position_to_bin(ft_pos[mask])
```

iii. Frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)[:nfr]
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
```

iii. Same source as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values from all corridor frames across ALL sessions are collected, then assigned to 4 bins by GLOBAL rank-based quartiles using `np.array_split` on sorted indices. This is done as a post-processing step after all sessions are converted.

ii.
```python
def assign_rank_speed_bins(all_speed):
    order = np.argsort(all_speed, kind="mergesort")
    speed_bins = np.empty(all_speed.size, dtype=np.int16)
    for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
        speed_bins[idx_chunk] = bin_idx
    ...
```

iii. The reference computes quartiles PER SESSION on the kept frames of that session. The AI computes them GLOBALLY across all sessions. Both use rank-based splitting to handle tied zero-speed values.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins assigned by global rank quartiles, each containing exactly 25% of all corridor frames across the dataset.

ii. See code in 10-b.

iii. The instruction says "4 bins, each corresponding to 25% of the data." The AI interprets "the data" as the entire dataset globally; the reference interprets it per-session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are collected from the same frame mask as neural data, then globally binned in a post-pass.

ii.
```python
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
# ... later global assignment
data["output"][session_idx][trial_idx] = np.vstack(
    [data["output"][session_idx][trial_idx], trial_speed_bins[np.newaxis, :]]
)
```

iii. Frame-based alignment maintained.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior arrays are trimmed to neural frame count via `trim_behavior_framewise`, (2) non-finite `ft_trInd` values are handled with a `valid_frame_trial` mask, (3) trials with non-finite `SoundTime` or `Trial_start_time` are skipped, (4) trials with non-finite neural/input/output values are skipped, (5) lick frames beyond nfr are filtered out.

ii.
```python
def trim_behavior_framewise(beh, nfr):
    trimmed = dict(beh)
    for key, value in beh.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= nfr:
            if key.startswith("ft_") or key in {"ft", "AftCueFr", "BefCueFr"}:
                trimmed[key] = value[:nfr]
    return trimmed

valid_frame_trial = np.isfinite(frame_trial_raw)
if not np.isfinite(cue_time) or not np.isfinite(start_time):
    skipped_trials += 1
    continue
```

iii. More defensive than the reference, which primarily just trims behavior to neural frame count and drops licks beyond nfr.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files and performing neuron selection (d-prime computation) are the most time-consuming steps. The full conversion took 1373.64 seconds (~23 minutes) for 89 sessions, averaging ~15.4 seconds per session.

ii.
```python
planes = load_spike_planes(triplet)
# ... followed by select_neurons which computes dprime on full spike arrays
dp = compute_dprime(plane[:, stim_a_fr], plane[:, stim_b_fr])
```

iii. The neuron selection step adds overhead not present in the reference, since it requires computing d-prime statistics on the full neuron arrays before selecting a subset.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop iterates over each trial individually to build per-trial arrays. The neuron selection loops over planes and regions. The behavior view loading in `collect_session_views` and `choose_canonical_view` re-loads behavior files multiple times.

ii.
```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
    ...
```

iii. The per-trial loop is standard and hard to avoid given variable-length trials.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded multiple times: once during `collect_session_views` to build view lists, once during `choose_canonical_view` to score priorities, once during `compute_frame_timing`, optionally during `choose_sample_sessions`, and finally during actual conversion. The same behavior file for a given session could be loaded 3-4 times.

ii.
```python
# In collect_session_views:
beh_all = np.load(beh_path, allow_pickle=True).item()
# In choose_canonical_view:
beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]
# In compute_frame_timing:
beh = load_behavior(view)
# In convert_dataset:
beh = load_behavior(view)
```

iii. This is inefficient but not functionally incorrect.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The d-prime-based neuron selection computes statistics (corridor responsiveness, selectivity) that are only used for filtering and are not part of the output format. The `selection_meta` dictionary is stored in metadata but not used by the decoder. The `compute_frame_timing` pre-pass loads all behavior files just to compute median frame intervals.

ii.
```python
meta = {
    "stimulus_pair_for_selection": [stim_a, stim_b],
    "corridor_responsive_fraction": ...,
    "abs_dprime_ge_0p3_fraction": ...,
    "selected_fraction": ...,
}
```

iii. The neuron selection itself is the largest piece of unnecessary processing, as the reference simply keeps all neurons in the 4 visual areas without any selectivity-based filtering.
