# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories under `data/`: `beh/` for behavior, `spk/` for neural traces, and `retinotopy/` for area assignments. It first loads `Imaging_Exp_info.npy` to discover experiment types, then iterates over all `Beh_*.npy` files to collect behavior records keyed by session ID. Neural data and retinotopy are loaded per session. The AI deduplicates recordings to 89 unique bases and selects one representative behavior key per recording (preferring non-swap keys).

ii.
```python
exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
# ...
for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
    beh_dict = np.load(beh_path, allow_pickle=True).item()
    for key, record in beh_dict.items():
        base = "_".join(key.split("_")[:5])
```

```python
spk = load_spike_matrix(root, spec.base)
# ...
spk_item = np.load(path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
```

iii. The AI documents in CONVERSION_NOTES that the master index lists 142 experiment-session records across 23 experiment types, collapsing to 89 unique neural recordings matching the paper's "89 recordings in 19 mice." The AI iterates over behavior files directly rather than through the experiment info index, selecting the best key per base using a scoring function.

## 1-b. How are the data split into subjects?

i. The mouse name is parsed from the first part of the recording base string (e.g., "TX108" from "TX108_2023_03_25_1"). Subjects are collected as sorted unique names, and `subject_idx` maps each session to its subject index.

ii.
```python
subject, date, blk = parse_base(base)
# ...
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
```

iii. The AI states 19 subjects are expected after deduplication, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is one unique recording base `<mouse>_<date>_<blk>`, giving 89 unique sessions. When multiple behavior keys exist for the same base (e.g., swap variants), the AI selects the "best" key using a scoring function that penalizes swap keys and prefers keys with more unique stimuli.

ii.
```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    # ...
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)

if base not in selected:
    selected[base] = SessionSpec(...)
else:
    current = selected[base]
    if session_key_score(key, record) < session_key_score(current.key, current.record):
        current.key = key
        current.record = record
```

iii. The AI documents that some sessions are reused across multiple experiment labels and that swap sessions reuse one neural recording with two behavior keys. The scoring function picks the most informative key.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd` (trial index per frame). The AI keeps frames where `ft_CorrSpc` is true AND `ft_move > 0` (running movement detected), matching the paper's analysis that "only considered timepoints during running for analysis."

ii.
```python
def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0
    # ...
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
```

iii. The AI justifies using the running mask (`ft_move > 0`) because the paper states "We only considered timepoints during running for analysis" and the reference code uses `VRmove = beh['ft_move'][:nfr] > 0` in position interpolation.

## 1-e. How are trials filtered based on quality controls?

i. Three quality filters are applied: (1) trials with fewer than 5 retained timepoints are dropped, (2) trials with retained duration > 60 seconds from trial start are dropped, and (3) trials with inter-frame gaps > 10 seconds are dropped. This removes 2,217 trials from the full dataset.

ii.
```python
MIN_TRIAL_TIMEPOINTS = 5
MAX_RETAINED_TRIAL_DURATION_S = 60.0
MAX_RETAINED_INTERFRAME_GAP_S = 10.0

def trial_passes_quality_filters(record, trial_idx, frame_idx, nfr):
    if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
        return False, "too_few_timepoints"
    retained_duration_s = float((frame_times[-1] - float(record["Trial_start_time"][trial_idx])) * SEC_PER_DAY)
    if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
        return False, "retained_duration_gt_60s"
    if frame_idx.size > 1:
        max_gap_s = float(np.max(np.diff(frame_times)) * SEC_PER_DAY)
        if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
            return False, "interframe_gap_gt_10s"
```

iii. The AI documents in CONVERSION_NOTES that extreme `time_to_sound_cue_s` and `time_since_trial_start_s` ranges were caused by raw trials with sparse late running segments, motivating these filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which is a list of arrays per imaging plane concatenated along the neuron dimension. The visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)

def load_region_index(root: Path, base: str, nneurons: int) -> np.ndarray:
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=float)
```

iii. The AI notes the stored `spks` arrays are deconvolved fluorescence traces, matching the paper statement.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are loaded as float32 (not float16 as in the reference). After loading, a stimulus-selective neuron subset is selected per session using a d-prime-based method, keeping only the top 5% positive and 5% negative selectivity neurons per brain area. The remaining neuron traces are sliced by the trial's retained frame indices.

ii.
```python
spk = load_spike_matrix(root, spec.base)
# ...
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
# ...
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The AI justifies the selective export as necessary for "tractable full-dataset decoding" and states it follows the paper's coding-direction analysis logic. CONVERSION_NOTES Step 5 decision 10 states: "Select neurons using the paper's 5% positive / 5% negative per-area logic."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Neurons are assigned to brain regions V1, mHV, lHV, aHV based on `iarea`. (2) Within each area, a d-prime selectivity metric is computed on running corridor frames comparing two reference stimuli, and only corridor-responsive neurons in the top 5% positive and bottom 5% negative d' are retained. If no reference stimuli pair is found, all mapped-area neurons are kept as fallback.

ii.
```python
def select_decoder_neurons(spk, record, region_idx):
    stim_pair = choose_reference_stimuli(record)
    if stim_pair is None:
        return np.arange(spk.shape[0], dtype=np.int32)
    # ...
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    for area in range(4):
        candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
```

iii. The AI documents this is based on the reference code's `Get_dprime_selective_neuron` and `Get_coding_direction` functions, reducing from ~52,000 to ~3,400 neurons per session on average.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). Each trial's neural data consists of the frames identified by the trial frame indices (corridor + running mask), which start at corridor entry. Trials are variable length.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The AI notes alignment is to corridor entry / trial start, and off_start is 0.0 with off_end as None due to variable trial lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate (~3.17 Hz, ~314.7 ms per frame) is preserved. The time bin size is stored as the median frame interval across sessions.

ii.
```python
"time_bin_size": float(np.median(frame_dt_medians)),
```

iii. The AI notes that raw frame timestamps provide a consistent ~314.7 ms bin and no resampling is needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (frame timestamps).

ii.
```python
def session_trial_info(record, trial, frame_idx):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. The AI uses the raw `SoundTime` timestamp (in MATLAB datenum format) and computes `SoundTime - frame_time`, converting from days to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the time to sound cue is `(SoundTime[trial] - ft[frame_idx]) * SEC_PER_DAY`. Positive values mean before the cue, negative values mean after.

ii.
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The AI documents this as a direct timestamp difference converted to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same `frame_idx` array as the neural data, so the time-to-cue vector has one value per retained neural frame.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. Alignment is inherent because both neural and input data index the same frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date parsed from the recording base string, relative to the first recording date for that mouse.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    # ...

specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
first_date_by_subject: dict[str, datetime] = {}
for spec in specs:
    first_date_by_subject.setdefault(spec.subject, spec.date)
    spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. The AI notes the paper does not define this variable, but the decoder task requires it and the raw dates provide a reproducible continuous proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The training day is the number of calendar days elapsed since the mouse's first recorded imaging session. For example, if the first session is on day 0, a session 7 calendar days later has training_day=7.0. This value ranges from 0.0 to 92.0, whereas the reference (which counts session ordinals) ranges from 0 to 7.

ii.
```python
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
# ...
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. The AI states this is "derived from raw session dates to remain objective and reproducible."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (per-trial timestamp) and `ft` (per-frame timestamp).

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. The AI uses the raw `Trial_start_time` timestamp rather than `StartFr` (start frame number).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame: `(ft[frame_idx] - Trial_start_time[trial]) * SEC_PER_DAY`. This gives time in seconds since trial start, always non-negative.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. Straightforward timestamp subtraction converted to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same `frame_idx` as the neural data.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
```

iii. Alignment is inherent from shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean indicating whether the trial is in a rewarded corridor.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. The AI notes unsupervised/naive sessions have all-zero `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[trial]` is cast to float (0.0 or 1.0) and broadcast across all timepoints of the trial.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. No further processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial wall texture name, and `UniqWalls`, the unique wall names across sessions.

ii.
```python
visual_categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
stim_idx = np.full(frame_idx.size, visual_to_idx[str(spec.record["WallName"][trial_idx])], dtype=np.int16)
```

iii. The AI collects all unique wall names across sessions and uses them as individual categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is mapped to an index into the global sorted list of 15 unique wall names (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5). This gives 15 categories rather than the 4 base textures used by the reference.

ii.
```python
visual_categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
# Results in 15 categories
```

iii. The AI treats each wall name as its own stimulus category rather than grouping by base texture.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of licks) and `LickTrind` (trial index of each lick).

ii.
```python
def build_lick_frame_lookup(record: dict, nfr: int) -> dict[int, np.ndarray]:
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    # ...
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```

iii. The AI uses both `LickFr` and `LickTrind` to build a per-trial lick frame lookup.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames are looked up from the per-trial dictionary, and `np.isin` checks which retained frames had a lick event, producing a binary vector.

ii.
```python
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Binary lick/no-lick per retained frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick events are matched to the same retained frame indices used for neural data, so alignment is inherent.

ii.
```python
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Same frame indexing as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Directly from the frame-level position array.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is clipped to [0, 39.999] and floor-divided by 10 to get 4 bins of 1 m each.

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Converts decimeters to 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 dm gives bins 0-1m, 1-2m, 2-3m, 3-4m. Position is clipped to [0, 39.999] first to prevent bin 4.

ii.
```python
pos = np.clip(pos_decimeters, 0.0, 39.999)
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The clipping at 39.999 ensures positions at exactly 40 dm don't exceed bin 3.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same `frame_idx` as neural data.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Same frame indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Directly from the frame-level speed array.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized using global quartile edges computed across ALL retained frames in ALL retained sessions. The edges are computed once in `prepare_session_specs()` from all valid running-corridor frames.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
# ...
def digitize_speed(speed, edges):
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. The AI states "the decoder task specifies 25% bins, so edges must be computed from the full retained dataset, not per session."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three quartile edge values split the speed into 4 bins. `np.searchsorted` with `side="right"` assigns each speed to the appropriate bin.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
bins = np.searchsorted(edges, speed, side="right")
```

iii. Global edges: [13.83, 26.68, 41.84].

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same `frame_idx` as neural data.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Same frame indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior arrays are truncated to the neural frame count (behavior is slightly longer by 1-3 frames). Lick frames with NaN or out-of-range values are filtered. Trials with fewer than 5 retained frames, duration > 60s, or inter-frame gaps > 10s are dropped. Sessions with < 2 valid trials are dropped (none actually dropped).

ii.
```python
frame_idx = frame_idx[frame_idx < nfr]
# ...
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
lick_frames = lick_frames[valid].astype(int)
# ...
valid = (lick_frames >= 0) & (lick_frames < nfr)
```

iii. The AI documents the behavior-neural length mismatch and truncation approach matching the reference code convention.

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural spike files (each session's full spike matrix must be loaded before neuron selection) and computing the d-prime selectivity for neuron selection.

ii.
```python
spk = load_spike_matrix(root, spec.base)
# Full spike matrix loaded, then subset selected
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
```

iii. The conversion log shows ~5-16 seconds per session, totaling 635s for all 89 sessions.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `select_decoder_neurons` function loops over 4 brain areas to compute percentiles, which could potentially be vectorized. The `build_trial_frame_indices` function loops over trials individually.

ii.
```python
for area in range(4):
    candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
    # ...
```

```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]
```

iii. These loops are minor compared to I/O costs.

## 12-c. What processing does the code repeat multiple times?

i. The `prepare_session_specs` function iterates over all sessions to compute trial frame indices and speed quartiles before the main conversion pass, and then the main conversion pass reloads neural data and recomputes frame indices. The trial quality filter `trial_passes_quality_filters` is called twice for each trial: once in `prepare_session_specs` and once in the main processing loop.

ii.
```python
# First pass in prepare_session_specs:
spec.trial_frame_indices = build_trial_frame_indices(record, spec.estimated_nfr)
# ...
for trial_idx, (keep_trial, frame_idx) in enumerate(zip(spec.kept_trial_mask, spec.trial_frame_indices)):
    if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:

# Second pass in process_sessions:
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
```

iii. The double-pass design separates behavior-only preparation from the I/O-heavy neural loading.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The full spike matrix is loaded for every session even though only ~5-10% of neurons are retained after the selectivity filter. The `select_decoder_neurons` function computes d-prime, corridor responsiveness checks, and percentile thresholds on all neurons before selecting the subset. Additionally, the `prepare_session_specs` pass computes speed quartiles from behavior-estimated frame counts, but these are then applied during the neural-loading pass where the actual neural frame count may differ slightly.

ii.
```python
spk = load_spike_matrix(root, spec.base)  # loads ALL neurons
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)  # keeps ~5-10%
spk = spk[selected_neurons]  # discards 90-95%
```

iii. The full load is necessary because neuron selection requires computing selectivity on all neurons.
