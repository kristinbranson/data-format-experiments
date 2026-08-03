# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every behavior file matching `data/beh/Beh_*.npy`, uses `Imaging_Exp_info.npy` to collect experiment labels, collapses those records to one representative behavior record per recording base (`<mouse>_<date>_<blk>`), and then loads spikes and retinotopy session by session from `data/spk` and `data/retinotopy`.

ii.
```python
def load_experiment_type_map(root: Path) -> dict[str, set[str]]:
    exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
    ...

def select_representative_sessions(root: Path) -> list[SessionSpec]:
    beh_dir = root / "data" / "beh"
    ...
    for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for key, record in beh_dict.items():
            base = "_".join(key.split("_")[:5])
            ...

def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent explicitly chose “89 unique recording bases as sessions” to match the paper’s “89 recordings in 19 mice” and to avoid duplicated experiment labels and `swap1`/`swap2` aliases. Trajectory step 160 repeats that rationale.

## 1-b. How are the data split into subjects?

i. Subjects are split by parsing the first underscore-delimited token of each recording base, then taking the sorted unique set of those subject IDs. Session-to-subject assignment is stored in `subject_idx`.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    subject = parts[0]
    ...
    return subject, date, blk

subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
...
subject_idx.append(subject_to_idx[spec.subject])
```

iii. The notes repeatedly anchor the dataset to “19 imaging mice” and use mouse name parsed from the recording base as the subject definition.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique recording bases `<mouse>_<date>_<blk>`, not as every behavior key or every experiment-session record. If multiple behavior keys map to the same base, the agent picks one representative key with a heuristic that prefers non-`swap` keys and more complete stimulus information. Sessions are then sorted by subject, date, and block.

ii.
```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    non_nan_stim = int(np.isfinite(stim_id).sum()) if stim_id.size else 0
    unique_walls = int(len(np.unique(record.get("WallName", []))))
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)

...
if session_key_score(key, record) < session_key_score(current.key, current.record):
    current.key = key
    current.record = record

specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
```

iii. Step 5 decisions 1 and 2 say to use 89 unique recording bases and to prefer a plain behavior key when duplicates exist because it “already contains the full `WallName` set.”

## 1-d. How are the data split into trials?

i. Trials are defined from frame-level trial IDs in `ft_trInd`. The code first truncates behavior arrays to a session frame count, keeps finite `ft_trInd` entries, applies the running-corridor mask, and then collects the frame indices belonging to each integer trial ID from `0` to `ntrials - 1`.

ii.
```python
def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0

    ntrials = int(record["ntrials"])
    frame_indices = []
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
        frame_indices.append(trial_frames.astype(np.int32, copy=False))
    return frame_indices
```

iii. Step 5 decision 4 says to align trials using frame-level `ft_trInd` instead of `StartFr`/`EndFr` because that matches the reference code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if the retained running-corridor segment has fewer than 5 timepoints, if the retained duration from trial start exceeds 60 s, or if the largest gap between retained frames exceeds 10 s. Entire sessions are dropped if fewer than 2 valid trials remain.

ii.
```python
MIN_TRIAL_TIMEPOINTS = 5
MAX_RETAINED_TRIAL_DURATION_S = 60.0
MAX_RETAINED_INTERFRAME_GAP_S = 10.0

def trial_passes_quality_filters(...):
    if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
        return False, "too_few_timepoints"
    ...
    if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
        return False, "retained_duration_gt_60s"
    ...
    if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
        return False, "interframe_gap_gt_10s"

...
if len(session_neural) < 2:
    removed_sessions.append((spec.base, len(session_neural), "fewer_than_two_valid_trials"))
```

iii. The justification appears in Step 10/README: the agent says these extra filters were added as “decoder-specific curation” to remove “pathological trial-start timing outliers” from stalled trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from the raw `spks` arrays stored in `data/spk/<base>_neural_data.npy`, concatenated across planes along the neuron axis. Retinotopy is loaded separately for brain-region labels and neuron selection, not for the neural values themselves.

ii.
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```

iii. Step 1 and Step 4 notes say the stored `spks` arrays are already the deconvolved fluorescence traces used in the paper and should be loaded directly without recomputing dF/F.

## 2-b. How is the `neural` data processed?

i. The agent keeps the stored deconvolved traces as `float32`, does not dF/F-normalize, does not z-score, and does not interpolate them onto position bins for export. Instead, it slices the raw spike matrix to selected neurons and to the retained frame indices for each trial.

ii.
```python
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
spk = spk[selected_neurons]
...
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
...
"frame_bin_source": "native imaging frame timestamps; no temporal resampling",
```

iii. Step 5 decision 3 says to “keep the stored `spks` arrays directly,” and decision 9 says to keep native frame timestamps rather than interpolate to a synthetic grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code applies a functional neuron-selection step, not just a quality-control step: it chooses a reference stimulus pair, computes `d'` on running corridor frames, keeps neurons that respond more in corridor than gray, and then retains the top 5% positive and top 5% negative `d'` neurons within each of `V1`, `mHV`, `lHV`, and `aHV`. If that fails, it falls back to all mapped visual neurons, and finally to all `region_idx < 4` neurons.

ii.
```python
def select_decoder_neurons(spk, record, region_idx):
    stim_pair = choose_reference_stimuli(record)
    ...
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
    selected = np.zeros(spk.shape[0], dtype=bool)
    for area in range(4):
        candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
        ...
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
```

iii. Step 5 decision 10 and the README explicitly justify this as a “paper-style top 5% positive and top 5% negative stimulus-selective neuron” export to make full decoding tractable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start / corridor entry. For each trial, the code keeps only the frames whose `ft_trInd` equals that trial ID and stores those columns as the trial matrix. Metadata labels the temporal alignment event as “corridor entry / trial start” with `off_start = 0.0`.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
...
metadata = {
    ...
    "temporal_alignment_event": "corridor entry / trial start",
    "off_start": 0.0,
    "off_end": None,
}
```

iii. The task instructions required trial-start alignment, and trajectory step 160 says the agent would “align on frame timestamps to trial start.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported data use the native imaging frame bins. The script records a single metadata `time_bin_size` as the median of per-session median frame intervals in milliseconds, but it does not resample or rebin the actual trial matrices.

ii.
```python
dt_ms = np.diff(ft) * MS_PER_DAY
spec.median_frame_dt_ms = float(np.median(dt_ms))
...
"time_bin_size": float(np.median(frame_dt_medians)),
"frame_bin_source": "native imaging frame timestamps; no temporal resampling",
```

iii. Step 5 decision 9 says the agent intentionally used raw frame timestamps because the median imaging interval was stable enough and synthetic resampling would be less faithful.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level cue times in `SoundTime` and frame timestamps in `ft`.

ii.
```python
def session_trial_info(record: dict, trial: int, frame_idx: np.ndarray):
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. Step 5 variable mapping says `input[0]` comes from `beh['SoundTime']` and retained frame times from `beh['ft']`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the code subtracts frame time from the trial’s `SoundTime` and converts days to seconds. The resulting value is positive before the cue and negative after the cue.

ii.
```python
time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The notes explicitly describe this as `SoundTime - frame_time` in seconds and call out the intended sign convention.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is sampled on exactly the same `frame_idx` used for each neural trial, so it has one value per retained neural frame.

ii.
```python
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
input_trial = np.vstack(
    [time_to_cue, training_day, time_since_start, reward_available]
).astype(np.float32, copy=False)
```

iii. Step 5 planned sanity checks explicitly say the agent intended to verify that `time_to_sound_cue_s` equals raw `SoundTime - ft[mask]` on sampled trials.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not read from a single raw variable. The agent derives it from the session date encoded in the recording base string and the first retained imaging-session date for that subject.

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    ...
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    ...

first_date_by_subject: dict[str, datetime] = {}
for spec in specs:
    first_date_by_subject.setdefault(spec.subject, spec.date)
    spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. Step 5 decision 12 says this variable is not defined by the paper, so the agent used elapsed days since the subject’s first retained imaging session as a reproducible proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code sorts sessions by subject/date/block, computes the integer day difference from the subject’s first session, stores it as `spec.training_day`, and then repeats that scalar across every timepoint in the trial’s input matrix.

ii.
```python
spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
...
training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. The notes justify this as a decoder-required addition rather than a paper-defined measurement.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame timestamps in `ft` and trial-level start times in `Trial_start_time`.

ii.
```python
frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. Step 5 maps `input[2]` to retained frame times from `beh['ft']` and `beh['Trial_start_time']`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the code subtracts the trial’s `Trial_start_time` from that frame time and converts the result from days to seconds.

ii.
```python
time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. The notes describe this field as a straightforward frame-time analysis consistent with the paper’s use of raw frame timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices used to slice each neural trial, so it is frame-aligned one-to-one with the neural columns.

ii.
```python
time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. Step 5 planned sanity checks say the agent intended to verify that `time_since_trial_start_s` equals raw `ft[mask] - Trial_start_time`.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew`.

ii.
```python
reward_available = np.full(
    frame_idx.size,
    float(bool(spec.record["isRew"][trial_idx])),
    dtype=np.float32,
)
```

iii. Step 5 maps `input[3]` directly to `beh['isRew']` and notes that unsupervised / naive / grating sessions should therefore be all zeros.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code converts the per-trial `isRew` flag to `0.0` or `1.0` and repeats that value across every retained frame in the trial.

ii.
```python
reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
input_trial = np.vstack(
    [time_to_cue, training_day, time_since_start, reward_available]
)
```

iii. The notes justify this as a trial-level binary variable that must be broadcast to match the uniform `(4, T)` input format.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from trial-level wall / corridor labels in `WallName`, with the category vocabulary assembled from all sessions’ `UniqWalls`.

ii.
```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories
...
stim_idx = np.full(
    frame_idx.size,
    visual_to_idx[str(spec.record["WallName"][trial_idx])],
    dtype=np.int16,
)
```

iii. Step 5 variable mapping says `output[0]` comes from `beh['WallName']`, with a global category mapping over all unique stimulus names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code builds a global sorted category list, maps each trial’s `WallName` to an integer index, and repeats that same integer across every retained frame in the trial.

ii.
```python
visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
...
stim_idx = np.full(frame_idx.size, visual_to_idx[str(spec.record["WallName"][trial_idx])], dtype=np.int16)
output_trial = np.vstack([stim_idx, licking, position_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. The notes justify repeating the per-trial label across timepoints so all outputs share a uniform `(4, T)` structure.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and `LickTrind`, not from lick times or lick positions.

ii.
```python
def build_lick_frame_lookup(record: dict, nfr: int) -> dict[int, np.ndarray]:
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    ...
```

iii. The notes say this was chosen to create a frame-aligned binary output, while the reference code’s lick analyses showed the relevant raw variables and timing conventions.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code cleans `LickFr` / `LickTrind` for finite in-range entries, groups unique lick frames by trial, and then marks each retained frame as `1` if its frame index appears in that trial’s lick-frame set, otherwise `0`.

ii.
```python
for trial in np.unique(lick_trial):
    lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
...
lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Step 5 planned sanity checks say the agent intended to verify that the converted licking vector matches raw `LickFr` events projected onto retained imaging frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by evaluating lick-frame membership on the exact same `frame_idx` array used to slice the neural data.

ii.
```python
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. The notes and code both treat licking as a binary time series on retained neural frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level corridor positions in `ft_Pos`.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Step 5 maps `output[2]` to `beh['ft_Pos']` on retained frames.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code takes the retained-frame `ft_Pos` values, clips them to `[0, 39.999]` decimeters, and then assigns bins based on decimeter position within the 0-4 m textured corridor.

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. The notes justify this with the paper’s 4 m textured corridor and the reference code’s decimeter units (`Texture_Length = 40`, `Corridor_Length = 60`).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are fixed at 10, 20, and 30 decimeters, yielding four categories: `0-1m`, `1-2m`, `2-3m`, and `3-4m`.

ii.
```python
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
...
return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Step 5 explicitly says to use edges `[0,10,20,30,40]` because the decoder task requested four equal-length 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled from `ft_Pos` at the same retained frame indices used for the neural trial, so it is one categorical label per neural frame.

ii.
```python
position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. Step 5 planned sanity checks say the agent intended to check that position-bin transitions occur at the raw `ft_Pos` thresholds on sampled trials.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level running speeds in `ft_RunSpeed`.

ii.
```python
run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
...
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Step 5 maps `output[3]` directly to raw `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first aggregates all retained running-corridor speed samples across all sessions and valid trials, computes global quartile edges, and then bins each trial’s retained `ft_RunSpeed` samples using those dataset-wide thresholds.

ii.
```python
all_speeds.append(run_speed[frame_idx])
...
speed_values = np.concatenate(all_speeds).astype(np.float32, copy=False)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```

iii. Step 5 decision 13 says global quartiles were used because the decoder task specified bins corresponding to 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the three global quartile edges from `np.quantile(..., [0.25, 0.5, 0.75])`. Each sample is assigned with `np.searchsorted(..., side="right")` and clipped to categories `0` through `3`.

ii.
```python
def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. The notes say the speed-bin check was to recompute quartiles from retained `ft_RunSpeed` and verify that the bins match this exact operation.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read on the same retained frame indices used for each neural trial, so each neural frame gets one speed-bin label.

ii.
```python
speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The agent’s whole export strategy was to keep all decoder variables on the retained native frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles mismatches and missing values mostly by truncation and dropping: it estimates a behavior frame count using the minimum of several frame-array lengths, clips frame indices to `nfr`, discards non-finite `ft_trInd`, removes non-finite or out-of-range lick events, falls back to broader neuron sets if selectivity masks fail, and drops pathological trials instead of repairing them.

ii.
```python
def estimate_behavior_frame_count(record: dict) -> int:
    frame_keys = ["ft", "ft_trInd", "ft_move", "ft_CorrSpc", "ft_Pos", "ft_RunSpeed"]
    return min(int(np.asarray(record[key]).shape[0]) for key in frame_keys)

valid = np.isfinite(ft_tr)
...
valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
...
frame_idx = frame_idx[frame_idx < nfr]
...
if not np.any(selected):
    selected = (region_idx < 4) & corr_neu & np.isfinite(dp)
if not np.any(selected):
    selected = region_idx < 4
```

iii. Step 4 notes justify truncating behavior arrays to neural frame count because the raw behavior streams are usually 1-3 frames longer than the spike matrices; Step 10 justifies removing pathological stalled trials rather than keeping extreme timing outliers.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading very large spike matrices from disk, computing the per-session stimulus-selective neuron subset on those full matrices, and then iterating trial-by-trial to slice neural matrices and build the input/output arrays.

ii.
```python
spk = load_spike_matrix(root, spec.base)
...
selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
...
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...
    neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. The Step 6 notes explicitly call out full neural-file loading as the main memory / runtime burden, and the structure of `process_sessions` shows that the large spike load plus neuron selection sit on the hot path for every session.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code has several obvious Python loops that could be vectorized or precomputed better: trial grouping in `build_trial_frame_indices`, trial-level lick lookup construction, the per-trial conversion loop in `process_sessions`, and repeated percentile selection per area in `select_decoder_neurons`.

ii.
```python
for trial in range(ntrials):
    trial_frames = valid_idx[keep & (trial_ids == trial)]

for trial in np.unique(lick_trial):
    lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)

for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    ...

for area in range(4):
    candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
```

iii. The agent did note some speedups in Step 6, but the final code still leaves these loops in Python.

## 12-c. What processing does the code repeat multiple times?

i. The biggest repeated work is that `trial_passes_quality_filters` is called during `prepare_session_specs` and then again during `process_sessions`. Inside the per-trial loop, the code also repeatedly converts the same record arrays (`ft_Pos`, `ft_RunSpeed`) to numpy arrays instead of caching them once per session.

ii.
```python
for trial_idx, (keep_trial, frame_idx) in enumerate(...):
    if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
        all_speeds.append(run_speed[frame_idx])

...
for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
    keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
    ...
    position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
    speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. This repetition is visible directly in the final script; the notes only partly acknowledge it by saying session preparation is a first pass and conversion is a second pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only for bookkeeping or diagnostics rather than the exported decoder arrays: `kept_trial_mask` is built as an intermediate for the speed-edge pass and then effectively superseded by a second filter pass, `source_files` / `all_keys` / `experiment_types` are tracked only for metadata, and optional processing plots plus long `removed_trials` logs do not affect downstream decoding.

ii.
```python
spec.kept_trial_mask = np.array(
    [len(frame_idx) >= MIN_TRIAL_TIMEPOINTS for frame_idx in spec.trial_frame_indices],
    dtype=bool,
)
...
selected[base].source_files.add(beh_path.name)
selected[base].all_keys.add(key)
selected[base].experiment_types.add(exp_type)
...
processing_plots_remaining = 2 if show_processing else 0
...
"removed_trials": removed_trials[:1000],
```

iii. The notes and README treat these as sanity-check / documentation aids rather than part of the decoder signal itself.
