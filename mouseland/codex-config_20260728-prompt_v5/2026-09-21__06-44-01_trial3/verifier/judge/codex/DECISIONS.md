# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads `/app/data/beh/Imaging_Exp_info.npy` first, uses it to enumerate all behavior views in `Beh_<exp_type>.npy`, chooses one canonical behavior view per `(mouse, datexp, blk)` triplet, and then loads one behavior dict, one spike file, and one retinotopy file for each canonical session during conversion.

ii. 
```python
def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def collect_session_views(exp_info):
    for exp_type, records in exp_info.items():
        beh_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh_all = np.load(beh_path, allow_pickle=True).item()
```

```python
beh = load_behavior(view)
planes = load_spike_planes(triplet)
iarea = load_iarea(triplet)
```

iii. In Step 5 notes, the agent says the meaningful session unit is the unique recording, not every duplicated experiment-label view, so it added a canonical-view pass before conversion.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name, the first element of the session triplet. The final `subjects` list is the sorted set of mouse names from canonical sessions, and `subject_idx` maps each session to that list.

ii. 
```python
subjects = sorted({triplet[0] for triplet, _view, _exp_types in sessions})
subject_lookup = {name: idx for idx, name in enumerate(subjects)}
...
data["subject_idx"].append(subject_lookup[mouse])
```

iii. The Step 5 notes explicitly say “subject is the mouse name,” taken from the session key / `mname`.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` recording. When the same recording appears under multiple experiment labels or `stimtype` views, the script picks one “canonical” behavior view by ranking candidates with `behavior_view_priority()` instead of simply keeping the first occurrence.

ii. 
```python
triplet = (rec["mname"], rec["datexp"], rec["blk"])
key = "_".join(triplet)
if has_stimtype:
    key = f"{key}_{rec['stimtype']}"
views[triplet].append(SessionView(...))
```

```python
def choose_canonical_view(views):
    for triplet in sorted(views.keys(), key=lambda x: (x[0], parse_date(x[1]), int(x[2]))):
        ...
        candidates.append((behavior_view_priority(beh, view.has_stimtype), view))
        ...
        out.append((triplet, candidates[0][1], sorted(set(exp_types))))
```

iii. In Step 5 notes and trajectory step 115, the agent says repeated labels are analysis-specific reinterpretations of the same recording and should be collapsed to 89 unique recordings.

## 1-d. How are the data split into trials?

i. Trials are defined framewise from `ft_trInd` and restricted to corridor frames by `ft_CorrSpc`. For each integer `trial_idx`, the script collects the session frames whose rounded trial index equals that trial and keeps the resulting variable-length corridor-frame sequence.

ii. 
```python
frame_trial_raw = np.asarray(beh["ft_trInd"], dtype=float)[:nfr]
frame_trial = np.full(frame_trial_raw.shape, -1, dtype=int)
valid_frame_trial = np.isfinite(frame_trial_raw)
frame_trial[valid_frame_trial] = np.rint(frame_trial_raw[valid_frame_trial]).astype(int)
...
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
```

iii. In Step 5 notes and trajectory step 110, the agent decided a trial should be the corridor-only frame sequence aligned to entry, not the following grey-space segment.

## 1-e. How are trials filtered based on quality controls?

i. The code does not apply the reference 99th-percentile long-trial filter. Instead it skips a trial only if it has fewer than 2 kept corridor frames, has an unrecognized stimulus family, or has non-finite `SoundTime` or `Trial_start_time`.

ii. 
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
if mask.sum() < 2:
    skipped_trials += 1
    continue
...
if family not in family_lookup:
    skipped_trials += 1
    continue
...
if not np.isfinite(cue_time) or not np.isfinite(start_time):
    skipped_trials += 1
    continue
```

iii. The Step 5 notes say the agent wanted to keep corridor-only trials and the trajectory repeatedly notes that no trials were lost on the final full run; it did not adopt the reference long-trial outlier removal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `obj["spks"]` in the per-session spike file, stored plane-by-plane, and the neuron region labels come from `iarea` in the per-session retinotopy file.

ii. 
```python
def load_spike_planes(triplet):
    obj = np.load(path, allow_pickle=True).item()
    return [np.asarray(x) for x in obj["spks"]]

def load_iarea(triplet):
    return np.load(path, allow_pickle=True)["iarea"]
```

iii. The Step 1 notes explicitly record that the reference code directly loads deconvolved `spks` traces and retinotopy labels rather than recomputing fluorescence features.

## 2-b. How is the `neural` data processed?

i. The script does not export all neurons. It first computes stimulus selectivity and corridor responsiveness, keeps at most 64 neurons per grouped region, and then builds each trial’s neural array by concatenating only the selected neurons’ corridor-frame traces. The exported trial matrices are cast to `float16`.

ii. 
```python
selected_blocks, region_idx, region_counts, selection_meta = select_neurons(planes, beh, iarea)
selected_plane_arrays = [
    np.asarray(planes[plane_idx][local_idx], dtype=np.float32)
    for plane_idx, local_idx, _region_id in selected_blocks
]
```

```python
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
).astype(np.float16, copy=False)
```

iii. Step 5 and Step 6 notes justify this as a decoder-specific curation step: exporting all neurons was judged too large, so the agent used a paper-inspired selective-neuron subset to keep the dataset tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered far more aggressively than in the reference. The script keeps only neurons in `V1`, `mHV`, `lHV`, or `aHV`, requires corridor responsiveness, prefers neurons with `|d′| >= 0.3` for the primary stimulus pair, and caps the retained set at 64 neurons per region.

ii. 
```python
region_pool = region_mask & corr_neu & np.isfinite(dp)
strong_pool = region_pool & (np.abs(dp) >= 0.3)
chosen_pool = strong_pool if strong_pool.any() else region_pool
...
candidates.sort(key=lambda x: x[0], reverse=True)
chosen = candidates[:N_PER_REGION]
```

iii. The Step 5 notes say this was intended to stay “close to the paper” while bounding memory and output size, and the metadata records the neuron-selection rule under `metadata["neuron_selection"]`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry / trial start by using the session’s framewise trial index and corridor mask. Each retained trial begins at its first kept corridor frame and keeps all corridor frames for that trial.

ii. 
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
...
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
).astype(np.float16, copy=False)
```

iii. The Step 5 notes and the metadata explicitly state `temporal_alignment_event = "corridor entry (trial start)"` and `corridor_only = True`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay on the native imaging-frame grid. The script estimates the time-bin size from the median `ft` frame interval across sessions and stores that value in milliseconds. No temporal rebinning is applied.

ii. 
```python
def compute_frame_timing(sessions):
    dts.append(float(np.nanmedian(np.diff(ft)) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(dts)), dts
```

```python
"time_bin_size": float(time_bin_ms),
```

iii. The Step 5 notes say the decoder needs native frame timing for time-to-cue and time-since-start signals, so the agent kept framewise trials rather than position-interpolated tensors.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The code derives it from the per-trial MATLAB-datenum field `SoundTime` and the per-frame time vector `ft`.

ii. 
```python
trial_times = ft[mask]
cue_time = float(beh["SoundTime"][trial_idx])
```

iii. In Step 5 notes, the agent mapped `SoundTime` plus `ft` to the decoder input, treating the stored event time as the most direct cue-time representation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, it computes `(cue_time - trial_times) * 86400`, producing a continuous value in seconds that is positive before the cue and negative after it.

ii. 
```python
input_trial = np.vstack(
    [
        (cue_time - trial_times) * SECONDS_PER_DAY,
        ...
    ]
).astype(np.float32)
```

iii. The Step 5 notes justify this as preserving real trial timing on the native frame axis instead of collapsing time to position bins.

## 3-c. How is `input` *Time to sound cue* aligned with the neural data?

i. It is computed from `trial_times = ft[mask]`, where `mask` is exactly the same per-trial frame mask used to slice the neural data.

ii. 
```python
trial_times = ft[mask]
...
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
)
```

iii. The agent’s notes repeatedly state that all exported streams are kept on the same corridor-frame grid as the neural activity.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata, specifically the mouse id, parsed recording date `datexp`, and block `blk` from each session triplet.

ii. 
```python
for triplet, _view, _exp_types in sessions:
    mouse, datexp, blk = triplet
    per_subject[mouse].append((parse_date(datexp), int(blk), triplet))
```

iii. The Step 5 notes explicitly call this a “continuous training-day proxy” because the dataset does not provide a direct trial-level training-day variable.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are sorted within each mouse by calendar date and block, and the value exported for each session is `(date - first_date).days + 0.01 * (blk - 1)`. That scalar is then broadcast across all time bins of every trial in the session.

ii. 
```python
entries.sort()
first_date = entries[0][0]
for dt, blk, triplet in entries:
    out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
```

```python
np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. Step 5 notes justify this as the “most defensible continuous proxy” for training progress when no explicit day counter is present in the imaging files.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The code derives it from the per-trial MATLAB-datenum field `Trial_start_time` and the per-frame time vector `ft`.

ii. 
```python
trial_times = ft[mask]
start_time = float(beh["Trial_start_time"][trial_idx])
```

iii. The Step 5 notes map `Trial_start_time` plus `ft` directly to this decoder input.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, it computes `(trial_times - start_time) * 86400`, yielding a continuous seconds-since-start signal on the imaging-frame grid.

ii. 
```python
input_trial = np.vstack(
    [
        ...,
        (trial_times - start_time) * SECONDS_PER_DAY,
        ...
    ]
).astype(np.float32)
```

iii. The justification in the notes is the same as for cue timing: keep trial timing on the native frame axis rather than switching to the paper’s position-interpolated representation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the exact same `trial_times = ft[mask]` vector used for the neural frame selection, so it has the same length and indexing as the neural trial matrix.

ii. 
```python
trial_times = ft[mask]
...
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
)
```

iii. The alignment follows the agent’s general frame-synchronous design for all trial tensors.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial field `isRew`.

ii. 
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. In Step 5 notes the agent mapped `isRew` directly to `reward_available`, with no extra transformation beyond casting.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script converts the per-trial value to a boolean/float and broadcasts it across all time bins of the trial.

ii. 
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. The notes treat this as a direct trial-level contextual variable; no additional processing is described.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial corridor wall identity `WallName`.

ii. 
```python
wall_name = str(beh["WallName"][trial_idx])
family = family_from_wall_name(wall_name)
```

iii. The Step 5 notes say `WallName` is more stable than experiment-specific `stim_id` remappings when the same recording is reused under multiple behavior views.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script collapses exemplar names such as `circle1`, `leaf1_swap1`, `wood5`, or `rock2` to their coarse family by prefix parsing, converts that family to an integer category, and repeats the category across every time bin of the trial.

ii. 
```python
def family_from_wall_name(name: str) -> str:
    for prefix in ("circle", "leaf", "rock", "wood", "brick"):
        if lowered.startswith(prefix):
            return prefix
```

```python
stim_out = np.full(mask.sum(), family_lookup[family], dtype=np.int16)
```

iii. The Step 5 notes explicitly justify collapsing exemplars into the four broad categories requested by the decoder task.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the session’s lick event frame numbers.

ii. 
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_idx = np.asarray(np.floor(lick_fr), dtype=int)
...
lick_vec[lick_idx[valid]] = 1
```

iii. The notes map `LickFr` directly to a binary time-varying lick output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code floors each lick frame to an integer frame index, drops lick events outside the imaged frame range, and sets a binary per-frame vector to 1 if one or more licks fall in that frame.

ii. 
```python
lick_vec = np.zeros(nfr, dtype=np.int8)
lick_idx = np.asarray(np.floor(lick_fr), dtype=int)
valid = (lick_idx >= 0) & (lick_idx < nfr)
lick_vec[lick_idx[valid]] = 1
```

iii. The Step 5 notes describe this as a direct event-to-binary-series conversion on the imaging-frame grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. After building the full-session binary lick vector, the script slices it with the same per-trial `mask` used for the neural data.

ii. 
```python
lick_vec = build_lick_frame_vector(beh, nfr)
...
lick_out = lick_vec[mask].astype(np.int16)
```

iii. The alignment is justified by the agent’s frame-synchronous export design.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the per-frame position trace `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)[:nfr]
```

iii. The Step 5 notes identify `ft_Pos` as the raw source for the decoder’s position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code divides position by 10 data units (decimeters), floors to an integer, and clips the result into four bins representing the 0–4 m texture corridor.

ii. 
```python
def position_to_bin(pos: np.ndarray) -> np.ndarray:
    bins = np.floor(np.asarray(pos, dtype=float) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. The Step 5 notes justify this as the required 4 equal 1 m bins over the texture corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Category 0 is 0–1 m, 1 is 1–2 m, 2 is 2–3 m, and 3 is 3–4 m, implemented by `floor(ft_Pos / 10)` with clipping at the upper boundary.

ii. 
```python
"output_values": [
    ...,
    ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
    ...
]
```

```python
pos_out = position_to_bin(ft_pos[mask])
```

iii. The notes explicitly say the corridor was treated as 4 m of texture and divided into 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced from the full-session `ft_Pos` array with the same retained-frame `mask` used for the neural trial.

ii. 
```python
pos_out = position_to_bin(ft_pos[mask])
...
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
)
```

iii. The Step 5 notes emphasize that all exported streams stay on the same corridor-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the per-frame running-speed trace `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)[:nfr]
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
```

iii. The Step 5 mapping table names `ft_RunSpeed` as the raw source for running-speed decoding.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first stores raw framewise speed values for every kept trial, then after all sessions are converted it assigns quartile labels by global rank across all retained corridor frames in the dataset.

ii. 
```python
speed_value_trials.append(speed_values)
...
all_speed_values = np.concatenate(
    [trial_speed for session_speed in speed_value_trials_per_session for trial_speed in session_speed]
)
all_speed_bins, speed_edges, speed_value_ranges = assign_rank_speed_bins(all_speed_values)
```

iii. The Step 5 notes and trajectory steps 246–254 justify this as the only way to satisfy the “25% of the data per bin” requirement when many frames share the same zero-speed value.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is not thresholded by fixed speed cutoffs per session. Instead the code sorts all retained speeds, splits the sorted frame indices into four equally sized chunks, and labels those chunks `q1` to `q4`.

ii. 
```python
order = np.argsort(all_speed, kind="mergesort")
speed_bins = np.empty(all_speed.size, dtype=np.int16)
for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
    speed_bins[idx_chunk] = bin_idx
```

iii. The trajectory explicitly says percentile edges failed because many corridor frames are tied at zero speed, so the agent switched to rank-based quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The script preserves the per-trial frame order of the raw speed values, then writes the assigned quartile labels back into each trial using the original retained trial lengths.

ii. 
```python
for session_idx, session_speed in enumerate(speed_value_trials_per_session):
    for trial_idx, trial_speed in enumerate(session_speed):
        n_time = int(trial_speed.size)
        trial_speed_bins = all_speed_bins[cursor : cursor + n_time]
        ...
        data["output"][session_idx][trial_idx] = np.vstack(
            [data["output"][session_idx][trial_idx], trial_speed_bins[np.newaxis, :]]
        ).astype(np.int16)
```

iii. The design keeps speed-bin labels aligned one-for-one with the same retained neural frames.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code trims framewise behavior arrays to the imaged frame count, drops out-of-range licks, treats non-finite `ft_trInd` entries as invalid frames, and skips any trial whose corridor-frame mask is too short or whose cue/start time is non-finite.

ii. 
```python
if key.startswith("ft_") or key in {"ft", "AftCueFr", "BefCueFr"}:
    trimmed[key] = value[:nfr]
```

```python
valid = (lick_idx >= 0) & (lick_idx < nfr)
...
valid_frame_trial = np.isfinite(frame_trial_raw)
...
if mask.sum() < 2:
    skipped_trials += 1
    continue
if not np.isfinite(cue_time) or not np.isfinite(start_time):
    skipped_trials += 1
    continue
```

iii. The trajectory mentions a sample-run warning from `ft_trInd` NaNs; the agent then patched the code to make invalid frame-trial indices explicit instead of casting NaNs blindly.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading the very large spike files, computing neuron-selection statistics over all planes, and then repeatedly slicing/concatenating selected traces trial-by-trial.

ii. 
```python
planes = load_spike_planes(triplet)
...
selected_blocks, region_idx, region_counts, selection_meta = select_neurons(planes, beh, iarea)
...
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
)
```

iii. Step 6 notes say raw dense spike files are very large and that naive export of all neurons would be impractical; the trajectory also shows an attempted inner-loop optimization that was later reverted.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the full-frame scan performed once per trial to build `mask`, the per-trial `np.concatenate` over selected plane blocks, and the repeated behavior-file passes used for canonical-view selection and frame-timing estimation.

ii. 
```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
```

```python
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
)
```

iii. The Step 6 notes acknowledge an attempted optimization around pre-concatenating selected traces per session, but the final version keeps the more repetitive inner-loop construction.

## 12-c. What processing does the code repeat multiple times?

i. The script rereads behavior dictionaries in several separate passes: once to collect views, again to rank canonical views, again to estimate frame timing, optionally again to score sample sessions, and again during final conversion. It also recomputes trial masks one trial at a time by rescanning the whole frame index.

ii. 
```python
beh_all = np.load(beh_path, allow_pickle=True).item()
...
beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]
```

```python
for triplet, view, _exp_types in sessions:
    beh = load_behavior(view)
```

```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
```

iii. Step 6 notes explicitly warn about duplicated behavior views and heavy data scale, but the final code still performs multiple behavior-loading passes before and during conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It does extra work for diagnostics and metadata that are not needed by the downstream decoder itself: optional plotting, whole-dataset frame-interval estimation, canonical-view scoring over duplicate behavior entries, and speed-bin summaries / neuron-selection summaries that are stored only as metadata.

ii. 
```python
def make_processing_plot(...):
    ...
    fig.savefig(f"processing_{session_id}.png", dpi=150)
```

```python
time_bin_ms, frame_dt_ms_all = compute_frame_timing(canonical_sessions)
...
data["metadata"]["speed_bin_edges"] = speed_edges.astype(float).tolist()
data["metadata"]["speed_bin_value_ranges"] = speed_value_ranges
```

iii. The agent justified these additions in Step 6 and later trajectory messages as sanity checks and user-facing diagnostics, but they are not consumed by the decoder training pipeline.
