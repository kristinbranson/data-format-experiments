# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed all `sub-*/sub-*_behavior+ophys.nwb` files under `/app/data`, extracted subject/session/scene metadata with `h5py`, sorted sessions by subject and session number, and then re-opened each NWB file in `process_session` to load the behavior and ophys arrays directly from HDF5 groups. It did not use `pynwb`.

ii.
```python
def list_nwb_files(data_root: Path) -> list[SessionMeta]:
    session_meta: list[SessionMeta] = []
    for file_path in sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb")):
        with h5py.File(file_path, "r") as f:
            subject = decode_scalar(f["general/subject/subject_id"][()])
            identifier = decode_scalar(f["identifier"][()])
        ...
        session_meta.append(SessionMeta(...))

def process_session(session_meta: SessionMeta, block_size: int = 256) -> tuple[dict, dict]:
    with h5py.File(session_meta.file_path, "r") as f:
        behavior = f["processing/behavior/BehavioralTimeSeries"]
        ophys = f["processing/ophys"]
```

iii. `CONVERSION_NOTES.md` says the source material was the NWB sessions in `data/sub-*/sub-*_behavior+ophys.nwb`, and the trajectory says the agent chose to “read the NWBs directly with `h5py`” after confirming the structure was usable.

## 1-b. How are the data split into subjects?

i. Subjects are defined from the NWB metadata field `general/subject/subject_id`, collected across all files and deduplicated in sorted order.

ii.
```python
with h5py.File(file_path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"][()])
...
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes state the final dataset contains `11` subjects, and the trajectory shows the AI used the NWB metadata rather than only directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session number is parsed from the filename, and sessions are sorted by `(subject, session_num)`.

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
...
session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
```

iii. `CONVERSION_NOTES.md` explicitly says the supplied `data/` directory contains `152` NWB sessions and that the final dataset contains those `152` sessions across `11` subjects.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start > 0` for trial starts and `teleport > 0` for trial ends. The per-trial sample range is `[trial_start_idx : trial_stop_idx)`, so the teleport sample itself is excluded.

ii.
```python
trial_start_series = behavior["trial_start"]["data"][:]
teleport_series = behavior["teleport"]["data"][:]
...
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
...
kept_events[:, start:stop]
```

iii. `CONVERSION_NOTES.md` says “Trials are aligned to `trial start`” and “the kept sample range is `[trial_start_idx : teleport_idx)`; the teleport sample itself is excluded.”

## 1-e. How are trials filtered based on quality controls?

i. The AI does not use a minimum-trial-length filter. Instead, it drops an entire trial if more than `30%` of frames in that trial have `lick count > 2`, which it treats as the paper’s lick-sensor artifact criterion.

ii.
```python
def bad_lick_trial_mask(lick_counts: np.ndarray, start_idx: np.ndarray, stop_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask

lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. The notes say “A full trial was dropped if more than `30%` of its frame samples had `lick count > 2`” and that this “matches the paper’s lick-sensor error criterion.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The saved neural signal is derived from raw Suite2p fluorescence and neuropil traces in the NWB file, not from the NWB `Deconvolved` array.

ii.
```python
for plane_key in plane_keys:
    fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
    neuropil_ds = ophys["Neuropil"][plane_key]["data"]
    ...
    fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
    neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
    dff, events = compute_dff_and_events(...)
```

iii. `CONVERSION_NOTES.md` says the saved neural activity is “recomputed from Suite2p fluorescence and neuropil, rather than taking the NWB `Deconvolved` array directly.” The trajectory also says the agent chose to reproduce the repo’s dF/F and OASIS steps.

## 2-b. How is the `neural` data processed?

i. For curated ROIs only, the AI recomputes dF/F and event activity: neuropil subtraction with coefficient `0.7`, per-trial maximin baseline (`sigma=15`, min filter `300`, max filter `300`), dF/F normalization, Gaussian smoothing with `sigma=2`, then OASIS deconvolution with `tau=0.7`.

ii.
```python
f -= neu_coef * f_neu
...
baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
...
dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
...
dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
```

iii. The notes list these steps explicitly and say the repo logic was treated as the operational reference. The trajectory states the AI “pinned down the original dF/F procedure from the repo.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only Suite2p-curated ROIs with `iscell[:, 0] == 1`. It then removes putative interneurons whose dF/F has Pearson correlation `> 0.5` with running speed over valid in-trial samples.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
...
block_corr = np.array(
    [
        np.corrcoef(trace, speed_valid)[0, 1]
        if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
        else np.nan
        for trace in dff_valid
    ],
    dtype=np.float32,
)
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
keep_block = ~block_interneuron
```

iii. `CONVERSION_NOTES.md` says the conversion keeps curated `iscell` ROIs and excludes speed-correlated putative interneurons, matching the paper’s description.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing the event matrix with the same `[start:stop)` trial windows used for the behavior streams. No additional offset is applied.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
for out_idx, trial_idx in enumerate(kept_trial_indices):
    start = trial_start_idx[trial_idx]
    stop = trial_stop_idx[trial_idx]
    session_trial_blocks[out_idx].append(
        kept_events[:, start:stop].astype(np.float16, copy=False)
    )
```

iii. The notes say trials are aligned to `trial start`, and the stored metadata sets `temporal_alignment_event` to `"trial start"` and `off_start` to `0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame resolution and does not rebin trials into coarser bins. It infers frame rate from behavior timestamps and stores `metadata["time_bin_size"] = 1000 / mean_frame_rate`, which is about `64.48 ms`.

ii.
```python
def infer_frame_rate(position_timestamps: np.ndarray) -> float:
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))
...
frame_rate_hz = infer_frame_rate(timestamps)
...
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. `CONVERSION_NOTES.md` says frame rate was inferred from behavior timestamps, the mean frame rate was `15.507813` Hz, and “No rebinning” was applied beyond the smoothing inside the neural preprocessing pipeline.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior `position` timestamps array.

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes say `input[0]` is computed “from the behavior timestamps relative to the first frame of each trial.”

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp in that trial from every timestamp in the same trial.

ii.
```python
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
trial_input = np.vstack(
    [
        trial_times,
        ...
    ]
)
```

iii. The notes describe `time_from_trial_start_s` exactly this way.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same `[start:stop)` trial indices as the neural event matrix, after first truncating all streams in a session to a common minimum aligned length.

ii.
```python
session_len = min(..., len(timestamps), *plane_lengths)
...
timestamps = timestamps[:session_len]
...
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)
```

iii. The notes say behavior and fluorescence streams were truncated to the common minimum length when needed, then trial-wise arrays were built on the shared sample index.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The primary label comes from the session scene string stored in the NWB identifier metadata, not from the behavior `environment` time series. The behavior `environment` stream is only used as a consistency check.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
...
parts = identifier.strip("/").split("/")
scene = parts[-1]
...
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
env_in_trial = env_timeseries[start:stop]
env_in_trial = env_in_trial[env_in_trial >= 0]
```

iii. The notes say “Trial environment ... labels are derived from the session scene name” and were “cross-checked against the aligned behavior `environment` stream on every kept trial.”

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses the scene name with regexes, converts `Env1`/`Env2` to `0`/`1`, assumes switch sessions change at trial `30`, and then repeats the resulting per-trial environment label across all timepoints in the trial.

ii.
```python
def parse_scene(scene: str, n_trials: int, switch_trial_count: int = 30) -> list[TrialLabel]:
    ...
    return [
        TrialLabel(env=env0 if trial_idx < switch_trial_count else env1, ...)
        for trial_idx in range(n_trials)
    ]
...
np.full(trial_times.shape, float(label.env), dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the parser handles fixed-location and switch scenes and that “for switch sessions, the switch point is trial `30`, matching the paper/task structure.”

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is taken from the NWB behavior `trial number` stream, sampled at the trial start index, then repeated across the trial.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The notes say “`trial_number` is copied from the aligned NWB `trial number` stream at trial start.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied other than taking the scalar trial-start value and broadcasting it across all timepoints in the trial.

ii.
```python
trial_input = np.vstack(
    [
        trial_times,
        np.full(trial_times.shape, float(label.env), dtype=np.float32),
        np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
        ...
    ]
)
```

iii. The notes only describe copying the aligned `trial number` value; there is no extra normalization or reindexing step.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` timestamps together with the trial start and trial stop times.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcome_per_trial(
    reward_timestamps,
    timestamps[trial_start_idx],
    timestamps[trial_stop_idx],
)
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. The notes say reward outcome is taken from reward timestamps that fall between trial start and teleport, and previous-trial outcome is derived from that per-trial reward outcome sequence.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary `reward_outcomes` array per trial, then shifts it by one trial to create `prev_reward_outcomes`, sets the first trial to `0`, and repeats the scalar value across each kept trial’s timepoints. This is done before lick-error trial removal, so a kept trial still references the actual previous trial.

ii.
```python
reward_outcomes = reward_outcome_per_trial(...)
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
...
np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says previous-trial outcome was computed before removing bad lick trials so the previous outcome still refers to the actual preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from the behavior `position` stream plus reward-zone bounds inferred from the session scene label (`A/B/C`), not from the behavior `reward_zone` stream.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
...
label = trial_labels[trial_idx]
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
...
discretize_distance_to_zone(trial_pos, zone_bounds)
```

iii. The notes say “Trial environment and reward-zone labels are derived from the session scene name” and that distance is computed relative to the active reward-zone bounds for that label.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest edge of the active reward zone: negative before the zone, `0` inside the zone, positive after the zone.

ii.
```python
signed_distance = np.where(
    position_cm < zone_start,
    position_cm - zone_start,
    np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0),
)
```

iii. `CONVERSION_NOTES.md` describes the same signed-distance rule in prose.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses explicit threshold logic to map the signed distance into seven integer classes. Its bins are implemented with inclusive upper bounds at `-10`, `10`, and `50`, and exact zero gets its own class.

ii.
```python
bins[signed_distance < -50.0] = 0
bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
bins[signed_distance == 0.0] = 3
bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
bins[signed_distance > 50.0] = 6
```

iii. The notes say the output was “binned into the requested `7` categories,” but the precise boundary handling comes from this hard-coded logic in `convert_data.py`, not from a more declarative bin-edge table.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[start:stop)` slices as the neural arrays, so alignment is by shared sample index after trial segmentation.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
trial_output = np.vstack(
    [
        discretize_distance_to_zone(trial_pos, zone_bounds),
        ...
    ]
)
```

iii. The notes emphasize that all trial-wise outputs are aligned to the same trial-start-based sample windows as the neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavior `position` stream.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The notes describe absolute position as the animal’s corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Before binning, the AI clips position to the corridor range `0` to `450` cm.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
discretize_absolute_position(trial_pos)
```

iii. `CONVERSION_NOTES.md` says absolute position was built from a `450` cm corridor and then binned into five equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI discretizes clipped position into five equal bins of `90` cm each by taking `floor(position / 90)`, with the top bin capped at `4`.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. The notes say “Corridor clipped to `0` to `450` cm” and “binned into `5` equal bins of `90` cm each.”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial `[start:stop)` slices as the neural data.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The notes describe all time-varying outputs as trial-wise arrays built on the same sample windows as neural activity.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` stream.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
...
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The notes call this output “binary `lick_count > 0`.”

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick count is binarized so any positive value becomes `1` and zero stays `0`.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The notes explicitly document `lick` as “Binary `lick_count > 0`.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced with the same `[start:stop)` trial indices as the neural data.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
...
trial_output = np.vstack(
    [
        ...,
        trial_lick,
        ...
    ]
)
```

iii. The notes say the trial windows are shared across neural and behavioral variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The reward-zone location label is derived from the scene name embedded in the NWB identifier metadata, via `parse_scene`, then converted from `A/B/C` to `0/1/2`.

ii.
```python
scene = parts[-1]
...
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
zone_idx = ZONE_TO_INDEX[label.zone]
```

iii. The notes say reward-zone labels are derived from the session scene name, with support for fixed-location and switch sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene string, applies a fixed switch point of trial `30` in switch sessions, maps zone letters to integer categories, and repeats the category across all timepoints in the trial.

ii.
```python
def parse_scene(scene: str, n_trials: int, switch_trial_count: int = 30) -> list[TrialLabel]:
    ...
zone_idx = ZONE_TO_INDEX[label.zone]
...
np.full(trial_times.shape, zone_idx, dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` says switch sessions use trial `30` as the switch point and that reward-zone location is a per-trial label repeated across the trial.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` timestamps plus the trial start and stop times.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcome_per_trial(
    reward_timestamps,
    timestamps[trial_start_idx],
    timestamps[trial_stop_idx],
)
```

iii. The notes say reward outcome was taken from “reward timestamps falling between trial start and teleport.”

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI scans the sorted reward timestamps against each trial’s `[start_time, stop_time)` interval, sets a per-trial binary outcome, and repeats that scalar over all timepoints in the trial.

ii.
```python
def reward_outcome_per_trial(reward_timestamps: np.ndarray, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    reward_idx = 0
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        while reward_idx < len(reward_timestamps) and reward_timestamps[reward_idx] < start_time:
            reward_idx += 1
        probe_idx = reward_idx
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes
...
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. The notes describe the same interval-based rule and emphasize that reward outcome is a per-trial label repeated across time.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles stream-length mismatches by truncating every session to the common minimum length across behavior arrays, timestamps, and fluorescence planes. It ignores negative environment values during the environment cross-check, drops lick-artifact trials, keeps NaNs outside trial windows during dF/F computation, and raises hard errors on environment mismatches or sessions where no neurons survive filtering. It does not use the noisy `reward_zone` behavior stream at all.

ii.
```python
session_len = min(
    len(position),
    len(speed),
    len(lick_counts),
    len(env_timeseries),
    len(trial_number_series),
    len(trial_start_series),
    len(teleport_series),
    len(timestamps),
    *plane_lengths,
)
...
env_in_trial = env_timeseries[start:stop]
env_in_trial = env_in_trial[env_in_trial >= 0]
...
if env_mode != label.env:
    raise RuntimeError(...)
...
if not block_list:
    raise RuntimeError(...)
```

iii. `CONVERSION_NOTES.md` says a few multi-plane sessions had one-frame mismatches and that “all streams in a session were truncated to the common minimum length” before further processing.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading large fluorescence and neuropil arrays from every NWB file, recomputing dF/F and OASIS events blockwise for every curated ROI, computing per-cell speed correlations for interneuron exclusion, and then serializing the full pickle.

ii.
```python
for plane_key in plane_keys:
    fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
    neuropil_ds = ophys["Neuropil"][plane_key]["data"]
    ...
    dff, events = compute_dff_and_events(...)
    ...
    block_corr = np.array([... for trace in dff_valid], dtype=np.float32)
...
save_pickle(args.full_output, full_dataset)
```

iii. The trajectory says the AI intentionally used a blockwise implementation to keep memory bounded while recomputing the paper-style deconvolved signal on the full dataset.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-trial loops in `bad_lick_trial_mask` and `reward_outcome_per_trial`, the Python list comprehension used to compute one correlation per cell, and the nested `(plane -> block -> kept trial)` loop that appends trial slices for each block.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
    if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
        mask[trial_idx] = True
...
for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
    ...
block_corr = np.array(
    [np.corrcoef(trace, speed_valid)[0, 1] ... for trace in dff_valid],
    dtype=np.float32,
)
...
for out_idx, trial_idx in enumerate(kept_trial_indices):
    session_trial_blocks[out_idx].append(kept_events[:, start:stop].astype(np.float16, copy=False))
```

iii. The trajectory repeatedly mentions the implementation is blockwise and trial-wise for memory safety, which explains why some loops were left in Python despite the efficiency cost.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code repeatedly slices every kept trial out of every kept cell block for every imaging plane. The dF/F pipeline also performs multiple passes over the same block (`nansmooth_2d`, min filter, max filter, smoothing again, OASIS). Across outputs, the same trial windows are reused repeatedly for different variables.

ii.
```python
for plane_key in plane_keys:
    ...
    for block_start in range(0, len(curated_plane_indices), block_size):
        ...
        dff, events = compute_dff_and_events(...)
        ...
        for out_idx, trial_idx in enumerate(kept_trial_indices):
            ...
            session_trial_blocks[out_idx].append(kept_events[:, start:stop].astype(np.float16, copy=False))
```

iii. The trajectory says the AI accepted this repetition because it wanted a memory-safe blockwise converter and later optimized sample creation so it could reuse the full pickle instead of rerunning the full conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes full dF/F traces even though the saved `neural` output keeps only deconvolved events; dF/F is retained only long enough to compute interneuron correlations. It also parses `date`, reads `planeIdx`, checks environment consistency, and stores extensive per-session stats that are not consumed by the downstream decoder.

ii.
```python
plane_idx = imgseg["planeIdx"][:].astype(np.int16)
...
dff, events = compute_dff_and_events(...)
...
block_corr = np.array([... for trace in dff_valid], dtype=np.float32)
...
stats = {
    "subject": session_meta.subject,
    "session_num": session_meta.session_num,
    ...
    "plane_keys": plane_keys,
    "plane_curated_counts": curated_counts_by_plane,
    "plane_kept_counts": kept_counts_by_plane,
}
```

iii. The notes emphasize validation and sanity checking, so the extra statistics and cross-checks were intentional, but they are not required by `train_decoder.py` itself.
