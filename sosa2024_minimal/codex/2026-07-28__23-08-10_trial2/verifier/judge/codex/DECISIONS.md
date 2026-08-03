# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for every file matching `sub-*/sub-*_behavior+ophys.nwb`, opens each NWB file with `h5py`, extracts subject/session metadata, and processes every discovered file. It does not use `pynwb`; it works directly against the HDF5 structure.

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
    session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
    return session_meta
```

iii. `CONVERSION_NOTES.md` says the supplied `data/` directory contains `152` NWB sessions across `11` subjects and that the final dataset therefore uses all `152` sessions. The trajectory also shows the AI intentionally audited the NWB layout before writing the converter.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from each file’s embedded NWB subject metadata, then uniqued and sorted numerically (`m3`, `m11`, etc.).

ii.
```python
with h5py.File(file_path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"][()])
...
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI appears to have preferred file-internal metadata over directory names so the subject labels come from the NWB source itself.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session number is parsed from the filename and carried in `SessionMeta`.

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
session_meta.append(
    SessionMeta(
        file_path=file_path,
        subject=subject,
        session_num=session_num,
        scene=scene,
        date=date,
    )
)
```

iii. `CONVERSION_NOTES.md` describes the final dataset as `152` sessions, which matches the “one file = one session” interpretation.

## 1-d. How are the data split into trials?

i. Trial starts are the indices where `trial_start > 0`. Trial ends are the indices where `teleport > 0`. For each trial the kept samples are `[start:stop)`, so the teleport sample itself is excluded.

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

iii. `CONVERSION_NOTES.md` explicitly states that trials are aligned to `trial start` and that the kept sample range is `[trial_start_idx : teleport_idx)`.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops entire trials when more than `30%` of frames in that trial have `lick count > 2`, treating those as lick-sensor artifact trials. It does not implement the human reference’s short-trial filter.

ii.
```python
def bad_lick_trial_mask(lick_counts: np.ndarray, start_idx: np.ndarray, stop_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask

...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. `CONVERSION_NOTES.md` says this was chosen to match the paper’s lick-sensor error criterion, and reports that `81` trials were removed this way.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural data are not taken from NWB `Deconvolved`. They are recomputed from raw `Fluorescence` and `Neuropil` arrays, after ROI curation with `iscell`.

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
...
plane_iscell = iscell[plane_slice, 0] == 1
...
fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the repo logic was treated as the operational reference, so the AI chose to recreate the paper-style preprocessing rather than use the stored NWB deconvolved signal.

## 2-b. How is the `neural` data processed?

i. The AI recomputes events by doing `0.7` neuropil subtraction, per-trial maximin baseline estimation, dF/F calculation, Gaussian smoothing, OASIS deconvolution, and then concatenating kept neurons across planes. It saves the resulting event activity as `neural`.

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
...
neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)
```

iii. `CONVERSION_NOTES.md` lists these steps explicitly and says they were chosen to mirror the paper repo’s neural preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, only curated Suite2p ROIs with `iscell[:, 0] == 1` are kept. Then putative interneurons are removed if their dF/F trace has Pearson correlation `> 0.5` with running speed over valid in-trial samples.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
curated_plane_indices = np.where(plane_iscell)[0]
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

iii. `CONVERSION_NOTES.md` says the AI followed the paper’s description of excluding speed-correlated putative interneurons, in addition to keeping only curated `iscell` ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing the session-long event matrix on the same `[trial_start_idx:teleport_idx)` intervals used to define trials.

ii.
```python
for out_idx, trial_idx in enumerate(kept_trial_indices):
    start = trial_start_idx[trial_idx]
    stop = trial_stop_idx[trial_idx]
    session_trial_blocks[out_idx].append(
        kept_events[:, start:stop].astype(np.float16, copy=False)
    )
```

iii. `CONVERSION_NOTES.md` states that trials are aligned to `trial start`, so no extra offsetting beyond trial segmentation is applied.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The AI keeps the native frame sampling and infers frame rate from behavior timestamps, then stores the mean session frame rate as metadata (`~64.48 ms` bins).

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

iii. `CONVERSION_NOTES.md` says the AI did not trust the NWB rate metadata for multi-plane sessions and therefore inferred frame rate from timestamps instead.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior `position` timestamps, after session-length truncation and trial slicing.

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes say `input[0]` is computed from behavior timestamps relative to the first frame of each trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI subtracts the first timestamp of that trial from every timestamp in that trial.

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

iii. The notes explicitly describe this as “computed from the behavior timestamps relative to the first frame of each trial.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses exactly the same `[start:stop)` frame indices as the per-trial neural slice, so the time vector is frame-aligned with the saved event matrix.

ii.
```python
neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The AI’s working assumption, reflected in the code and notes, is that the behavior timestamps and imaging frames are already sample-aligned once all streams are truncated to the same session length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved environment label is derived primarily from session metadata (`identifier` scene string), not directly from the per-frame `environment` timeseries. The behavior `environment` stream is only used as a consistency check.

ii.
```python
identifier = decode_scalar(f["identifier"][()])
...
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
env_timeseries = behavior["environment"]["data"][:].astype(np.float32)
...
np.full(trial_times.shape, float(label.env), dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says trial environment labels were derived from the session scene name and cross-checked against the aligned behavior `environment` stream on every kept trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses scene strings such as `Env1_LocationA`, `Env1_LocationA_to_B`, and `Env1_A_to_Env2_B`. For switch sessions it hardcodes the switch at trial `30`, then repeats the resulting environment label across all timepoints of that trial.

ii.
```python
def parse_scene(scene: str, n_trials: int, switch_trial_count: int = 30) -> list[TrialLabel]:
    ...
    return [
        TrialLabel(
            env=env0 if trial_idx < switch_trial_count else env1,
            zone=zone0 if trial_idx < switch_trial_count else zone1,
        )
        for trial_idx in range(n_trials)
    ]

...
np.full(trial_times.shape, float(label.env), dtype=np.float32)
```

iii. The notes say this was done to match the paper/task structure, with the switch point fixed at trial `30`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the NWB `trial number` behavior timeseries, sampled at the trial start index.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` states that `trial_number` is copied from the aligned NWB `trial number` stream at trial start.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra transformation is applied beyond taking the value at trial start and repeating it across all frames in the trial.

ii.
```python
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The AI’s notes imply that the stored `trial number` stream already had the desired per-trial label, so it was propagated over the trial window.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` timestamps, together with the per-trial start and stop times.

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

iii. `CONVERSION_NOTES.md` says reward outcome and previous-trial outcome were computed before removing lick-artifact trials so that a kept trial still uses the actual previous trial’s outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first marks each trial as rewarded or omitted based on whether any reward timestamp falls between that trial’s start and teleport time. It then shifts that vector by one trial, with the first trial set to `0`, and repeats the result across all frames in the current trial.

ii.
```python
def reward_outcome_per_trial(reward_timestamps: np.ndarray, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    ...
    while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
        outcomes[trial_idx] = 1
        probe_idx += 1
    return outcomes

...
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
...
np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32)
```

iii. The notes explicitly justify computing previous outcome before trial exclusion so the label remains tied to the actual immediately preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from clipped position samples plus a reward-zone label that comes from parsed session scene metadata (`A/B/C`), not from the raw `reward_zone` behavior stream.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
label = trial_labels[trial_idx]
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
...
discretize_distance_to_zone(trial_pos, zone_bounds)
```

iii. `CONVERSION_NOTES.md` says trial reward-zone labels were derived from the session scene name and that the scene parser was treated as matching the task structure in the paper.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest reward-zone edge: negative before the zone, zero inside it, positive after it. It then discretizes that value.

ii.
```python
signed_distance = np.where(
    position_cm < zone_start,
    position_cm - zone_start,
    np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0),
)
```

iii. The notes describe this as the “signed nearest distance to the active reward zone.”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Thresholding is implemented manually into `7` bins: `< -50`, `[-50, -10]`, `(-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
bins = np.zeros_like(position_cm, dtype=np.int8)
bins[signed_distance < -50.0] = 0
bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
bins[signed_distance == 0.0] = 3
bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
bins[signed_distance > 50.0] = 6
```

iii. `CONVERSION_NOTES.md` says these were the prompt-requested `7` categories, implemented directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial `[start:stop)` slices used for neural activity.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
kept_events[:, start:stop]
```

iii. The code and notes both assume samplewise alignment once all streams are truncated to the same session length.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavior `position` timeseries.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The notes describe absolute position as corridor position clipped to `0` to `450` cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips positions to `[0, 450)` cm and then bins them into five equal 90 cm bins by flooring `position / 90`.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. `CONVERSION_NOTES.md` says the corridor was treated as `0` to `450` cm and divided into five equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Categories are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm, produced by the floor-based binning above.

ii.
```python
dataset["output_values"] = [
    ...,
    ["0_to_90cm", "90_to_180cm", "180_to_270cm", "270_to_360cm", "360_to_450cm"],
    ...
]
```

iii. The AI’s justification in the notes is that this directly satisfies the prompt’s “5 equal-sized bins” requirement.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial frame slice as neural data.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The notes treat behavior and neural streams as already frame-aligned after common-length truncation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` timeseries.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
...
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` describes lick as binary `lick_count > 0`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick counts are thresholded to a binary indicator: `0` for no lick, `1` for any positive lick count.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The notes say the decoder output was built as binary `lick_count > 0`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same per-trial `[start:stop)` slice as the neural data.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
...
kept_events[:, start:stop]
```

iii. The AI’s general alignment strategy is to slice all streams with the same frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from parsed session scene metadata, not from the raw `reward_zone` behavior signal.

ii.
```python
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
label = trial_labels[trial_idx]
zone_idx = ZONE_TO_INDEX[label.zone]
```

iii. `CONVERSION_NOTES.md` says the trial reward-zone labels were derived from the session scene name and checked against task structure expectations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene name, assigns `A/B/C` labels trial-by-trial with a hardcoded switch at trial `30` for switch sessions, maps `A/B/C` to `0/1/2`, and repeats that label across all frames of the trial.

ii.
```python
label = trial_labels[trial_idx]
zone_idx = ZONE_TO_INDEX[label.zone]
...
np.full(trial_times.shape, zone_idx, dtype=np.int8)
```

iii. The notes say this was intended to match the session/task design described in the paper.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` timestamps together with per-trial start and stop times.

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

iii. The notes describe reward outcome as based on reward timestamps falling between trial start and teleport.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls within that trial’s `[start_time, stop_time)` window. The result is a binary trial-level label, repeated across all frames of the trial.

ii.
```python
while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
    outcomes[trial_idx] = 1
    probe_idx += 1

...
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` states that reward outcome is a trial-level label computed from reward timestamps between trial start and teleport.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates every behavior stream, timestamp vector, and fluorescence plane to the common minimum session length. It treats lick-sensor artifact trials as invalid and drops them. It also checks that scene-derived and behavior-derived environment labels agree on kept trials. It does not implement explicit handling of missing `reward_zone` data because it never uses the raw `reward_zone` stream.

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
position = position[:session_len]
...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
...
if env_in_trial.size:
    env_mode = int(round(float(np.median(env_in_trial))))
    if env_mode != label.env:
        raise RuntimeError(...)
```

iii. `CONVERSION_NOTES.md` says common-length truncation was needed because some multi-plane sessions differ by one frame, and that the lick-artifact filter was chosen to match the paper’s criterion.

## 13-a. What are the most time-consuming steps of the code?

i. The slowest parts are reading large HDF5 arrays, recomputing dF/F and OASIS events block-by-block for every plane/session, and then building/storing the full trialized dataset. The AI’s implementation makes neural preprocessing much heavier than a simple read-and-slice converter.

ii.
```python
for plane_key in plane_keys:
    fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
    neuropil_ds = ophys["Neuropil"][plane_key]["data"]
    ...
    for block_start in range(0, len(curated_plane_indices), block_size):
        ...
        dff, events = compute_dff_and_events(...)
```

iii. The trajectory explicitly says the AI was monitoring full-run timing around deconvolution and interneuron exclusion, and the code structure makes those the dominant costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial lick-artifact loop, the reward-outcome loop over trials, the per-neuron correlation list comprehension used for interneuron removal, and the repeated per-trial concatenation/slicing loops could all be vectorized or batched further.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
    if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
        mask[trial_idx] = True

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
```

iii. The AI did not explicitly justify these loops in notes, but the code shows they were accepted as a pragmatic implementation tradeoff.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly loops over trials inside `compute_dff_and_events` to mask/fill trial periods and then loops over the same trial boundaries again to extract trial-aligned neural blocks. It also recomputes per-block correlations after already computing dF/F solely to decide which neurons to keep.

ii.
```python
for start, stop in zip(start_idx, stop_idx):
    f[:, start:stop] = fluorescence[:, start:stop]
    f_neu[:, start:stop] = neuropil[:, start:stop]

...
for start, stop in zip(start_idx, stop_idx):
    sl = slice(start, stop)
    ...

...
for out_idx, trial_idx in enumerate(kept_trial_indices):
    start = trial_start_idx[trial_idx]
    stop = trial_stop_idx[trial_idx]
    session_trial_blocks[out_idx].append(
        kept_events[:, start:stop].astype(np.float16, copy=False)
    )
```

iii. No explicit written justification was given beyond following the repo-style preprocessing; this repetition is visible directly in the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and keeps full dF/F traces only to support interneuron filtering and OASIS deconvolution, but the saved dataset contains only event activity. It also reads `planeIdx`, stores `date` in `SessionMeta`, and cross-checks the behavior `environment` stream without using those values in the final tensors.

ii.
```python
dff, events = compute_dff_and_events(...)
...
dff_valid = dff[:, valid_mask]
...
neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)

...
plane_idx = imgseg["planeIdx"][:].astype(np.int16)
...
if env_in_trial.size:
    env_mode = int(round(float(np.median(env_in_trial))))
    if env_mode != label.env:
        raise RuntimeError(...)
```

iii. This is not justified explicitly in the notes; it follows from the AI’s choice to rebuild the paper-style preprocessing pipeline and then save only the final event matrix.
