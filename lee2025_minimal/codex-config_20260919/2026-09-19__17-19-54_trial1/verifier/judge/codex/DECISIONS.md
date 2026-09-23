# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the `.mat` files. It hard-coded the seven subject IDs, loaded one joblib file per subject from `/app/data`, and read precomputed arrays named `trace`, `position`, `envs`, and `blocked`. It then iterated through the first axis of those arrays as sessions and later split each session into trials.

ii.
```python
SUBJECTS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-51",
    "QLAK-CA1-56",
    "QLAK-CA1-74",
    "QLAK-CA1-75",
]

for subject_number, subject in enumerate(SUBJECTS):
    source_path = data_dir / subject
    wrapped = joblib.load(source_path)
    source = wrapped[subject]
    traces = np.asarray(source["trace"])
    positions = np.asarray(source["position"])
    environments = np.asarray(source["envs"]).reshape(-1)
```

iii. The trajectory says the agent believed "the source data already contains the paper’s final binary rising-phase events and aligned x–y tracking" and therefore chose the published joblib files instead of reproducing upstream preprocessing from the `.mat` files (step 9). The same rationale was copied into the script docstring (step 26).

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `SUBJECTS` list. Each list entry is treated as one animal, and the list index is used for `subject_idx`.

ii.
```python
for subject_number, subject in enumerate(SUBJECTS):
    source_path = data_dir / subject
    ...
    subject_idx.append(subject_number)
```

iii. The trajectory does not show a separate argument for subject identification beyond inspecting the available files and then fixing the subject list in code. The final dataset summary reports the same seven hard-coded subjects (steps 40 and 45).

## 1-c. How are the data split into sessions?

i. The AI treated each source recording day as one output session. It iterated over the first axis of the per-subject `trace` and `position` arrays and appended one session per day.

ii.
```python
if traces.shape[0] != positions.shape[0] or traces.shape[0] != len(environments):
    raise ValueError(f"Session count mismatch for {subject}")

for day in range(traces.shape[0]):
    position = positions[day]
    ...
    neural.append(...)
    decoder_input.append(...)
    decoder_output.append(...)
```

iii. The agent explicitly documented "A source recording day is one session" in the script header (step 26). In step 9 it described "session-level cell registration" as one of the central design choices.

## 1-d. How are the data split into trials?

i. Each continuous session is split into consecutive non-overlapping 60-second trials. With a 30 Hz source rate, the code uses 1800 raw frames per trial and drops any leftover tail that does not fill a full minute.

ii.
```python
SOURCE_FPS = 30
TRIAL_SECONDS = 60
RAW_FRAMES_PER_TRIAL = TRIAL_SECONDS * SOURCE_FPS

total_frames = traces.shape[2]
n_trials = total_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
...
neural.append(
    [np.ascontiguousarray(binned_neural[:, trial, :]) for trial in range(n_trials)]
)
```

iii. The trajectory says "Sessions without a complete final minute will have only full one-minute trials retained" (step 21). The docstring repeats that only complete consecutive 60 s trials are kept (step 26).

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial behavioral or neural quality-control filter. The only trial-level filtering is structural: incomplete trailing data are discarded because only full 60-second trials are retained. At the session level, the code also raises an error if fewer than two complete trials are available.

ii.
```python
n_trials = total_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
if n_trials < 2:
    raise ValueError(f"Fewer than two complete trials for {subject}, day {day}")
```

iii. The agent justified this as a decoder-format requirement rather than a biological QC rule: step 21 says only full one-minute trials are kept, and the script header says it discards short tails rather than padding or changing the time-bin duration (step 26).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from `source["trace"]` in the subject joblib files. The AI treated these values as already-thresholded rising-phase calcium-event traces rather than as raw fluorescence.

ii.
```python
source = wrapped[subject]
traces = np.asarray(source["trace"])
...
selected = traces[day, cell_mask, :].astype(np.float32, copy=True)
```

iii. In step 9 the agent stated that the source already contained "final binary rising-phase events." The script header repeats that the converter starts from "thresholded rising-phase calcium events" already present in the published joblib files (step 26).

## 2-b. How is the `neural` data processed?

i. The AI filtered cells by movement-related activity, smoothed the selected event traces with a temporal Gaussian (`sigma=3` frames), then average-pooled them into non-overlapping 1-second bins. The stored neural trials are therefore `(n_cells, 60)` float32 matrices, not raw 30 Hz traces.

ii.
```python
selected = traces[day, cell_mask, :].astype(np.float32, copy=True)
smoothed = np.empty_like(selected)
gaussian_filter1d(selected, sigma=3, axis=1, output=smoothed)
...
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The agent said it was following the repository’s within-session decoding pipeline: step 26 says the repository smooths event traces with a 3-frame Gaussian and that the converter "retain[s] that smoothing and pool[s] to non-overlapping 1 s bins." Step 45 highlights filtering, smoothing, and spatial binning as the main documented choices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are kept only if they have more than five events while the animal is moving faster than 5 cm/s. Unregistered cells with NaNs fail this test automatically because the `np.sum(...) > 5` comparison becomes false.

ii.
```python
def moving_mask(position: np.ndarray) -> np.ndarray:
    speed = np.zeros(position.shape[1], dtype=np.float64)
    instantaneous = np.linalg.norm(np.diff(position, axis=1) * SOURCE_FPS, axis=0)
    speed[1:] = gaussian_filter1d(instantaneous, sigma=5)
    return speed > 5.0

moving = moving_mask(position)
events_while_moving = np.sum(traces[day][:, moving], axis=1)
cell_mask = events_while_moving > 5
```

iii. The script header says this matches `decode_position_within` from the paper repository: cells must have more than five events while speed exceeds 5 cm/s, with the speed estimate Gaussian-filtered over 5 frames (step 26). Step 31 says the "activity filter is working as intended" after an indexing bug was fixed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event-based alignment to an external stimulus. Trials are aligned to the start of each consecutive complete 60-second recording segment, and the metadata explicitly describes that as the temporal alignment event.

ii.
```python
"metadata": {
    ...
    "temporal_alignment_event": "start of each consecutive complete 60-second segment",
    "off_start": 0.0,
    "off_end": 60.0,
    ...
}
```

iii. The agent did not identify any stimulus-locked event in the source data. Instead, step 21 says the session is chopped into complete one-minute chunks, and the metadata in the finished script encodes the start of each chunk as the alignment point (step 26).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural and output time series use 1-second bins. Yes, temporal rebinning is applied: 30 source frames are pooled into each output time bin after Gaussian smoothing of the neural traces.

ii.
```python
TIME_BIN_FRAMES = 30
TIME_BIN_MS = 1000.0
TIME_BINS_PER_TRIAL = int(TRIAL_SECONDS * SOURCE_FPS / TIME_BIN_FRAMES)
...
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The trajectory is explicit here: step 26 says the converter pools to non-overlapping 1 s bins because the agent considered that a tractable representation for downstream decoding. Steps 40 and 45 confirm the final output has 60 time bins per 60-second trial.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input geometry is derived from the session-specific `source["blocked"]` field, not from `envs`. `envs` is only saved into session metadata.

ii.
```python
environments = np.asarray(source["envs"]).reshape(-1)
...
geometry, blocked_bins = blocked_geometry(source["blocked"][day])
```

iii. In step 9 the agent described "translating each environment’s blocked sectors into decoder inputs" as a central task. Step 21 then explains why it re-indexed those blocked sectors before using them.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The code converts each session’s blocked-sector indices into a 3x3 binary matrix, transposes that matrix, flattens it, and stores the result as a 9-element float32 vector with `1=blocked`. The same vector is copied into every trial of that session.

ii.
```python
raw = np.asarray(blocked_for_day[0]).reshape(-1)
blocked_yx = np.zeros((N_SPATIAL_BINS, N_SPATIAL_BINS), dtype=np.float32)
for value in raw:
    idx = int(value)
    if idx >= 0:
        blocked_yx.flat[idx] = 1.0

blocked_xy = blocked_yx.T
vector = blocked_xy.reshape(-1)
...
decoder_input.append([geometry.copy() for _ in range(n_trials)])
```

iii. The recorded justification is in step 21: the agent believed the `blocked` indexing was transposed relative to the `(x, y)` position arrays and therefore converted the geometry into the same `x-major` convention as its output classes. That same rationale appears in the header comments (step 26).

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The output position labels are derived from `source["position"]` in the subject joblib files.

ii.
```python
positions = np.asarray(source["position"])
...
position = positions[day]
classes = position_classes(position, used_frames).reshape(
    n_trials, TIME_BINS_PER_TRIAL
)
```

iii. Step 9 says the source joblib files already contain aligned x-y tracking, so the agent chose to derive the decoder target directly from that `position` array.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The code first averages the 2D position within each 1-second bin, then converts the average position to one of nine 25 cm x 25 cm spatial sectors. The resulting class sequence is reshaped into `(n_trials, 60)` and each trial is stored as shape `(1, 60)`.

ii.
```python
mean_position = position[:, :used_frames].reshape(
    2, -1, TIME_BIN_FRAMES
).mean(axis=2)
xy_bin = np.floor(mean_position / SPATIAL_BIN_CM).astype(np.int64)
xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy_bin[0] + xy_bin[1]).astype(np.int64)
...
decoder_output.append(
    [np.ascontiguousarray(classes[trial][None, :]) for trial in range(n_trials)]
)
```

iii. The justification is explicit in the script header: the agent wanted the output to be "averaged within each temporal bin and discretized into the arena’s 3 x 3, 25-cm sectors" (step 26). Steps 40 and 45 confirm that the final saved trials have 60 time bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each coordinate is thresholded by flooring `x` and `y` after dividing by 25 cm, then clipping each axis to the valid range `[0, 2]`. The final category label is `3 * x_bin + y_bin`, i.e. an `x-major` class order.

ii.
```python
xy_bin = np.floor(mean_position / SPATIAL_BIN_CM).astype(np.int64)
xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
return (N_SPATIAL_BINS * xy_bin[0] + xy_bin[1]).astype(np.int64)
```

iii. Step 21 says the agent intentionally used an `x-major` nine-bin convention so that geometry and position labels matched. The same formula is documented in the script header and metadata (step 26).

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Neural activity and position are aligned by using the same raw frame range (`used_frames`), the same 1-second bin boundaries, and the same trial reshaping. Neural trials and output trials therefore share the same 60 bins per 60-second trial.

ii.
```python
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)

classes = position_classes(position, used_frames).reshape(
    n_trials, TIME_BINS_PER_TRIAL
)
```

iii. Step 9 says the agent believed the source traces and tracking were already timestamp-aligned. Steps 21 and 26 show that it then enforced a shared binning and trialization scheme for both streams.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles absent cells indirectly: cells with NaNs do not pass the movement-event threshold and are dropped. It also clips exact arena-boundary positions into the outermost spatial bin and discards any incomplete session tail shorter than 60 seconds. It does not impute missing positions or patch corrupted sessions.

ii.
```python
events_while_moving = np.sum(traces[day][:, moving], axis=1)
cell_mask = events_while_moving > 5
...
xy_bin = np.floor(mean_position / SPATIAL_BIN_CM).astype(np.int64)
xy_bin = np.clip(xy_bin, 0, N_SPATIAL_BINS - 1)
...
"discarded_tail_frames": int(total_frames - used_frames),
```

iii. The trajectory justification appears in two places: step 26 says absent cells are NaN in the registered-cell tensor and are removed session by session, and step 21 says short tails are discarded rather than padded. The boundary clipping is only justified in the inline code comment, which says exact 75 cm values should map to the outermost sector rather than an invalid bin.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive steps in this code are loading the large joblib tensors for each subject, computing the movement mask, Gaussian-smoothing all retained neural traces, and then reshaping and averaging them into 1-second bins. Those are the only heavy per-frame operations.

ii.
```python
wrapped = joblib.load(source_path)
...
moving = moving_mask(position)
...
gaussian_filter1d(selected, sigma=3, axis=1, output=smoothed)
...
binned_neural = smoothed[:, :used_frames].reshape(
    n_cells, n_trials, TIME_BINS_PER_TRIAL, TIME_BIN_FRAMES
).mean(axis=3, dtype=np.float32)
```

iii. The trajectory does not spell out a runtime profile, but the agent repeatedly monitored the long-running conversion and validation steps and described the main operations it added: activity filtering, smoothing, and pooling (steps 31, 40, and 45).

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest avoidable Python loop is the loop over blocked-sector indices in `blocked_geometry`. The per-trial list construction for `neural`, `input`, and `output` also materializes many small arrays and copies instead of keeping more vectorized session-level blocks until the end.

ii.
```python
for value in raw:
    idx = int(value)
    if idx >= 0:
        blocked_yx.flat[idx] = 1.0

neural.append(
    [np.ascontiguousarray(binned_neural[:, trial, :]) for trial in range(n_trials)]
)
decoder_input.append([geometry.copy() for _ in range(n_trials)])
decoder_output.append(
    [np.ascontiguousarray(classes[trial][None, :]) for trial in range(n_trials)]
)
```

iii. There is no explicit trajectory justification for keeping these loops. The trajectory focuses on correctness of indexing and alignment rather than optimization.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats static geometry replication for every trial, repeats per-trial array materialization from already-binned session arrays, and recomputes `total_frames`, `n_trials`, and `used_frames` for every day even though frame count is constant within a subject tensor.

ii.
```python
total_frames = traces.shape[2]
n_trials = total_frames // RAW_FRAMES_PER_TRIAL
used_frames = n_trials * RAW_FRAMES_PER_TRIAL
...
decoder_input.append([geometry.copy() for _ in range(n_trials)])
...
[np.ascontiguousarray(classes[trial][None, :]) for trial in range(n_trials)]
```

iii. The trajectory does not claim these repetitions were intentional. The emphasis was on matching the agent’s chosen decoding convention and satisfying the validator, not on removing redundant work.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and stores metadata that the downstream decoder does not use, including `environment`, `blocked_bins_x_major`, `source_frame_count`, `used_frame_count`, `discarded_tail_frames`, `n_registered_cells`, and `n_cells_after_activity_filter` for every session. It also loads `envs` only to place it into metadata, not into `input`.

ii.
```python
environments = np.asarray(source["envs"]).reshape(-1)
...
session_info.append(
    {
        "subject": subject,
        "source_day_index": int(day),
        "environment": str(environments[day]),
        "blocked_bins_x_major": blocked_bins,
        "source_frame_count": int(total_frames),
        "used_frame_count": int(used_frames),
        "discarded_tail_frames": int(total_frames - used_frames),
        "n_complete_trials": int(n_trials),
        "n_registered_cells": int(np.all(np.isfinite(traces[day]), axis=1).sum()),
        "n_cells_after_activity_filter": n_cells,
    }
)
```

iii. The trajectory never claims this metadata is needed for training. Step 45 only says the converter documents its choices; the extra bookkeeping is therefore best understood as explanatory output rather than decoder-required processing.
