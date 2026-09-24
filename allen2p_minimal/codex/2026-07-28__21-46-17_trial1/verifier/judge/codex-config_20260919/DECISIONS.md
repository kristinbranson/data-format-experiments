# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the local `behavior_ophys_experiments` directory for NWB IDs, intersects them with the CSV experiment table, keeps `active_behavior`, and loads each selected NWB directly with `BehaviorOphysExperiment.from_nwb_path`. Thus “all” means all locally present active-behavior experiment files, including both project codes present locally.

ii. ```python
available_ids = get_available_experiment_ids(data_root)
exp_table = load_experiment_table(data_root)
selected_df = select_experiments(exp_table, available_ids, args.max_sessions)
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
```

iii. The trajectory says the local data are an incomplete 284-file subset, so the AI deliberately processed what was locally available. It inspected SDK objects and chose direct local NWB loading to avoid assuming that the full manifest was available.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among retained sessions; each session receives the corresponding sorted subject index.

ii. ```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
subject_idx.append(subject_to_idx[session["mouse_id"]])
```

iii. The AI treated the SDK metadata’s `mouse_id` as the animal identifier and reported 38 locally represented mice.

## 1-c. How are the data split into sessions?

i. One output session is one `ophys_experiment_id` (one imaging plane), not one `ophys_session_id`; the latter is retained only as metadata.

ii. ```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    session, stats = load_session(experiment_id, nwb_path, meta_row)
```

iii. The AI explicitly reasoned that a plane has one native ophys timestamp stream and that keeping planes separate avoids merging differing neuron sets/timestamps. It initially considered aggregation, then chose experiment-level sessions.

## 1-d. How are the data split into trials?

i. The AllenSDK `trials` table supplies trial start/stop boundaries. For each retained trial, the AI selects change-detection `stimulus_presentations` whose starts lie inside those boundaries, making a variable-length sequence of presentation intervals.

ii. ```python
for trial_id, row in valid_trials.iterrows():
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
```

iii. After testing a fixed change-centered window, the AI inspected actual trials and concluded that native trials contain 10–17 image-presentation intervals and should retain the full SDK trial extent.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be go or catch, non-aborted, non-auto-rewarded, and have finite `change_time`. Trials are additionally dropped for missing stimulus rows, non-finite binned neural/behavior values, all-zero neural data, or inconsistent counts of `is_change`. Sessions require at least two trials both before and after binning.

ii. ```python
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
if np.any(~np.isfinite(neural_trial)) or np.all(neural_trial == 0):
    continue
```

iii. The core exclusions follow the prompt. The AI added silent-trial filtering after validator warnings and change-flag checks after inspecting stimulus omissions; it considered these necessary for coherent decoder labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from AllenSDK inferred calcium `events`, not dF/F traces.

ii. ```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
```

iii. The trajectory and notes say the AI interpreted the paper as using inferred/discrete calcium events and therefore preferred `events` to fluorescence.

## 2-b. How is the `neural` data processed?

i. Event traces are averaged within successive stimulus-presentation bins. Empty bins use the nearest sample. Each experiment/plane is processed separately; planes are not stacked.

ii. ```python
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + 0.75]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
# inside reduce_to_bins
reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The AI sought to match the paper’s assignment of activity to 750 ms image-presentation intervals while preserving trial structure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are excluded by the SDK loader. Sessions with no events and trials with non-finite or entirely zero binned neural arrays are removed.

ii. ```python
BehaviorOphysExperiment.from_nwb_path(str(nwb_path), exclude_invalid_rois=True)
if len(dataset.events) == 0: return None, {"skip_reason": "no_valid_rois"}
if np.any(~np.isfinite(neural_trial)) or np.all(neural_trial == 0): continue
```

iii. The AI verified the SDK’s invalid-ROI behavior and added the all-zero rule after validation exposed silent trials that it judged unhelpful for decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural column is aligned to a stimulus-presentation onset within the SDK trial; the trial begins with its first selected presentation rather than retaining every native ophys frame from trial start.

ii. ```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The AI’s final reasoning was that presentation-onset alignment most directly implements the paper’s interval assignment; it abandoned an earlier fixed window around change time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 750 ms. Native event samples are averaged into image-presentation intervals, so temporal rebinning is applied.

ii. ```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
if not math.isclose(args.time_bin_ms, TIME_BIN_MS_DEFAULT, ...):
    raise ValueError("This converter uses native 750 ms image-presentation intervals.")
```

iii. The AI cited the 250 ms flash plus 500 ms gray task cadence and the paper’s 750 ms “image presentation interval.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It derives from `stimulus_presentations.image_name` and `stimulus_presentations.omitted` within the change-detection block.

ii. ```python
for image_name, omitted in zip(
    trial_stim["image_name"].tolist(), omitted_flags.tolist()
)
```

iii. The AI chose the stimulus table because it supplies the actual identity for every presentation, including repeated post-change images and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted presentations are labeled `gray`; all labels are collected globally, sorted, and integer encoded.

ii. ```python
NO_IMAGE_LABEL = "gray"
NO_IMAGE_LABEL if omitted else str(image_name)
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
```

iii. The AI reasoned that the requested identity is the image during the non-gray portion, while omissions require an explicit no-image category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Each stimulus row supplies one identity and exactly the same onset-defined interval used to average one neural column.

ii. ```python
image_identity = np.asarray([
    image_name_to_idx[name]
    for name in session["interval_image_names"][trial_idx]
], dtype=np.int64)
```

iii. The AI emphasized direct bin-wise labeling from the stimulus table to avoid treating the whole post-change trial as a constant label.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is taken from `stimulus_presentations.is_change`; trial `go`/`catch` is used to validate that go trials contain one change and catch trials none.

ii. ```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
```

iii. The AI considered the stimulus table the most direct source of the true change interval and used trial type only as a consistency check.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Missing flags become false, flags are cast to Boolean/integer, and malformed go/catch flag counts cause trial exclusion.

ii. ```python
if bool(row["go"]) and int(change_flags.sum()) != 1: continue
if bool(row["catch"]) and int(change_flags.sum()) != 0: continue
image_change = np.asarray(..., dtype=np.int64)
```

iii. The AI added the count checks as label sanity checks after investigating omission timing.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is applied: the raw Boolean `is_change` becomes category 0 (`no_change`) or 1 (`change`).

ii. ```python
"output_values": [
    ...,
    ["no_change", "change"],
]
```

iii. The raw SDK field is already categorical, so the AI retained it directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. One change flag is paired with the neural average for the same onset-defined 750 ms stimulus interval.

ii. ```python
output_trial = np.vstack([image_identity, image_change, running_bins,
                          pupil_bins, trial_outcome])
```

iii. The AI designed all time-varying outputs and neural activity around the same presentation bins.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `dataset.running_speed.speed` and its `timestamps`.

ii. ```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
```

iii. The AI used the SDK’s synchronized running-wheel stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are removed, speed is averaged within each presentation interval (nearest sample for an empty bin), and the resulting values are globally discretized.

ii. ```python
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
running_edges = compute_quantile_edges(all_running_values, nbins=5)
```

iii. The AI wanted behavior summarized over exactly the same 750 ms units as neural events and computed quintiles over all retained data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global quantile cut points create five approximately equal-occupancy categories; ties are made strictly increasing with `nextafter`.

ii. ```python
percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, percentiles)
return np.searchsorted(edges, values, side="right").astype(np.int64)
```

iii. This directly implements the requested five equal percentile bins and, according to the trajectory, produced balanced counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples are independently timestamp-selected and averaged over the identical stimulus bin edges used for neural activity.

ii. ```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```

iii. The AI used shared absolute-time bin edges to synchronize streams without first resampling running to ophys frames.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `dataset.eye_tracking.pupil_width` and eye-tracking timestamps. The `likely_blink` column is not used explicitly.

ii. ```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(), eye_timestamps
)
```

iii. The AI selected pupil width as the pupil-size measure and handled missing samples through interpolation; its notes do not justify omitting the SDK blink flag.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaNs are linearly interpolated in eye-time (a single valid value is broadcast), then values are averaged per presentation interval and globally quantile-discretized.

ii. ```python
filled = np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
```

iii. The AI described interpolation as missing-data repair and interval means as alignment with the paper’s temporal unit.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global quantile thresholds create five categories, using the same functions as running speed.

ii. ```python
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. This implements the requested five equal percentile bins across the kept dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye samples are averaged over the same presentation onset edges used for neural bins.

ii. ```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. Shared absolute-time intervals were intended to synchronize pupil and neural data despite different native sampling clocks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It derives from the trial-table Boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
if bool(row["hit"]): return TRIAL_OUTCOME_TO_INT["hit"]
if bool(row["miss"]): return TRIAL_OUTCOME_TO_INT["miss"]
```

iii. The AI used the AllenSDK’s curated behavioral outcome flags, matching the four requested go/catch outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true outcome flag maps to a fixed integer 0–3 and is repeated across all time bins in that trial; no valid flag raises an error.

ii. ```python
trial_outcome = np.full(T, session["trial_outcomes"][trial_idx], dtype=np.int64)
```

iii. Repetition makes the static trial label compatible with the common time-varying output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions are skipped for missing eye tracking, all-NaN pupil, no valid ROIs, missing running, too few trials, or missing stimulus tables. Pupil NaNs are interpolated; invalid running samples are removed; empty temporal bins use nearest samples; malformed/non-finite/silent trials are skipped; skip reasons are recorded in metadata.

ii. ```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
if valid.sum() == 0: return None
if hi <= lo: reduced[i] = values[nearest]
full_data["metadata"]["skipped_sessions"] = skipped
```

iii. The AI repeatedly smoke-tested and tightened handling after concrete validator findings. It favored dropping unusable units while preserving auditable skip reasons.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated AllenSDK/NWB deserialization is the dominant cost; the full conversion loads each selected experiment serially.

ii. ```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    session, stats = load_session(...)
```

iii. The trajectory explicitly reports that NWB loading, at roughly seconds per file, dominated runtime rather than binning or validation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial DataFrame filtering loop and `reduce_to_bins`’ per-bin loops (run separately for neural, running, and pupil) could be vectorized/group-reduced. The assembly loop and Python image-name mapping could also be vectorized, though I/O dominates.

ii. ```python
for trial_id, row in valid_trials.iterrows():
    ...
for i in range(n_bins):
    reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```

iii. The AI did not claim these loops were optimized; its trajectory judged loading to be the material bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. `reduce_to_bins` repeats timestamp searches/bin traversal for neural, running, and pupil for every trial. Dataset assembly is also run twice (full and sample), and final summary statistics rescan all outputs. NWB files themselves are loaded only once in the final conversion.

ii. ```python
neural_trial = reduce_to_bins(...)
running_trial = reduce_to_bins(...)
pupil_trial = reduce_to_bins(...)
full_data = convert_sessions_to_dataset(...)
sample_data = convert_sessions_to_dataset(...)
```

iii. The AI emphasized avoiding repeat NWB loads; the repeated in-memory passes were accepted as inexpensive relative to I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It creates and writes an unrequested sample dataset, computes extensive session summaries/sanity counters and end-of-run distribution scans, and retains metadata fields not consumed by decoder training. These are useful for auditing but discarded by downstream decoding.

ii. ```python
sample_data = convert_sessions_to_dataset(...)
with args.sample_output.open("wb") as f:
    pickle.dump(sample_data, f)
session_summary = summarize_sessions(session_stats, sessions)
```

iii. The trajectory shows these extras were deliberate smoke-test, validation, reproducibility, and documentation aids rather than required decoder inputs.
