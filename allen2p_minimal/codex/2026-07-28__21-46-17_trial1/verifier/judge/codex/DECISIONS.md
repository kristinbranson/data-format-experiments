# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local subset of Allen Visual Behavior NWB files from `/app/data/visual-behavior-ophys-1.1.0`, reads `ophys_experiment_table.csv`, filters to experiment IDs that have a local NWB file and `behavior_type == "active_behavior"`, then opens each experiment with `BehaviorOphysExperiment.from_nwb_path(...)`. It does not use the Allen project cache or load the full project-wide dataset.

ii. 
```python
def get_available_experiment_ids(data_root: Path):
    experiment_dir = data_root / "behavior_ophys_experiments"
    ...
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb")):
        ...

def load_experiment_table(data_root: Path):
    exp_table = pd.read_csv(data_root / "project_metadata" / "ophys_experiment_table.csv")

def select_experiments(exp_table: pd.DataFrame, available_ids, max_sessions=None):
    selected = exp_table.loc[exp_table.index.intersection(available_ids)].copy()
    selected = selected[selected["behavior_type"] == "active_behavior"].copy()

for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    nwb_path = (
        data_root / "behavior_ophys_experiments"
        / f"behavior_ophys_experiment_{experiment_id}.nwb"
    )
    session, stats = load_session(...)
```

iii. `CONVERSION_NOTES.md` says the local workspace does not contain the full Allen release and explicitly frames the conversion as operating on the local subset: “284 NWB experiment files present locally” and “202 active-behavior experiments selected from those files.”

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the selected experiment metadata.

ii.
```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["mouse_id"]])
```

iii. This is implicit in the code and consistent with the session metadata extracted from the experiment table; no separate justification beyond standard Allen metadata use is given.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one decoder session. It does not merge multiple experiments from the same `ophys_session_id`.

ii.
```python
for idx, (_, meta_row) in enumerate(selected_df.iterrows(), start=1):
    experiment_id = int(meta_row["ophys_experiment_id"])
    ...
    session, stats = load_session(
        experiment_id=experiment_id,
        nwb_path=nwb_path,
        meta_row=meta_row,
    )
```

iii. `CONVERSION_NOTES.md` states: “One decoder session is one `ophys_experiment_id`,” justified because one experiment is “one imaging plane with one ophys timestamp stream” and because this “avoids merging planes with different neuron sets.”

## 1-d. How are the data split into trials?

i. Trials are taken from `dataset.trials`. The AI filters the Allen trial table to valid go/catch trials, then for each trial uses the Allen `start_time` and `stop_time` bounds to collect all stimulus-presentation intervals whose onset falls inside the trial. Each resulting trial is a variable-length sequence of 750 ms stimulus intervals.

ii.
```python
trials = dataset.trials.copy()
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()

for trial_id, row in valid_trials.iterrows():
    trial_start = float(row["start_time"])
    trial_stop = float(row["stop_time"])
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
```

iii. `CONVERSION_NOTES.md` says the final representation uses “variable-length sequence[s] of native `750 ms` image-presentation intervals” inside each AllenSDK trial, based on the methods text and stimulus table.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps trials only if they are go or catch, not aborted, not auto-rewarded, and have finite `change_time`. It then further drops trials with no stimulus rows in the change-detection block, with non-finite binned neural/running/pupil values, with all-zero binned neural data, or with inconsistent `is_change` flags for go/catch status. Sessions with fewer than two surviving trials are skipped.

ii.
```python
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
...
if len(trial_stim) == 0:
    continue
...
if (
    np.any(~np.isfinite(neural_trial))
    or np.any(~np.isfinite(running_trial))
    or np.any(~np.isfinite(pupil_trial))
):
    continue
if np.all(neural_trial == 0):
    continue
...
if bool(row["go"]) and int(change_flags.sum()) != 1:
    ...
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    ...
    continue
...
if len(session["neural_trials"]) < 2:
    return None, {"skip_reason": "too_few_binned_trials", ...}
```

iii. `CONVERSION_NOTES.md` justifies the main Allen-table filter (`go`/`catch`, not `aborted`, not `auto_rewarded`, finite `change_time`). The added all-zero / non-finite / change-flag consistency checks are implicit in the code and are not separately justified beyond “sanity checks.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `dataset.events["events"]`, i.e. inferred calcium event traces, not `dff_traces`.

ii.
```python
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
...
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says: “Used `dataset.events["events"]` from AllenSDK” and justifies this as following “the paper’s use of inferred/discrete calcium events rather than raw fluorescence.”

## 2-b. How is the `neural` data processed?

i. The AI keeps one imaging plane per session, takes the event matrix for that plane, and reduces it into 750 ms stimulus-interval bins using `reduce_to_bins`, which computes a nanmean within each interval and falls back to the nearest sample if no timestamp falls inside a bin.

ii.
```python
def reduce_to_bins(values, timestamps, bin_edges):
    ...
    for i in range(n_bins):
        lo = start_idx[i]
        hi = end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
        else:
            nearest = np.searchsorted(timestamps, centers[i], side="left")
            ...
            reduced[:, i] = values[:, nearest]

...
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. `CONVERSION_NOTES.md` says the converter uses “variable-length sequence[s] of native `750 ms` image-presentation intervals” and that all streams are “reduced onto these interval bins using ophys / behavior timestamps.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI excludes invalid ROIs at load time, skips sessions with zero remaining event traces, and discards individual trials if their binned neural data are non-finite or entirely zero.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)

if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
...
if np.any(~np.isfinite(neural_trial)) ...:
    continue
if np.all(neural_trial == 0):
    continue
```

iii. `CONVERSION_NOTES.md` explicitly cites `exclude_invalid_rois=True` and says “Trials with non-finite binned values or all-zero neural activity were dropped.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to successive stimulus-presentation interval onsets within each trial, not to native ophys frames. It uses `stimulus_presentations.start_time` values inside the trial window as bin starts, plus one final edge at `last_start_time + 0.75`.

ii.
```python
trial_stim = stimulus_presentations[
    (stimulus_presentations["start_time"] >= trial_start - 1e-6)
    & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
].copy()
...
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. `CONVERSION_NOTES.md` says the interval-based representation was chosen because the paper and whitepaper describe the task in 750 ms flashed-image intervals and the methods assign events to those intervals.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 750 ms bins, one per image-presentation interval. The AI explicitly forces `--time-bin-ms=750` and rebins neural and behavioral streams onto those bins.

ii.
```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
...
if not math.isclose(args.time_bin_ms, TIME_BIN_MS_DEFAULT, ...):
    raise ValueError(
        "This converter uses native 750 ms image-presentation intervals. "
        "Keep --time-bin-ms=750."
    )
```

iii. `CONVERSION_NOTES.md` says the time axis is the “native `750 ms` image-presentation interval,” and the trajectory notes say the agent moved from a shorter window to trial-long “native per-trial image intervals.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations["image_name"]` for stimulus rows within each trial, plus the `omitted` flag to convert omitted flashes to the synthetic label `"gray"`.

ii.
```python
omitted_flags = trial_stim["omitted"].fillna(False).to_numpy(dtype=bool)
...
session["interval_image_names"].append(
    [
        NO_IMAGE_LABEL if omitted else str(image_name)
        for image_name, omitted in zip(
            trial_stim["image_name"].tolist(),
            omitted_flags.tolist(),
        )
    ]
)
```

iii. `CONVERSION_NOTES.md` says image identity is “interval-wise,” uses the flashed `image_name` for non-omitted intervals, and uses `gray` for omitted intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI collects the per-interval image-name sequence from the stimulus table, substitutes `"gray"` on omitted intervals, builds a global sorted mapping from image names to integer IDs, and converts each trial’s image-name sequence to integer labels.

ii.
```python
image_names = {NO_IMAGE_LABEL}
...
for trial_image_names in session["interval_image_names"]:
    image_names.update(trial_image_names)
...
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
...
image_identity = np.asarray(
    [
        image_name_to_idx[name]
        for name in session["interval_image_names"][trial_idx]
    ],
    dtype=np.int64,
)
```

iii. `CONVERSION_NOTES.md` justifies the `"gray"` label via omitted flashes and lists the final global label set.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned one label per 750 ms stimulus interval, using exactly the same trial-wise interval sequence as the binned neural data.

ii.
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
...
session["interval_image_names"].append([...])
...
image_identity = np.asarray(
    [image_name_to_idx[name] for name in session["interval_image_names"][trial_idx]],
    dtype=np.int64,
)
```

iii. `CONVERSION_NOTES.md` says every trial timepoint corresponds to one native 750 ms image-presentation interval and that all streams are aggregated over those same intervals.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `dataset.stimulus_presentations["is_change"]` for each interval, with additional consistency checks against trial-level `go`/`catch`.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    ...
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    ...
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
```

iii. `CONVERSION_NOTES.md` says image change is a “Binary interval-wise output from the Allen stimulus table `is_change`,” and that catch trials contain no positive change interval.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI uses the stimulus-table `is_change` flag directly, after filtering the stimulus table to the change-detection block and to the trial window. It also performs sanity checks that go trials have exactly one change interval and catch trials have none.

ii.
```python
stimulus_presentations = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"]
    .fillna("")
    .str.contains("change_detection")
].copy()
...
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    ...
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    ...
```

iii. The notes justify this as the native interval-wise representation of changes in the task and as a sanity check against omission/change inconsistencies.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is converted into a binary categorical variable with `0 = no_change` and `1 = change`.

ii.
```python
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
...
"output_values": [
    ...,
    ["no_change", "change"],
    ...
]
```

iii. This is implicit in the boolean `is_change` representation and in the declared output value names.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned as one binary label per 750 ms stimulus interval, using the same interval sequence as the binned neural data.

ii.
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
...
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
...
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
```

iii. `CONVERSION_NOTES.md` describes the full dataset as interval-wise, with neural and label streams aggregated over the same native image-presentation intervals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed["speed"]` and `dataset.running_speed["timestamps"]`.

ii.
```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
```

iii. `CONVERSION_NOTES.md` explicitly says: “Used AllenSDK `running_speed["speed"]`.”

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI removes non-finite running samples, averages running speed within each 750 ms trial interval using `reduce_to_bins`, computes global 5-quantile bin edges across all kept interval values, and digitizes each trial into quintile bins.

ii.
```python
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
...
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
...
running_edges = compute_quantile_edges(all_running_values, nbins=5)
...
running_bins = digitize_with_edges(
    session["running_cont"][trial_idx], running_edges
)
```

iii. `CONVERSION_NOTES.md` says running speed was “Binned by interval mean” and “Discretized into 5 global quantile bins across the full kept dataset.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into 5 global quantile bins named `bin_0` through `bin_4`.

ii.
```python
running_edges = compute_quantile_edges(all_running_values, nbins=5)
...
"output_values": [
    ...,
    [f"bin_{i}" for i in range(5)],
    ...
]
```

iii. `CONVERSION_NOTES.md` describes “5 global quantile bins across the full kept dataset.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by reducing the running stream onto the same per-trial 750 ms stimulus-interval bins used for the neural data.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The interval-based alignment is part of the main “Time axis” decision in `CONVERSION_NOTES.md`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking["pupil_width"]` and eye-tracking timestamps.

ii.
```python
eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
```

iii. `CONVERSION_NOTES.md` explicitly says the pupil-size measure is `eye_tracking["pupil_width"]`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI linearly fills NaNs in `pupil_width` over timestamp space with `fill_nan_by_time`, reduces the filled trace to 750 ms interval means with `reduce_to_bins`, computes global 5-quantile edges from all kept interval values, and digitizes each trial’s interval values. It does not remove `likely_blink` rows.

ii.
```python
def fill_nan_by_time(values, timestamps):
    ...
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)

...
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
...
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
...
pupil_bins = digitize_with_edges(
    session["pupil_cont"][trial_idx], pupil_edges
)
```

iii. `CONVERSION_NOTES.md` says: “Missing values were linearly interpolated in timestamp space before interval binning,” then interval means and global quantile binning were used.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into 5 global quantile bins named `bin_0` through `bin_4`.

ii.
```python
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
...
"output_values": [
    ...,
    [f"bin_{i}" for i in range(5)],
    ...
]
```

iii. `CONVERSION_NOTES.md` explicitly says pupil diameter is “Discretized into 5 global quantile bins across the full kept dataset.”

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by reducing the eye-tracking stream onto the same 750 ms stimulus-interval bins used for the neural data.

ii.
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. This follows the main interval-based time-axis decision documented in `CONVERSION_NOTES.md`.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
...
def get_trial_outcome(row):
    if bool(row["hit"]):
        return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]):
        return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]):
        return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]):
        return TRIAL_OUTCOME_TO_INT["correct_reject"]
```

iii. `CONVERSION_NOTES.md` says the trial-outcome categories are `hit`, `miss`, `false_alarm`, and `correct_reject`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the mutually exclusive trial outcome flags to integer IDs 0 to 3 and then repeats that single integer across all time bins in the trial.

ii.
```python
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
session["trial_outcomes"].append(get_trial_outcome(row))
...
trial_outcome = np.full(
    T, session["trial_outcomes"][trial_idx], dtype=np.int64
)
```

iii. `CONVERSION_NOTES.md` states that trial outcome is “Static per trial, repeated over all intervals in that trial.”

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data mostly by skipping sessions or trials. Sessions are skipped for missing eye tracking, all-NaN pupil traces, missing running speed, no valid ROIs, no change-detection stimulus table, or too few trials. Trial-level missing pupil values are linearly interpolated in time. Empty stimulus tables, non-finite reduced values, all-zero neural trials, and inconsistent change-flag patterns also cause the trial to be dropped. For empty bins in `reduce_to_bins`, the nearest sample is used.

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
...
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
...
if running_valid.sum() == 0:
    return None, {"skip_reason": "missing_running_speed"}
...
if len(trial_stim) == 0:
    continue
...
if np.any(~np.isfinite(neural_trial)) ...:
    continue
if np.all(neural_trial == 0):
    continue
...
return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)
...
if hi > lo:
    ...
else:
    nearest = np.searchsorted(timestamps, centers[i], side="left")
```

iii. `CONVERSION_NOTES.md` explicitly mentions missing-eye-tracking session skips and pupil interpolation. The broader trial-drop behavior is visible in `convert_data.py`; the notes frame these as sanity/quality checks.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading every NWB experiment with `BehaviorOphysExperiment.from_nwb_path(...)` and then iterating through all valid trials to bin neural, running, and pupil streams with `reduce_to_bins`.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
...
for trial_id, row in valid_trials.iterrows():
    ...
    neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
    running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
    pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. This is not stated directly in the notes, but it follows from the structure of the code and from the trajectory, which repeatedly discusses full NWB loading and full conversion reruns as the long-running part.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-trial loop over `valid_trials`, the per-bin loops inside `reduce_to_bins`, and the per-trial Python list comprehensions used to map image names to integer labels.

ii.
```python
for _, row in valid_trials.iterrows():
    ...

for i in range(n_bins):
    ...

image_identity = np.asarray(
    [
        image_name_to_idx[name]
        for name in session["interval_image_names"][trial_idx]
    ],
    dtype=np.int64,
)
```

iii. The AI does not explicitly justify leaving these as loops. They appear to have been kept for implementation simplicity.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several traversals over the same processed trial/session data. It separately scans all sessions to collect running values, pupil values, and image names; it converts sessions into decoder format twice (once for full data and once for sample data); and it walks the final dataset again to compute summary counts for printing.

ii.
```python
all_running_values.extend(session["running_cont"])
all_pupil_values.extend(session["pupil_cont"])
for trial_image_names in session["interval_image_names"]:
    image_names.update(trial_image_names)
...
full_data = convert_sessions_to_dataset(...)
...
sample_data = convert_sessions_to_dataset(...)
...
running_bins = np.concatenate(
    [trial[2] for session_trials in full_data["output"] for trial in session_trials]
)
```

iii. This repetition is not justified in the notes. It appears to be a pragmatic implementation choice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores substantial bookkeeping that is not needed for the decoder inputs/outputs themselves: omission sanity counters, detailed `session_summary` metadata, skip-reason metadata, sample-dataset generation, and trial/session bookkeeping fields such as `trial_ids`, `trial_interval_counts`, and `go_flags` that exist only to support summary reporting and metadata assembly.

ii.
```python
"sanity_total_omitted_intervals": 0,
"sanity_change_interval_omission_count": 0,
"sanity_pre_change_omission_count": 0,
"sanity_change_flag_mismatch_count": 0,
...
session["trial_ids"].append(int(trial_id))
session["trial_interval_counts"].append(int(neural_trial.shape[1]))
session["go_flags"].append(bool(row["go"]))
...
full_data["metadata"]["session_summary"] = session_summary.to_dict(orient="records")
full_data["metadata"]["skipped_sessions"] = skipped
...
sample_data = convert_sessions_to_dataset(...)
```

iii. `CONVERSION_NOTES.md` frames these extras as validation and sanity checks rather than core conversion logic. They are useful for auditing, but downstream decoding does not require most of them.
