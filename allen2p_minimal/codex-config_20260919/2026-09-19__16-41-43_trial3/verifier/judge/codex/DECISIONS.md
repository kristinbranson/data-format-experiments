# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local `ophys_experiment_table.csv`, intersects it with experiment IDs parsed from locally present NWB filenames, retains active non-passive experiments, and loads each retained NWB directly with `BehaviorOphysExperiment.from_nwb_path`. It produced 199 usable experiments from 202 active candidates, skipping three later for pupil-data failures.

ii.
```python
table = pd.read_csv(EXPERIMENT_TABLE, index_col="ophys_experiment_id")
local_ids = _local_experiment_ids()
table = table.loc[table.index.intersection(local_ids)].copy()
active = table[table["behavior_type"].eq("active_behavior") & ~table["passive"]]
...
dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
```

iii. The trajectory says passive replay sessions lack genuine choice/outcome labels and that local NWBs avoid downloading absent data. It explicitly chose every locally supplied, QC-passed active experiment.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` values among successfully converted experiments; each session gets the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({session["mouse_id"] for session in converted})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
[subject_lookup[session["mouse_id"]] for session in converted]
```

iii. The agent relied on the SDK/NWB mouse identifier as the stable animal identity. The trajectory reports 38 subjects in the completed artifact.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`—one imaging plane—is treated as one decoder session. Simultaneous planes sharing an `ophys_session_id` are not merged.

ii.
```python
for number, experiment_id in enumerate(experiments.index, start=1):
    converted.append(convert_experiment(int(experiment_id)))
...
"session_unit": "one QC-passed ophys experiment (imaging plane)",
```

iii. The trajectory says this matches the paper's plane-wise decoding and avoids interpolation among interleaved Multiscope planes with different timestamps/frame rates.

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. Each retained trial spans its experiment-defined `start_time` to `stop_time`, represented by 100 ms bin centers starting 50 ms after trial start; incomplete trailing fractions of a bin are discarded.

ii.
```python
duration = float(row["stop_time"] - row["start_time"])
n_bins = int(np.floor(duration / BIN_SEC))
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
```

iii. The agent described the rows as experiment-defined Go/Catch trials and used the full trial interval so all time-varying targets remain aligned.

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are retained; aborted and auto-rewarded trials are removed. Empty trials are skipped, and an experiment is rejected if fewer than two nonempty trials remain. Active/non-passive experiment filtering is also applied upstream.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
...
if len(trial_centers) < 2:
    raise ValueError(...)
```

iii. The trajectory says passive outcomes are synthetic, while aborted/auto-rewarded trials do not represent the requested normal Go/Catch task. The two-trial minimum is required by the decoder format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the AllenSDK inferred calcium-event trace in `dataset.events["events"]`, plus `dataset.ophys_timestamps` for timing.

ii.
```python
events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
ophys_times = np.asarray(dataset.ophys_timestamps, dtype=np.float64)
```

iii. The agent explicitly preferred inferred events to dF/F, arguing that this follows the paper and avoids carrying slow GCaMP decay into later stimulus epochs.

## 2-b. How is the `neural` data processed?

i. Event traces are converted to `float32`; for every 100 ms trial-bin center the nearest ophys event sample is selected. Planes are retained separately, with no normalization or plane stacking.

ii.
```python
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
```

iii. The trajectory identifies a common grid as necessary to give 31 Hz Scientifica and 11 Hz Multiscope recordings the same bin size, and treats each plane independently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No new cell-level filter is applied. The agent accepts the NWB/SDK's QC-passed cells, verifies at least one cell and matching event/timestamp lengths, and does not discard silent trials.

ii.
```python
if events.shape[1] != len(ophys_times):
    raise ValueError(...)
if events.shape[0] == 0:
    raise ValueError("no QC-passed cells")
```

iii. The agent reasoned that NWB cells already passed Allen ROI/ophys QC. The trajectory says silent trials were deliberately retained to avoid biasing outcome distributions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are anchored to trial `start_time`; centers begin at start + 50 ms and continue every 100 ms until the last complete bin before `stop_time`. Nearest synchronized ophys samples populate those centers.

ii.
```python
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest])
```

iii. The agent says this puts all streams on the synchronized ophys clock while preserving the entire experiment-defined trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at 100 ms. This is nearest-sample resampling rather than aggregation: one source neural sample is chosen per target center.

ii.
```python
BIN_SEC = 0.100
...
"time_bin_size": BIN_SEC * 1000.0,
"resampling": "nearest ophys event sample at each 100 ms center; ..."
```

iii. The agent argued that 10 Hz is finer than the roughly 200 ms effective resolution of inferred events and harmonizes the two acquisition regimes.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from each row of `dataset.stimulus_presentations`, principally `image_name`, `start_time`, and `end_time`.

ii.
```python
presentations = dataset.stimulus_presentations
...
name = stim["image_name"]
on = (centers >= float(stim["start_time"])) & (centers < float(stim["end_time"]))
```

iii. The agent chose presentation intervals so the label represents the image actually displayed during each non-gray interval, including omitted flashes as gray.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Labels start as global class 0 (`gray`). Actual named flashes overwrite bins falling in their display interval; NaN and `omitted` remain gray. Known image names use a fixed global 17-class mapping and unknown names raise an error.

ii.
```python
image = np.zeros(len(centers), dtype=np.uint8)
if pd.notna(name) and name != "omitted":
    if name not in IMAGE_TO_CLASS:
        raise ValueError(...)
    image[on] = IMAGE_TO_CLASS[name]
```

iii. The agent states that global labels keep meanings consistent across sessions and that an omission is continued gray, not an image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus intervals are evaluated at the exact same 100 ms centers used to sample neural events.

ii.
```python
image, change = _stimulus_labels(centers, presentations)
nearest = _nearest_indices(ophys_times, centers)
```

iii. The common center array was chosen to provide direct bin-for-bin alignment across all streams.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the presentation's display-lag-corrected `start_time`.

ii.
```python
if bool(stim.get("is_change", False)):
    idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
```

iii. The agent treated change as an event tied to the actual stimulus presentation, rather than a persistent post-change state.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created, and each true change marks the first target bin whose center is on or after the presentation onset.

ii.
```python
changed = np.zeros(len(centers), dtype=np.uint8)
...
if idx < len(changed):
    changed[idx] = 1
```

iii. The code comment says a change is an event, not a 250 ms state.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated. It is directly encoded into two categories: 0 `no_change` and 1 `change`, based on the boolean `is_change` flag.

ii.
```python
"output_values": [
    ...
    ["no_change", "change"],
]
```

iii. This follows the requested binary output definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The event is placed on the same 100 ms center grid as neural data, at the first center on or after its presentation onset.

ii.
```python
idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
changed[idx] = 1
```

iii. The agent used the common grid to guarantee corresponding output and neural columns.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `dataset.running_speed["speed"]` and its `timestamps`.

ii.
```python
running = dataset.running_speed
running["timestamps"].to_numpy(...)
running["speed"].to_numpy(...)
```

iii. This is the AllenSDK's synchronized running-wheel speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Nonfinite samples are removed, remaining samples are time-sorted, speed is linearly interpolated to all retained trial centers, and those values are discretized using within-session quintile cuts.

ii.
```python
run_values = _interp_valid(..., all_centers)
run_class = _percentile_classes(run_values)
```

iii. The agent wanted aligned categorical behavior labels and reported exactly balanced percentile outputs within sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles are computed separately within each experiment/session over retained trial bins. `searchsorted(..., side="right")` assigns classes 0–4.

ii.
```python
cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
return np.searchsorted(cuts, values, side="right").astype(np.uint8)
```

iii. The trajectory emphasizes balanced classes within sessions; metadata explicitly records the per-session scope.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated directly to the concatenated 100 ms trial centers, then sliced back into each trial in the same order as neural columns.

ii.
```python
run_values = _interp_valid(..., all_centers)
...
run_class[cursor : cursor + n_bins]
```

iii. The common target centers were selected to ensure exact output/neural correspondence.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `dataset.eye_tracking["pupil_area"]` and eye-tracking `timestamps`. Diameter is calculated from the SDK area measure.

ii.
```python
pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The agent says SDK `pupil_area` is based on the maximum ellipse radius after blink/outlier removal, and the monotonic conversion preserves percentile classes.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to diameter, nonfinite samples are omitted, values are time-sorted and linearly interpolated to trial centers, then discretized into within-session quintiles.

ii.
```python
pupil_values = _interp_valid(
    eye["timestamps"].to_numpy(dtype=np.float64), pupil_diameter, all_centers
)
pupil_class = _percentile_classes(pupil_values)
```

iii. The agent avoided inventing values when fewer than two finite samples exist and used the same aligned categorical procedure as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses the session's pupil-diameter 20/40/60/80 percentiles and assigns classes 0–4 with ties placed to the right.

ii.
```python
cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
return np.searchsorted(cuts, values, side="right").astype(np.uint8)
```

iii. The stated goal was balanced within-session decoding classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is linearly interpolated to the same concatenated trial centers, then cursor-sliced into trial arrays matching neural columns.

ii.
```python
pupil_values = _interp_valid(..., all_centers)
...
pupil_class[cursor : cursor + n_bins]
```

iii. The shared 100 ms centers provide direct temporal alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
```

iii. The agent resolved trial outcome as the four genuine mutually exclusive behavioral outcomes, not merely correctness or Go/Catch identity.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code requires exactly one flag, maps its position to integer 0–3, and repeats that static code across every time bin of the trial.

ii.
```python
if flags.sum() != 1:
    raise ValueError(...)
outcome = int(np.flatnonzero(flags)[0])
outcome_row = np.full(n_bins, outcome, dtype=np.uint8)
```

iii. The trajectory identifies these as four mutually exclusive outcomes; validation prevents ambiguous labels from silently entering the dataset.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite behavior samples are dropped before interpolation; interpolation requires two finite points and uses endpoint values outside the sampled range. Missing eye tracking, unusable pupil data, inconsistent neural dimensions, no cells, invalid outcomes, too few trials, and other per-experiment exceptions cause the entire experiment to be skipped and its exact reason recorded. Output is written atomically through a temporary file.

ii.
```python
valid = np.isfinite(times) & np.isfinite(values)
if valid.sum() < 2:
    raise ValueError(...)
...
except Exception as exc:
    skipped.append({"ophys_experiment_id": int(experiment_id), "reason": ...})
...
os.replace(temp_output, args.output)
```

iii. The trajectory says three sessions with wholly unusable pupil data were excluded rather than silently imputed; exact exclusions were retained in metadata.

## 9-a. What are the most time-consuming steps of the code?

i. Loading/parsing each large NWB, materializing full event matrices, and converting all 202 candidate experiments dominate runtime; serialization of the 2.53 GiB result and full validation/training are also expensive.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
events = np.vstack(dataset.events["events"].to_numpy())
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Progress updates in the trajectory span the long full conversion, while the later decoder run spent substantial time loading and initializing 199 session-specific projections.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-row loop and especially the nested loop over overlapping stimulus presentations could potentially be vectorized or interval-indexed. Experiment loading must largely remain iterative because each NWB is independent.

ii.
```python
for _, row in trials.iterrows():
    ...
for _, stim in overlap.iterrows():
    ...
```

iii. The trajectory contains no explicit vectorization discussion; this assessment follows from the implemented pandas row iteration. I/O and full-matrix loading likely dominate any savings.

## 9-c. What processing does the code repeat multiple times?

i. For every experiment it repeats NWB initialization, event stacking, metadata extraction, finite filtering/sorting/interpolation, trial-center construction, nearest-index searches, and stimulus-overlap scans. Within an experiment, `_stimulus_labels` repeatedly filters the full presentations table once per trial.

ii.
```python
for number, experiment_id in enumerate(experiments.index, start=1):
    converted.append(convert_experiment(int(experiment_id)))
...
overlap = presentations[(...) & (...)]
```

iii. No explicit justification was given for repetition; the per-experiment structure keeps memory bounded and accommodates different timestamps and metadata.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It constructs rich `session_info`, skipped-session records, and empty input arrays that the decoder does not use for prediction. It also repeatedly runs `gc.collect()`, and reads metadata fields used only for provenance. These are small relative to neural data and useful for auditability.

ii.
```python
input_trials.append(np.empty((0, n_bins), dtype=np.float32))
...
"session_info": {"behavior_session_id": ..., "indicator": ..., ...}
...
gc.collect()
```

iii. The trajectory emphasizes validation and explicit provenance for exclusions; it does not identify these operations as waste, so their retention appears intentional despite no downstream decoder use.
