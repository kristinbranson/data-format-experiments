# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers local `behavior_ophys_experiment_*.nwb` files and joins their parsed experiment IDs to `ophys_experiment_table.csv`. It reads NWB arrays directly with `h5py`, retains active (non-passive) experiments, and excludes experiments without pupil tracking.

ii.
```python
paths = {experiment_id(path): path for path in nwb_dir.glob("*.nwb")}
table = pd.read_csv(table_path).set_index("ophys_experiment_id")
supplied = table.loc[sorted(paths)].copy()
active = supplied[~supplied["passive"].astype(bool)].copy()
```

iii. The trajectory says direct HDF5 access avoids materializing irrelevant image templates and ROI masks. It excludes passive replay because those recordings do not contain an animal performing Go/Catch trials, and excludes missing-eye experiments rather than fabricating a required pupil output.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` values among selected experiments; each experiment receives the corresponding subject index.

ii.
```python
subjects = sorted({str(int(x)) for x in selected["mouse_id"]})
subject_to_index = {name: i for i, name in enumerate(subjects)}
subject_idx.append(subject_to_index[str(int(row["mouse_id"]))])
```

iii. The agent relies on `mouse_id` as the dataset's animal identifier and uses a deterministic sorted mapping.

## 1-c. How are the data split into sessions?

i. Each behavior ophys experiment/NWB (one imaging plane) is treated as one decoder session; experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for number, (exp_id, row) in enumerate(selected.iterrows(), start=1):
    neural, inputs, outputs, regions, info = convert_experiment(
        paths[int(exp_id)], row, image_to_index, region_to_index
    )
    neural_sessions.append(neural)
```

iii. The agent states that an experiment is the AllenSDK unit owning a simultaneously sampled population and that the paper performs plane-wise decoding.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`. For each retained row, the full `[start_time, stop_time)` duration is divided into complete 100 ms bins; fractional time at the end is dropped.

ii.
```python
for row in np.flatnonzero(keep):
    start = float(trials["start_time"][row])
    stop = float(trials["stop_time"][row])
    n_bins = int(np.floor((stop - start) / BIN_SECONDS + 1e-9))
    edges = start + np.arange(n_bins + 1) * BIN_SECONDS
```

iii. The agent chose full valid trial intervals so pre-change and post-change activity and time-varying targets are retained, while establishing a common bin size across recordings.

## 1-e. How are trials filtered based on quality controls?

i. It retains trials explicitly labeled Go or Catch and rejects aborted and auto-rewarded trials. Trials shorter than one bin are skipped; experiments with fewer than two retained grids error out. No engagement/performance threshold is imposed.

ii.
```python
keep = ((trials["go"].astype(bool) | trials["catch"].astype(bool))
        & ~trials["aborted"].astype(bool)
        & ~trials["auto_rewarded"].astype(bool))
if len(grids) < 2:
    raise ValueError(f"{path.name}: fewer than two retained trials")
```

iii. This is justified as the requested Go/Catch selection and explicit exclusion of aborted and free-reward trials, without activity- or performance-based selection bias.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `processing/ophys/event_detection/data`, paired with the event-detection ophys timestamps.

ii.
```python
event_group = ophys["event_detection"]
ophys_timestamps = read_array(event_group, "timestamps", np.float64)
events = read_array(event_group, "data", np.float32)
```

iii. The agent says detected calcium-event magnitude is the neural representation used in the paper and deliberately rejects dF/F and display-oriented filtered events.

## 2-b. How is the `neural` data processed?

i. Invalid ROIs are removed, event magnitudes are cumulatively summed in place, and timestamp-selected event values are summed into each 100 ms trial bin. No normalization is applied, and planes are kept separate.

ii.
```python
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
np.cumsum(events, axis=0, dtype=np.float32, out=events)
neural = cumulative_bin_sums(events, ophys_timestamps, edges)
```

iii. Cumulative sums make arbitrary timestamp bins efficient and avoid rereading or copying the large full-session event array.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi == True` are included; experiments with no remaining neurons are rejected.

ii.
```python
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = events[:, valid_roi]
if events.shape[1] == 0:
    raise ValueError(f"{path.name}: no valid neurons")
```

iii. The agent notes that released experiments/ROIs passed Allen QC but applies the explicit flag defensively.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial grid is anchored at trial `start_time`. Event samples are assigned using synchronized ophys timestamps and bin edges through `searchsorted`.

ii.
```python
edges = start + np.arange(n_bins + 1) * BIN_SECONDS
indices = np.searchsorted(ophys_timestamps, edges, side="left")
```

iii. The agent emphasizes timestamp alignment rather than frame number or nominal rate because supplied recordings have different native frame rates.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 100 ms. Native event frames are rebinned by summing event magnitudes in each bin.

ii.
```python
BIN_SECONDS = 0.100
"time_bin_size": BIN_SECONDS * 1000.0
```

iii. The agent chose a shared 100 ms grid to satisfy the common-bin requirement across recordings while retaining timestamp-defined alignment.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the task stimulus-presentation table's `image_name`, `start_time`, `stop_time`, and `omitted` fields.

ii.
```python
stim_start = read_array(stim, "start_time", np.float64)
stim_stop = read_array(stim, "stop_time", np.float64)
stim_names = decode_strings(read_array(stim, "image_name"))
omitted = read_array(stim, "omitted").astype(bool)
```

iii. The agent uses actual presentation intervals so identity represents the image displayed during non-gray periods, rather than assuming one image continuously occupies a trial epoch.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global categorical mapping is built with `gray` first and sorted non-omitted image names thereafter. Each bin starts as gray and is overwritten when its center falls inside a non-omitted presentation.

ii.
```python
return ["gray", *sorted(names)]
image_identity = np.full(n_time, gray_index, dtype=np.int16)
shown = ((centers >= stim_start[presentation])
         & (centers < stim_stop[presentation]))
image_identity[shown] = image_to_index[stim_names[presentation]]
```

iii. The explicit gray class covers inter-stimulus gray screens and omissions, while a global deterministic mapping keeps labels consistent.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the centers of exactly the same 100 ms bins used for the trial neural matrix.

ii.
```python
centers = edges[:-1] + BIN_SECONDS / 2.0
shown = ((centers >= stim_start[presentation])
         & (centers < stim_stop[presentation]))
```

iii. All streams use the synchronized NWB clock and common trial grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It comes from each stimulus presentation's `is_change` and `start_time` fields.

ii.
```python
is_change_raw = read_array(stim, "is_change", np.float64)
is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)
```

iii. The presentation-level true-change flag distinguishes real changes from catch/sham events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The output is initialized to zero, and each true change sets one bin: the first bin center at or after its onset.

ii.
```python
image_change = np.zeros(n_time, dtype=np.int16)
change_bin = int(np.searchsorted(centers, stim_start[presentation], side="left"))
if change_bin < n_time:
    image_change[change_bin] = 1
```

iii. The agent interprets “right after” as a transient one-bin event rather than a persistent post-change state.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is learned: the NWB `is_change` boolean directly yields categories 0 (“no change”) and 1 (“change”), with non-finite values treated as false.

ii.
```python
is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)
"output_values": [..., ["no change", "change"], ...]
```

iii. The source already supplies the categorical change annotation.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change onset is located among the common 100 ms bin centers used by the neural data.

ii.
```python
change_bin = int(np.searchsorted(centers, stim_start[presentation], side="left"))
```

iii. This uses synchronized timestamps and the same trial grid as neural event sums.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `processing/running/speed/data` and its timestamps.

ii.
```python
running_group = nwb["processing/running/speed"]
running_t = read_array(running_group, "timestamps", np.float64)
running_v = read_array(running_group, "data", np.float64)
```

iii. This is the NWB/SDK processed running-speed stream on the synchronized clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite speed samples are linearly interpolated at bin centers, then classified into empirical quintiles computed separately within each experiment over all retained trial bins.

ii.
```python
running_aligned = interpolate_finite(running_t, running_v, all_centers, "running-speed")
running_bins = quintile(running_aligned)
```

iii. Within-experiment quintiles were chosen to remove between-animal/recording scale differences and balance classes in the data decoded for that experiment.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles define integer labels 0–4; values equal to an edge enter the higher bin.

ii.
```python
edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. This implements the requested five equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly at the common trial-bin centers and sliced per trial in the same order.

ii.
```python
all_centers = np.concatenate([centers for _, _, centers in grids])
output[2] = running_bins[offset:offset + n_time]
```

iii. Both sources share synchronized NWB time, so target-time interpolation aligns them.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses SDK-processed pupil `area` from `acquisition/EyeTracking/pupil_tracking` and timestamps from `eye_tracking`.

ii.
```python
eye_t = read_array(nwb["acquisition/EyeTracking/eye_tracking"], "timestamps", np.float64)
pupil_area = read_array(pupil_group, "area", np.float64)
```

iii. The agent regards the area field as already outlier/blink filtered, with affected frames encoded as NaN.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Nonnegative area is converted to equivalent circular diameter, finite samples are linearly interpolated at bin centers (including across NaN gaps), and values are quintiled within experiment.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
pupil_aligned = interpolate_finite(eye_t, pupil_diameter, all_centers, "pupil-diameter")
pupil_bins = quintile(pupil_aligned)
```

iii. Equivalent diameter is monotonic in area, so percentile labels are preserved; interpolation fills short blink gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The same within-experiment empirical 20/40/60/80 percentile thresholds produce labels 0–4.

ii.
```python
pupil_bins = quintile(pupil_aligned)
```

iii. This supplies the requested five equal percentile categories while controlling camera/animal scale differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Diameter is interpolated from eye timestamps to the common neural-bin centers, concatenated across retained trials, and sliced back in trial order.

ii.
```python
pupil_aligned = interpolate_finite(eye_t, pupil_diameter, all_centers, "pupil-diameter")
output[3] = pupil_bins[offset:offset + n_time]
```

iii. Hardware-synchronized NWB timestamps permit direct interpolation to the neural grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
outcomes = np.column_stack(
    [trials[name].astype(bool) for name in OUTCOME_COLUMNS]
)
```

iii. These are the canonical mutually exclusive outcomes; the code validates exactly one is true for every retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The index of the true outcome column becomes category 0–3 and is repeated across every time bin of that trial.

ii.
```python
outcome = int(np.flatnonzero(outcomes[trial_row])[0])
output[4].fill(outcome)
```

iii. Repetition lets the static target share one rectangular output matrix with time-varying targets.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye-tracking experiments are excluded; non-finite behavioral samples are ignored during interpolation, with nearest finite endpoints used outside range; pupil area is clipped nonnegative; non-finite change flags are false. The converter raises errors for missing metadata, ambiguous stimulus tables, fewer than two finite behavioral samples, fewer than two trials, no valid neurons, or non-unique outcomes. It writes via a temporary file and atomic replacement.

ii.
```python
valid = np.isfinite(timestamps) & np.isfinite(values)
if np.count_nonzero(valid) < 2:
    raise ValueError(...)
return np.interp(targets, timestamps[valid], values[valid])
temporary.replace(output_path)
```

iii. The trajectory says excluding absent required signals is preferable to inventing them; finite-sample interpolation handles blink gaps. Strict checks prevent silently malformed data.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive work is reading each NWB's large event dataset, cumulatively summing it, materializing all trial neural matrices, and serializing the 2.55 GiB result. The full conversion processed 199 experiments and roughly 51,000 trials.

ii.
```python
events = read_array(event_group, "data", np.float32)
np.cumsum(events, axis=0, dtype=np.float32, out=events)
neural = cumulative_bin_sums(events, ophys_timestamps, edges)
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory anticipated a large output because every eligible trial and neuron is retained and reported progress during the full conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer experiment loop must read separate files, but trial-grid creation, per-trial bin extraction, and the nested per-trial/per-overlapping-presentation image labeling could potentially be batched. Image-name decoding and eye-availability discovery also use Python loops.

ii.
```python
for trial_row, edges, centers in grids:
    ...
    for presentation in overlaps:
        ...
```

iii. The agent vectorized the costly event aggregation within each trial via cumulative sums, favoring straightforward loops for variable-length trials and presentation overlaps.

## 9-c. What processing does the code repeat multiple times?

i. Every selected NWB is opened once during eye-availability discovery, again while collecting image names, and again for conversion. Stimulus-group discovery and image-name reads are consequently repeated; trial-specific `searchsorted` and overlap scans are also repeated.

ii.
```python
for exp_id in active.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb: ...
for exp_id in selected.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb: ...
```

iii. The passes keep discovery, global label construction, and conversion simple and avoid retaining full NWB contents in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It gathers and stores extensive session metadata not required by the decoder, scans all image presentations once to build labels and again to convert, and computes/stores empty input arrays for every trial. The event array is transformed to cumulative sums and then discarded after trial matrices are built.

ii.
```python
input_trials.append(np.empty((0, n_time), dtype=np.float32))
info = {"behavior_session_id": ..., "experience_level": ..., "imaging_depth_um": ...}
```

iii. Empty inputs are required by the target schema, and metadata improves provenance. The cumulative representation is an intentional temporary optimization rather than a downstream feature.
