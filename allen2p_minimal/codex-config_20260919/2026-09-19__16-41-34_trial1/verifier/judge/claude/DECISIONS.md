# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK `VisualBehaviorOphysProjectCache`. It reads the locally supplied release directory directly: it globs every
`behavior_ophys_experiment_*.nwb` under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (284 files), parses the experiment id out of the filename, and joins that id list against `project_metadata/ophys_experiment_table.csv`. Each NWB is then opened with raw `h5py` and specific HDF5 paths are read (`intervals/trials`, the natural-image `*_presentations` interval table, `processing/ophys/event_detection`, `processing/running/speed`, `acquisition/EyeTracking/pupil_tracking`). Two inclusion filters are applied at the table level:

* **Passive sessions are dropped** (`~table["passive"]`), leaving 202 of 284 experiments.
* **No `project_code` filter is applied**, so both `VisualBehavior` (168 active sessions, 37 mice) and `VisualBehaviorMultiscope` (6 active sessions, 1 mouse) experiments are kept.

A first pass over one representative NWB per session applies session-level exclusions and builds the global image vocabulary; a second pass re-opens the NWBs and does the actual extraction. Final dataset: 171 sessions, 38 mice, 43,975 trials, 29,168 neurons.

ii.
```python
files = {
    _experiment_id(path): path
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
}
if not files:
    raise FileNotFoundError(f"No experiment NWBs found under {experiment_dir}")

table = pd.read_csv(metadata_path)
table = table[table["ophys_experiment_id"].isin(files)].copy()
# Passive sessions do not contain the requested Go/Catch trial population.
table = table[~table["passive"].astype(bool)].copy()
table.sort_values(["ophys_session_id", "ophys_experiment_id"], inplace=True)
```
(`/app/convert_data.py:170-182`)

iii. From the trajectory (steps 15–21) the AI first loaded a file through `BehaviorOphysExperiment.from_nwb()`, inspected `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`, `events` and `dff_traces`, then dumped the raw HDF5 layout and switched to `h5py` — i.e. it verified that the raw NWB fields are the same objects the SDK exposes before bypassing the SDK (motivated by the ~247 GB of NWBs and the need for lazy, partial reads). Step 10: *"The local bundle contains 284 quality-controlled experiment NWBs (about 247 GB) plus the project metadata tables."* The passive exclusion is stated in the code comment as *"Passive sessions do not contain the requested Go/Catch trial population"*; the paper's methods independently support the exclusion (`methods.txt`: *"Imaging was also performed during passive viewing of the same stimulus, which was not analyzed here."*). No justification is given anywhere for keeping the `VisualBehaviorMultiscope` experiments; the AI did tabulate the data by `project_code` (steps 12–13) and simply chose to keep everything that was shipped.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values of the retained (active) experiments, cast to `str`, sorted lexicographically. Each session's `subject_idx` is the index of that session's mouse in the list. 38 subjects result.

ii.
```python
subjects = sorted({str(int(g.iloc[0]["mouse_id"])) for _, g in grouped})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
mouse_id = str(int(experiments.iloc[0]["mouse_id"]))
subject_idx.append(subject_to_idx[mouse_id])
```
(`/app/convert_data.py:219-220, 294-296`)

iii. `mouse_id` is the canonical animal identifier in the Allen experiment table; the AI takes it straight from the metadata CSV rather than re-deriving it. The set is built only from sessions that survive exclusion, so no empty subject entries appear.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. The experiment table is grouped by `ophys_session_id` (sorted), and **all imaging planes recorded simultaneously in that session are concatenated along the neuron axis** into one session entry. Per-session behavioural streams (trials, presentations, running, pupil) are read from the *first* experiment of the group (they are session-level and identical across planes), while neural data are read **plane by plane, each using that plane's own `event_detection/timestamps`**. Session-level metadata (experiment ids, behavior_session_id, session_type, experience_level, neurons per plane, quintile edges) is recorded in `metadata['session_info']`.

ii.
```python
for session_id, experiments in table.groupby("ophys_session_id", sort=True):
    representative = files[int(experiments.iloc[0]["ophys_experiment_id"])]
...
for experiment_id, region in zip(experiment_ids, experiments["targeted_structure"].astype(str)):
    with h5py.File(files[experiment_id], "r") as nwb:
        event_detection = nwb["processing/ophys/event_detection"]
        timestamps = event_detection["timestamps"][:]
        aggregated = _interval_sums(timestamps, event_detection["data"], starts)
    plane_activity.append(aggregated)
...
activity = np.concatenate(plane_activity, axis=1)
```
(`/app/convert_data.py:185-186, 257-267`)

iii. Metadata field `session_grouping`: *"Experiments sharing an ophys_session_id are simultaneous planes and are concatenated as neurons in one recording session."* Trajectory step 33: *"Simultaneous multi-plane sessions are being merged by `ophys_session_id`."* Because each plane's aggregation uses its own timestamp vector, the merge is valid even for Multiscope sessions where planes are sampled at different times.

## 1-d. How are the data split into trials?

i. Trials are the NWB `intervals/trials` rows kept by the task filter (`(go | catch) & ~aborted & ~auto_rewarded`). A trial's **timepoints are the image-presentation intervals belonging to it**: every row of the natural-image presentation table whose `trials_id` equals that trial's `id` and whose `active` flag is True, sorted by `start_time`. Each such presentation contributes one 750 ms time bin. Trials are therefore variable length (10–17 bins, mean 11.6 bins ≈ 8.7 s), which closely tracks the trials table's own `trial_length` (~8.5 s). A trial with zero active presentations raises rather than being silently dropped.

ii.
```python
trials = nwb["intervals/trials"]
keep = (
    (trials["go"][:] | trials["catch"][:])
    & ~trials["aborted"][:]
    & ~trials["auto_rewarded"][:]
)
trial_ids = trials["id"][:][keep].astype(np.int64)
...
for trial_id in trial_ids:
    idx = np.flatnonzero((source_trial_ids == trial_id) & active)
    if len(idx) == 0:
        raise RuntimeError(f"Retained trial {trial_id} has no active presentations")
    idx = idx[np.argsort(starts_source[idx])]
    row_groups.append(np.arange(cursor, cursor + len(idx), dtype=np.int64))
    cursor += len(idx)
```
(`/app/convert_data.py:52-63, 85-93`)

iii. Metadata `trial_filter`: *"Go and Catch trials only; aborted and auto-rewarded trials excluded"*; `trial_segmentation`: *"Active stimulus presentations carrying each retained NWB trials_id"*. Trajectory step 10: *"retain the task's native Go/Catch trial definitions while removing aborted and auto-rewarded trials"*. Step 26 explains the presentation-level segmentation: *"The reference paper's actual analysis unit is the 750 ms image-presentation interval (250 ms image plus 500 ms gray) ... each trial becomes a variable-length sequence of flashes ... This both matches the published processing and avoids inventing a higher-resolution label during the gray portion."* The `active` flag restricts to the behaving block, excluding any passive replay / movie / spontaneous blocks.

## 1-e. How are trials filtered based on quality controls?

i. No trial is dropped for data-quality reasons; filtering is at the **task level** (go/catch, not aborted, not auto-rewarded) and at the **session level**:

* session excluded if `acquisition/EyeTracking/pupil_tracking/width` is absent (3 sessions: 795625712, 805989030, 832881662);
* session excluded if fewer than 2 finite pupil samples;
* session excluded if fewer than 2 eligible go/catch trials (the format's minimum-2-trials rule).

Every exclusion is recorded with its reason in `metadata['excluded_sessions']`. Two invariants are asserted rather than repaired: exactly one of hit/miss/false_alarm/correct_reject per retained trial, and at least one ophys timestamp per presentation interval on the reference plane.

ii.
```python
if pupil_path not in nwb:
    excluded.append({"ophys_session_id": int(session_id),
                     "reason": "missing pupil tracking required by decoder output"})
    continue
if np.isfinite(nwb[pupil_path][:]).sum() < 2:
    excluded.append({... "reason": "insufficient finite pupil diameter samples"})
    continue
trial_ids, _ = _eligible_trial_ids(nwb)
if len(trial_ids) < 2:
    excluded.append({... "reason": "fewer than two eligible Go/Catch trials"})
    continue
```
(`/app/convert_data.py:188-208`)
```python
if not np.all(outcomes.sum(axis=1) == 1):
    raise RuntimeError("Every retained trial must have exactly one trial outcome")
```
(`/app/convert_data.py:60-61`)

iii. The exclusions are driven strictly by what the requested decoder outputs need: a session with no eye tracking cannot supply the pupil-diameter target, and a session with <2 trials cannot be split into train/validation. The AI reports this in trajectory step 33: *"Three active sessions are being excluded because their NWBs have no eye-tracking stream, which makes the required pupil-diameter target undefined; all retained sessions have well over the two-trial minimum."* Beyond that the AI relies on the Allen pipeline's own QC — step 10 calls the shipped NWBs *"284 quality-controlled experiment NWBs"*.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the AllenSDK **detected (deconvolved) calcium events**, i.e. per-cell event magnitude time series, with `processing/ophys/event_detection/timestamps` as the ophys clock. dF/F traces are deliberately **not** used.

ii.
```python
with h5py.File(files[experiment_id], "r") as nwb:
    event_detection = nwb["processing/ophys/event_detection"]
    timestamps = event_detection["timestamps"][:]
    aggregated = _interval_sums(timestamps, event_detection["data"], starts)
```
(`/app/convert_data.py:258-262`)
Metadata: `"neural_signal": "AllenSDK inferred calcium-event magnitude, summed per interval"`.

iii. Trajectory step 10: *"A key modeling choice is emerging from the references: use the SDK-provided inferred calcium events—not raw ΔF/F."* This follows `methods.txt` verbatim: *"For all analysis of neural data we used the detected calcium events as described in Garrett et al. ... This process produces, for each cell, a set of calcium events each with a time and magnitude"*, and *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."*

## 2-b. How is the `neural` data processed?

i. For each plane, the event magnitudes are **summed over the half-open 750 ms window `[flash_start, flash_start + 0.750)`**, evaluated against that plane's own ophys timestamps. The sum is computed with a cumulative-sum (prefix-sum) trick so the whole session is aggregated in one vectorised pass; only the contiguous slice of frames spanning the first to the last retained presentation is read from disk. Planes are then concatenated along the neuron axis, and the result is transposed per trial to `(n_neurons, n_timepoints)` in float32. No normalisation, smoothing, z-scoring or baseline subtraction is applied.

ii.
```python
first = int(np.searchsorted(timestamps, starts[0], side="left"))
last = int(np.searchsorted(timestamps, starts[-1] + BIN_SECONDS, side="left"))
event_data = np.empty((last - first, values.shape[1]), dtype=np.float32)
values.read_direct(event_data, source_sel=np.s_[first:last, :])

# Prefix sums make every half-open [start, start+750 ms) aggregation exact
# with respect to the plane's own ophys timestamps.
prefix = np.empty((len(event_data) + 1, event_data.shape[1]), dtype=np.float32)
prefix[0] = 0.0
np.cumsum(event_data, axis=0, dtype=np.float32, out=prefix[1:])
left = np.searchsorted(timestamps, starts, side="left") - first
right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
return prefix[right] - prefix[left]
...
activity = np.concatenate(plane_activity, axis=1)
...
session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
```
(`/app/convert_data.py:101-118, 269, 277`)

iii. Summing event magnitudes inside the image-presentation interval is the natural aggregation for a point-process-like signal and is the unit the paper analyses (`methods.txt`: *"By image presentation interval we refer to the 750 ms interval beginning with each image presentation"*). The prefix-sum comment in the code states the intent: exactness of the half-open aggregation against each plane's own timestamps. The AI accepted the consequence that sparse event traces in low-cell-count planes yield some all-zero trials — trajectory step 37: *"The validator's warnings are limited to 1,677 individual trials with no detected calcium events—expected for sparse deconvolved event traces, especially low-cell-count planes, and not a formatting failure."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control is applied.** Every ROI present in `event_detection` is kept (sessions range from 6 to 666 neurons, 29,168 total). There is no minimum-cell-count, minimum-event-rate, SNR or valid-ROI filter, and no removal of the 1,677 all-zero trials.

ii. No code — the absence of filtering is the decision. The nearest thing is the region bookkeeping that keeps every ROI:
```python
region_codes.append(
    np.full(aggregated.shape[1], region_to_idx[region], dtype=np.int64)
)
```
(`/app/convert_data.py:263-265`)

iii. The shipped NWBs are already the Allen pipeline's QC-passed output (trajectory step 10: *"284 quality-controlled experiment NWBs"*), and `methods.txt` documents that segmentation, crosstalk removal and container-level QC were performed upstream. Step 37 explicitly argues the zero-event trials are a property of sparse deconvolved traces rather than a defect to filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is **per time bin, to the onset of each image presentation**, not to a single per-trial event. Every bin `k` of a trial covers `[start_time_k, start_time_k + 0.750)` where `start_time_k` is the stimulus-presentation start time from the NWB presentation table. Because consecutive flashes in the change-detection block are spaced 750.6 ms apart, the bins tile the trial contiguously. All streams — events, running, pupil, labels — are cut on the **same** `starts` vector, and every stream is referred to ophys timestamps before being cut, which satisfies the instruction "Temporally align based on ophys timestamp".

ii.
```python
left = np.searchsorted(timestamps, starts, side="left") - first
right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
```
(`/app/convert_data.py:115-116`)
```python
"temporal_alignment_event": (
    "Start of each 750 ms image-presentation interval; all streams are "
    "aligned through ophys timestamps."
),
"off_start": None,
"off_end": None,
```
(`/app/convert_data.py:344-350`)

iii. Trajectory step 26: *"I'm adopting that native task cadence ... calcium-event magnitudes are aggregated by ophys timestamps within each interval, and stimulus/change labels come directly from the SDK's presentation table."* Using each plane's own timestamps (rather than the first plane's) keeps the alignment correct for multi-plane sessions. `off_start`/`off_end` are set to `None` because the alignment event repeats every bin rather than defining a fixed trial window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — aggressive rebinning.** Native ophys sampling (≈31 Hz for the single-plane Scientifica sessions, ≈11 Hz per plane for Multiscope) is rebinned to **one 750 ms bin per image presentation**. `metadata['time_bin_size'] = 750.0` ms, identical for every trial and every session (the true flash cadence is 750.6 ms; the declared value is the nominal 750 ms). Trials are 10–17 bins long (mean 11.6). This collapses each flash + grey period into a single sample, discarding within-flash response dynamics.

ii.
```python
BIN_SECONDS = 0.750
...
right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
...
"time_bin_size": BIN_SECONDS * 1000.0,
```
(`/app/convert_data.py:22, 116, 343`)

iii. Trajectory step 26: *"The reference paper's actual analysis unit is the 750 ms image-presentation interval (250 ms image plus 500 ms gray), and omissions are treated as their own 750 ms intervals. I'm adopting that native task cadence ... This both matches the published processing and avoids inventing a higher-resolution label during the gray portion."* This also makes the bin size trivially identical across the 31 Hz and 11 Hz rigs, satisfying the format's "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The `image_name` column of the natural-image stimulus-presentation interval table (the table is located generically by requiring the columns `image_name`, `omitted` and `trials_id`, so it works for image sets A/B/G/H). Omitted flashes carry `image_name == "omitted"` and are retained as their own identity category — 17 categories total (16 natural images + `omitted`, the latter 3.4% of bins).

ii.
```python
def _presentation_group(nwb: h5py.File) -> h5py.Group:
    """Return the natural-image presentation table, independent of image set."""
    matches = [
        group for group in nwb["intervals"].values()
        if isinstance(group, h5py.Group)
        and "image_name" in group and "omitted" in group and "trials_id" in group
    ]
...
names_source = np.asarray(_decode_strings(presentations["image_name"][:]), dtype=object)
```
(`/app/convert_data.py:34-47, 78`)

iii. Code comment: *"Natural image identifiers are stable strings (imXXX); omissions are a genuine task interval and get their own category after the image classes."* Metadata `image_omissions`: *"Omissions are labeled as their own image-identity category and occupy the same 750 ms interval used in the reference analysis"* — matching `methods.txt`: *"For image omissions we used the 750 ms following the time of the omission, when the image should have been presented."* Taking the identity from the presentation table (rather than reconstructing it from the trials table's `initial_image_name`/`change_image_name`) means the label is the stimulus that was actually shown in each bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Names are mapped to integer codes through a **global** vocabulary built in the pre-scan across all retained sessions. Real image names are sorted alphabetically and assigned codes 0..15; `omitted`, if present, is appended last (code 16) so the "no image" class never interleaves with real images. The resulting per-bin code vector is stored as `int16`, and the human-readable names are exported in `output_values[0]`.

ii.
```python
real_images = sorted(name for name in image_names if name != "omitted")
image_values = real_images + (["omitted"] if "omitted" in image_names else [])
image_to_code = {name: idx for idx, name in enumerate(image_values)}
...
image_codes = np.asarray([image_to_code[name] for name in names], dtype=np.int16)
```
(`/app/convert_data.py:215-217, 270`)

iii. A single global, deterministic mapping is required because the decoder's read-out head is shared across sessions and mice see different image sets (A and B). Sorting makes the mapping reproducible; putting `omitted` last keeps the image classes contiguous.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Perfectly, by construction: `names` is indexed by the same presentation rows (`row_groups`) used to slice `activity`, so output row 0 of a trial is exactly the label of the interval whose event sums form that trial's neural column.

ii.
```python
for rows, outcome in zip(row_groups, trial_outcomes):
    timepoints = len(rows)
    session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
    session_output.append(np.vstack([
        image_codes[rows],
        ...
    ]).astype(np.int16, copy=False))
```
(`/app/convert_data.py:275-284`)

iii. Because the presentation table is the single source of both the bin boundaries (`starts`) and the labels, no separate interpolation or searchsorted step is needed for the stimulus labels — alignment cannot drift.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean column of the same stimulus-presentation table. This is True on the flash at which the image identity actually changed, and False on sham (catch-trial) changes and on omissions.

ii.
```python
changes_source = presentations["is_change"][:].astype(bool)
...
return (
    np.concatenate(starts).astype(np.float64),
    row_groups,
    np.concatenate(names),
    np.concatenate(changes).astype(np.int16),
)
```
(`/app/convert_data.py:79, 95-100`)

iii. `is_change` is the SDK's canonical per-flash change flag; the AI checked it against the SDK object in trajectory step 27 (`add_is_change=True`) before reading it from the raw NWB. Task description: *"predicts image identity, **true image changes**, ..."* — i.e. catch-trial sham changes are intentionally labelled 0. The realised statistics confirm the semantics: `image_change == 1` in 7.50% of bins, which is exactly (fraction of go trials ≈ 0.875) / (mean 11.6 bins per trial).

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none: the boolean flag is cast to `int16` and placed as row 1 of the output matrix. There is no smoothing, no widening of the change marker beyond the single 750 ms interval, and no separate treatment of catch trials (they are already False in `is_change`).

ii.
```python
session_output.append(np.vstack([
    image_codes[rows],
    changes[rows],
    ...
]).astype(np.int16, copy=False))
```
(`/app/convert_data.py:279-285`)

iii. The 750 ms bin already *is* the change event's natural extent (one flash plus its following grey), so the binary indicator needs no extra windowing — this is the same window (`change_time` to `change_time + 0.75`) the expert reference constructs explicitly, obtained for free by the AI's binning.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the source variable is already binary. It is exported with `output_values[1] = ["no_change", "change"]`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ...
]
```
(`/app/convert_data.py:326-333`)

iii. The instruction asks for a binary variable that is "1 right after a change in image identity, otherwise 0"; `is_change` supplies exactly that at the resolution of the chosen bin.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity — `changes[rows]` uses the identical presentation-row index groups as the neural slice, so the 1 falls in exactly the bin whose event sums span the changed flash.

ii.
```python
session_output.append(np.vstack([
    image_codes[rows],
    changes[rows],
    running_codes[rows],
    pupil_codes[rows],
    np.full(timepoints, outcome, dtype=np.int16),
]).astype(np.int16, copy=False))
```
(`/app/convert_data.py:279-285`)

iii. Single-source alignment: the bin boundaries and the change flag come from the same table row, so there is no cross-clock interpolation error for this variable at all.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` with `processing/running/speed/timestamps` — the Allen-processed (filtered) running-wheel speed in cm/s, the same object the SDK exposes as `dataset.running_speed`.

ii.
```python
running_ts = nwb["processing/running/speed/timestamps"][:]
running_raw = nwb["processing/running/speed/data"][:]
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
```
(`/app/convert_data.py:249-252`)

iii. This is the standard, already-processed locomotion stream; the AI confirmed the field via the SDK in trajectory steps 15–19 before switching to the raw HDF5 path.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two stages. (1) The speed trace is linearly interpolated onto the reference plane's ophys timestamps, using only finite (timestamp, value) pairs. (2) The interpolated trace is **averaged over each 750 ms presentation interval** with a prefix-sum, giving one scalar per time bin. Then it is discretised (see 5-c). The code raises if any interval contains no ophys sample.

ii.
```python
def _interval_means(reference_timestamps, values_at_reference, starts):
    prefix = np.empty(len(values_at_reference) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values_at_reference, out=prefix[1:])
    left = np.searchsorted(reference_timestamps, starts, side="left")
    right = np.searchsorted(reference_timestamps, starts + BIN_SECONDS, side="left")
    counts = right - left
    if np.any(counts == 0):
        raise RuntimeError("A presentation interval contains no ophys timestamps")
    return (prefix[right] - prefix[left]) / counts
```
(`/app/convert_data.py:120-135`)

iii. Metadata `behavior_alignment`: *"Allen processed running speed and blink-filtered pupil ellipse width linearly interpolated to ophys timestamps, then averaged per interval."* Averaging (rather than point-sampling) is the right reduction when the target resolution is 750 ms and the source is ~60 Hz, and going through the ophys timebase first keeps the behavioural reduction on exactly the same clock as the neural reduction.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into 5 quintile bins using **within-session** 20th/40th/60th/80th percentiles of the per-interval mean speeds over that session's retained intervals. `np.searchsorted(edges, values, side="right")` yields codes 0–4. The per-session edges are saved in `metadata['session_info'][i]['running_quintile_edges_cm_per_s']`. The resulting global distribution is 20.0%/20.0%/20.0%/20.0%/20.0%.

ii.
```python
def _quintile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discretize values using within-session 20/40/60/80th percentiles."""
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int16)
    return codes, edges
...
running_codes, running_edges = _quintile_codes(running_values)
```
(`/app/convert_data.py:158-163, 254`)

iii. Metadata `discretization`: *"Running speed and pupil diameter use within-session 20th, 40th, 60th, and 80th percentile boundaries over retained intervals."* Within-session boundaries guarantee the requested "five equal percentile bins" hold exactly in every session and remove between-session offsets (e.g. sessions where the mouse barely ran at all, whose saved edges are near zero). The absolute cm/s boundaries are preserved in metadata so the mapping is recoverable.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Via the ophys timebase and the same `starts` vector. The speed is first put on `reference_ts` (the first plane's `event_detection/timestamps`), then reduced over the identical `[start, start+0.750)` windows used for the event sums, then indexed by the same `rows` groups.

ii.
```python
reference_ts = nwb["processing/ophys/event_detection/timestamps"][:]
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
...
running_codes[rows],
```
(`/app/convert_data.py:247-252, 281`)

iii. `methods.txt` documents that all data streams are hardware-synchronised on a single 100 kHz IO board, so linear interpolation between the running clock and the ophys clock is legitimate; routing everything through ophys timestamps before binning makes the neural/behaviour correspondence exact.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` (ellipse-fit pupil width, in pixels) with its own `timestamps`. Blink frames are already `NaN` in this array (verified: 100% of `likely_blink` frames are NaN and 0% of non-blink frames are), and the code drops non-finite samples before interpolating — equivalent to filtering on the SDK's `likely_blink` flag.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil_at_ophys = _interpolate_finite(
    pupil["timestamps"][:], pupil["width"][:], reference_ts
)
```
(`/app/convert_data.py:254-256`)
```python
"""Linearly align a behavioral stream to ophys timestamps.

Pupil samples marked as blinks are NaN in the AllenSDK/NWB.  Interpolation
over finite samples supplies categorical labels without treating blink
frames as real pupil measurements.
"""
finite = np.isfinite(source_timestamps) & np.isfinite(source_values)
```
(`/app/convert_data.py:140-151`)

iii. The docstring states the rationale directly. The AI inspected the eye-tracking group structure and the SDK's `likely_blink`/`pupil_width` implementation in trajectory steps 18–21 before settling on `width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: drop non-finite (blink) samples → linear interpolation onto the reference plane's ophys timestamps (which also bridges blink gaps rather than leaving holes) → prefix-sum mean over each 750 ms presentation interval → quintile coding. Units remain raw pixels.

ii.
```python
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
pupil_codes, pupil_edges = _quintile_codes(pupil_values)
```
(`/app/convert_data.py:256, 255`) and `_interpolate_finite` / `_interval_means` as quoted above.

iii. Interpolating across blinks (instead of emitting a NaN or a sentinel bin) means no time bin is assigned a spurious "smallest pupil" label; combined with the session-level exclusion of sessions with no eye tracking, the pupil output is never fabricated from missing data.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: **within-session** 20/40/60/80th percentile edges over the session's retained interval means, producing codes 0–4 with an exactly 20%/20%/20%/20%/20% global distribution. Edges are saved per session as `pupil_diameter_quintile_edges_pixels`.

ii.
```python
pupil_codes, pupil_edges = _quintile_codes(pupil_values)
...
"pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
```
(`/app/convert_data.py:255, 310`)

iii. Metadata `discretization` (quoted in 5-c). Within-session quintiles are particularly defensible here: pupil width is in uncalibrated camera pixels, so its absolute scale is not comparable across sessions, rigs or head-fixation geometry — session-relative bins are the only interpretation that is stable across the 171 sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same route as running speed — interpolated onto `reference_ts` (ophys clock), reduced over the same `starts` windows, indexed by the same `rows`.

ii.
```python
pupil_at_ophys = _interpolate_finite(pupil["timestamps"][:], pupil["width"][:], reference_ts)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
...
pupil_codes[rows],
```
(`/app/convert_data.py:254-256, 282`)

iii. As with running speed, the hardware-synchronised clocks documented in `methods.txt` make linear interpolation onto the ophys timebase valid, and reusing `starts`/`rows` makes the alignment exact by construction.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, read for the retained go/catch trials only.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcomes = np.column_stack([trials[name][:][keep] for name in OUTCOME_COLUMNS])
if not np.all(outcomes.sum(axis=1) == 1):
    raise RuntimeError("Every retained trial must have exactly one trial outcome")
return trial_ids, outcomes.argmax(axis=1).astype(np.int16)
```
(`/app/convert_data.py:23, 58-63`)

iii. These are the SDK's canonical four-way outcome labels for the change-detection task, described in `methods.txt` (*"yields 'HIT', 'MISS', 'FALSE ALARM', and 'CORRECT REJECTION' trials"*). The AI adds a hard assertion that exactly one is set rather than a silent `'other'` fallback, so a violated assumption fails loudly.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` over the one-hot outcome columns gives a code in 0–3 (`hit`=0, `miss`=1, `false_alarm`=2, `correct_reject`=3), and that scalar is **broadcast across every time bin of the trial** so the output array stays fully time-varying (row 4 of the `(5, T)` matrix). Names are exported in `output_values[4]`. Realised distribution over bins: hit 31.3%, miss 56.1%, false alarm 1.8%, correct reject 10.7%.

ii.
```python
np.full(timepoints, outcome, dtype=np.int16),
...
"output_values": [..., list(OUTCOME_COLUMNS)],
```
(`/app/convert_data.py:283, 333`)

iii. The format spec asks for time-varying outputs "if at all possible"; a static per-trial label broadcast over the trial's bins keeps the output block rectangular and the decoder's read-out uniform across the five variables.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI's posture is **exclude-and-record, or fail loudly** — there is no blanket `try/except`:

* **Missing eye tracking** → whole session excluded, reason recorded in `metadata['excluded_sessions']` (3 sessions).
* **<2 finite pupil samples** → session excluded, reason recorded.
* **<2 eligible trials** → session excluded, reason recorded.
* **Blinks / non-finite behavioural samples** → dropped, then bridged by linear interpolation (`_interpolate_finite`), so no NaN reaches the output and no bin is mislabelled as the lowest quintile.
* **Ambiguous trial outcome, trial with no active presentations, presentation interval with no ophys sample, more than one candidate presentation table** → `RuntimeError`.
* **Atomic write**: the pickle is written to a `.tmp` file and `os.replace`d, so a crash cannot leave a half-written `converted_data.pkl`.
* **Not** handled: trials whose presentation intervals fall outside a *non-reference* plane's imaging window would contribute zeros silently (the `counts == 0` check only guards the reference plane); and the 1,677 all-zero-neural trials are knowingly kept.

ii.
```python
excluded.append({"ophys_session_id": int(session_id),
                 "reason": "missing pupil tracking required by decoder output"})
...
if finite.sum() < 2:
    raise RuntimeError("Behavioral stream has fewer than two finite samples")
...
if np.any(counts == 0):
    raise RuntimeError("A presentation interval contains no ophys timestamps")
...
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(temporary, output_path)
```
(`/app/convert_data.py:190-208, 146-147, 132-133, 379-383`)

iii. Trajectory step 33 gives the reasoning for the only data-driven exclusions: *"Three active sessions are being excluded because their NWBs have no eye-tracking stream, which makes the required pupil-diameter target undefined."* Step 37 defends keeping the zero-event trials. The general principle is that anything that would require *inventing* a target value causes exclusion, while anything that would indicate a broken assumption causes a crash rather than a silently degraded dataset.

## 9-a. What are the most time-consuming steps of the code?

i. Disk I/O on the ~247 GB of NWBs dominates. Specifically: (1) `_interval_sums` reading the contiguous `event_detection/data` slice for every plane of every session (`read_direct` of an `(n_frames, n_cells)` float64→float32 block, up to ~140k × 666); (2) the pre-scan, which opens one NWB per session a *second* time and reads the whole pupil `width` array plus the full trials and presentation tables; (3) `np.isfinite(nwb[pupil_path][:]).sum()` and the full-session `np.interp` of running and pupil onto every ophys timestamp; (4) pickling the 363 MB result. The per-session scan in `_session_presentations` (`np.flatnonzero` per trial) is the largest pure-CPU cost but is small next to I/O.

ii.
```python
event_data = np.empty((last - first, values.shape[1]), dtype=np.float32)
values.read_direct(event_data, source_sel=np.s_[first:last, :])
```
(`/app/convert_data.py:109-110`)

iii. The AI sized the problem up front (trajectory steps 22–25 scanned every session for cell counts, frame counts and pupil availability, and checked `df -h` / `free -h` before committing) and mitigated the dominant cost by never loading a whole plane: `read_direct` reads only the frames between the first and last retained presentation, in float32.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate is the per-trial scan in `_session_presentations`: for each retained trial it runs `np.flatnonzero((source_trial_ids == trial_id) & active)` over the *entire* presentation table (~300 trials × ~4,800 presentations per session). Since the table is already chronological and `trials_id` is contiguous per trial, this could be done in one pass with `np.searchsorted` on a sorted key or `np.unique(..., return_index=True)`. Secondary candidates: the Python `for` over `rows` that builds the per-trial arrays (`session_neural`/`session_output`), and the list comprehension `[image_to_code[name] for name in names]` (replaceable by `np.unique(names, return_inverse=True)` plus a remap). The genuinely heavy work — event aggregation and behavioural interval means — is *already* fully vectorised via prefix sums, which is the main efficiency win of this implementation.

ii.
```python
for trial_id in trial_ids:
    idx = np.flatnonzero((source_trial_ids == trial_id) & active)
    ...
    idx = idx[np.argsort(starts_source[idx])]
```
(`/app/convert_data.py:84-91`)

iii. The code comment acknowledges the loop is written for explicitness rather than speed: *"The NWB table is chronological, but make the assumption explicit."* Since runtime is I/O-bound, the AI prioritised vectorising the array aggregation (where it wrote a prefix-sum implementation) over the bookkeeping loops.

## 9-c. What processing does the code repeat multiple times?

i. The session pre-scan duplicates work later redone in the main loop. For every session, the representative NWB is opened twice, and `_eligible_trial_ids()` and `_session_presentations()` are each called twice — once in the pre-scan (whose outputs are largely discarded: `trial_ids, _ = ...` throws away the outcomes, and `_, _, names, _ = ...` keeps only the image names) and once in the extraction loop. `np.searchsorted(timestamps, starts, ...)` is recomputed inside `_interval_sums` for every plane (unavoidable, since each plane has its own clock) and again inside `_interval_means`.

ii.
```python
# pre-scan
trial_ids, _ = _eligible_trial_ids(nwb)
...
_, _, names, _ = _session_presentations(nwb, trial_ids)
...
# main loop, same session
trial_ids, trial_outcomes = _eligible_trial_ids(nwb)
starts, row_groups, names, changes = _session_presentations(nwb, trial_ids)
```
(`/app/convert_data.py:201-209, 245-246`)

iii. The two-pass structure is deliberate and is what makes the global image vocabulary and the up-front `excluded_sessions` list possible without holding every session in memory. The repeated work touches only small metadata tables (trials and presentations), not the large event arrays, so the cost is a few percent of runtime.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:

* The pre-scan computes full trial outcomes and full `starts`/`row_groups`/`changes` arrays and throws all but `names` away (`/app/convert_data.py:201-209`).
* `np.isfinite(nwb[pupil_path][:]).sum()` reads the entire pupil array purely to test for ≥2 finite samples; the array is read again in the main loop.
* `running_at_ophys` and `pupil_at_ophys` are interpolated onto **every** ophys timestamp of the session (~140k points), although only the bins covered by retained trials are ever used — the intervals outside retained trials are averaged away or never indexed.
* `_interval_sums` reads the whole contiguous frame range from the first to the last retained presentation, including the inter-trial frames that fall in no bin.
* `_interval_means` is called for the full `starts` vector of every stream even for sessions where the per-interval mean is immediately re-expressed as a 5-level code — the continuous values themselves are not exported (only the quintile edges are).
* Rich `session_info` metadata (behavior_session_id, experience_level, per-plane neuron counts, quintile edges) is computed and stored but not consumed by the decoder.

ii.
```python
trial_ids, _ = _eligible_trial_ids(nwb)          # outcomes discarded
...
if np.isfinite(nwb[pupil_path][:]).sum() < 2:    # whole array read for a count
...
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)  # full session
```
(`/app/convert_data.py:196, 201, 251`)

iii. None of this is justified explicitly in the trajectory; it is the price of the two-pass design and of keeping each helper stream-agnostic and simple. The discarded metadata (`session_info`, `excluded_sessions`, quintile edges) is not waste in the scientific sense — it is provenance that makes the exclusions and the discretisation auditable, which the instructions ask for.
