# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK project-cache API. It enumerates the NWB files that are physically present in the local release directory (`data/.../behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb`), reads the flat metadata table `project_metadata/ophys_experiment_table.csv` directly with pandas, and intersects the two. The selection is then restricted to `behavior_type == "active_behavior"` (passive-viewing experiments are dropped) and sorted by `date_of_acquisition, mouse_id, ophys_experiment_id`. **No `project_code` filter is applied**, so both `VisualBehavior` (168 experiments) and `VisualBehaviorMultiscope` (34 experiments) are included — 202 experiments, 38 mice. Each experiment is loaded individually with `BehaviorOphysExperiment.from_nwb_path(..., exclude_invalid_rois=True)`, and per-experiment the AI pulls `events`, `ophys_timestamps`, `running_speed`, `eye_tracking`, `trials` and `stimulus_presentations`. 199 of 202 experiments survive (3 skipped for missing eye tracking).

ii.
```python
def get_available_experiment_ids(data_root: Path):
    experiment_dir = data_root / "behavior_ophys_experiments"
    pattern = re.compile(r"behavior_ophys_experiment_(\d+)\.nwb$")
    experiment_ids = []
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb")):
        match = pattern.match(path.name)
        if match is not None:
            experiment_ids.append(int(match.group(1)))
    return experiment_ids


def load_experiment_table(data_root: Path):
    exp_table = pd.read_csv(data_root / "project_metadata" / "ophys_experiment_table.csv")
    exp_table = exp_table.set_index("ophys_experiment_id", drop=False)
    return exp_table


def select_experiments(exp_table: pd.DataFrame, available_ids, max_sessions=None):
    available_ids = set(available_ids)
    selected = exp_table.loc[exp_table.index.intersection(available_ids)].copy()
    selected = selected[selected["behavior_type"] == "active_behavior"].copy()
    selected = selected.reset_index(drop=True)
    selected = selected.sort_values(
        ["date_of_acquisition", "mouse_id", "ophys_experiment_id"]
    )
```
```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
```

iii. From the trajectory (steps 54–57) and `CONVERSION_NOTES.md`: the AI discovered that "the local dataset is a downloaded subset, not the full release" (284 of the released NWB files), so it deliberately built the pipeline "around the available local NWBs" rather than around the SDK manifest, which lists experiments whose files are absent. It spent steps 34–58 trying to reproduce the cohort counts reported in the paper (`familiar / MESO.1 / VISp+VISl / cre-line` combinations) and could not match them against the local subset, so it fell back to the broadest defensible cohort: "all locally available active-behavior experiments". Passive experiments were excluded because the decoder must predict behavioral trial outcome, which is meaningless when the lick spout is retracted. It used `from_nwb_path` (rather than the cache) because the files are local, and `exclude_invalid_rois=True` because it verified in the SDK source (steps 14–22) that this is the SDK's own ROI quality filter.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the experiments that survive loading and QC, sorted as strings; `subject_idx` maps each output session to its mouse. 38 mice result (37 from `VisualBehavior` + 1 `VisualBehaviorMultiscope` mouse).

ii.
```python
subjects = sorted({session["mouse_id"] for session in sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session["mouse_id"]])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```
with `"mouse_id": str(meta_row["mouse_id"])` taken from the experiment table row.

iii. `mouse_id` is the SDK's canonical animal identifier in `ophys_experiment_table.csv`; the AI verified the counts against the local manifest ("available experiments 284 mice 38 sessions 247", step 56) and reported the same 38 mice in the final artifact.

## 1-c. How are the data split into sessions?

i. **One output "session" = one `ophys_experiment_id` (one imaging plane)**, not one `ophys_session_id`. For the 165 single-plane `VisualBehavior` experiments this is identical to a recording session. For the 34 `VisualBehaviorMultiscope` planes it is not: those 34 output sessions come from only 6 distinct `ophys_session_id`s of a single mouse, so the same behavioral session (same trials, same running/pupil/outcome labels) appears up to ~7 times, paired with different neuron sets. Final dataset: 199 sessions. `ophys_session_id` is retained only as metadata.

ii.
```python
session = {
    "experiment_id": int(experiment_id),
    "mouse_id": str(meta_row["mouse_id"]),
    "brain_region": str(meta_row["targeted_structure"]),
    ...
    "ophys_session_id": int(meta_row["ophys_session_id"]),
    "n_neurons": int(events_matrix.shape[0]),
```
```python
brain_region_idx.append(
    np.full(
        session["n_neurons"],
        brain_region_to_idx[session["brain_region"]],
        dtype=np.int64,
    )
)
```

iii. `CONVERSION_NOTES.md`: "An `ophys_experiment_id` is one imaging plane with one ophys timestamp stream. This keeps neural timestamps native and avoids merging planes with different neuron sets." In the trajectory (step 51) the AI explicitly framed this as the remaining open design choice and wrote a script (step 52) to test whether simultaneously recorded multiscope planes share ophys frame times — **that check crashed with `FileNotFoundError` (step 53) and was never re-run**, so the premise of the rationale was never verified. (Checking it here: planes within multiscope session 951410079 have identical-length timestamp vectors agreeing to ≤ 23 ms, i.e. they could have been merged.)

## 1-d. How are the data split into trials?

i. Trials are the rows of the SDK `dataset.trials` table. Kept trials must satisfy `(go | catch) & ~aborted & ~auto_rewarded & isfinite(change_time)`. The trial window is the SDK's own `start_time` → `stop_time` (variable length), and within that window the trial is represented as the sequence of change-detection-block stimulus presentations whose `start_time` falls inside it. Resulting trials are 10–17 image-presentation intervals long (48,655 trials from 51,075 candidates).

ii.
```python
trials = dataset.trials.copy()
valid_trials = trials[
    (trials["go"] | trials["catch"])
    & (~trials["aborted"])
    & (~trials["auto_rewarded"])
    & np.isfinite(trials["change_time"])
].copy()
```
```python
for trial_id, row in valid_trials.iterrows():
    trial_start = float(row["start_time"])
    trial_stop = float(row["stop_time"])
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
    if len(trial_stim) == 0:
        continue
    stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
    bin_edges = np.concatenate(
        [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
    )
```

iii. The instruction explicitly names the Go/Catch/Aborted/Auto-rewarded trial types, and the AI verified (steps 61–62, 205–211) that these flags exist as boolean columns in the SDK trial table. It initially implemented a fixed change-centred window (`[-0.75 s, +0.75 s)`, step 68) and then deliberately abandoned it: "The Allen trial table confirms the trials are genuinely variable-length and extend well beyond the change itself, so the fixed 2-interval representation is a simplification rather than the native trial definition" (step 208), and "a valid go trial contains a sequence of 750 ms image-presentation intervals from trial start through multiple post-change repeats" (step 211). It therefore switched to the full native trial window, matching "segment each recording session into individual trials based on how they are defined in the experiment".

## 1-e. How are trials filtered based on quality controls?

i. Several layers, in addition to the trial-type filter of 1-d:
- Whole experiment dropped if `len(dataset.eye_tracking) == 0` (3 experiments), if `len(dataset.events) == 0`, if pupil is entirely NaN, if running speed is entirely non-finite, if the change-detection stimulus block is empty, or if fewer than 2 valid/binned trials remain.
- Trial dropped if it contains no stimulus presentations.
- Trial dropped if **any** binned neural, running or pupil value is non-finite (i.e. incomplete behavioural coverage).
- Trial dropped if the binned neural matrix is **all zeros**.
- Trial dropped if the stimulus table's `is_change` flags are inconsistent with the trial type (a go trial without exactly one change flash, or a catch trial with any change flash); 0 such trials occurred.
Net effect: 4.7% of otherwise-valid trials were removed (51,075 → 48,655).

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
...
if len(valid_trials) < 2:
    return None, {"skip_reason": "too_few_valid_trials", ...}
```
```python
if (
    np.any(~np.isfinite(neural_trial))
    or np.any(~np.isfinite(running_trial))
    or np.any(~np.isfinite(pupil_trial))
):
    continue
if np.all(neural_trial == 0):
    continue

omitted_flags = trial_stim["omitted"].fillna(False).to_numpy(dtype=bool)
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
...
if len(session["neural_trials"]) < 2:
    return None, {"skip_reason": "too_few_binned_trials", ...}
```

iii. The all-zero-neural filter was added after the validator run exposed them: "it exposed a cluster of completely silent trials after the windowing step. Those are not useful for decoding, so I'm filtering out all-zero neural trials" (step 86). The non-finite filter avoids inventing behavioural labels for timepoints where running/pupil were not recorded. The ≥2-trial rule comes straight from the format spec ("There needs to be at least two trials within each session"). The `is_change` consistency check was introduced as a cross-validation between the two independent SDK tables (trials vs stimulus presentations) and is reported in the notes as "change-flag mismatches = 0". The AI also documents omission sanity checks against the whitepaper's statement that the change image and the image preceding it are never omitted (observed 0 in both cases).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dataset.events["events"]` — the AllenSDK **inferred/deconvolved calcium event traces** (not `dff_traces`), stacked into an (n_neurons, n_ophys_frames) matrix, with invalid ROIs already excluded by the loader. `dataset.ophys_timestamps` provides the time base.

ii.
```python
events_matrix = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32)
ophys_timestamps = dataset.ophys_timestamps.astype(np.float64)
```
and in the metadata: `"neural_signal": "AllenSDK ophys inferred events"`.

iii. `CONVERSION_NOTES.md`: "This follows the paper's use of inferred/discrete calcium events rather than raw fluorescence." The AI read the SDK tutorial (step 10), which states that events "are computed on unmixed dff traces … The magnitude of events approximates the firing rate of neurons with the resolution of about 200 ms. The biggest advantage of using events over dff traces is they exclude prolonged Ca transients that may contaminate neural responses to subsequent stimuli" — directly relevant here, because the decoder must separate responses to successive 750 ms image flashes.

## 2-b. How is the `neural` data processed?

i. Two operations only: (1) invalid ROIs excluded at load time; (2) per trial, the event traces are **averaged (nanmean) over the ophys frames falling inside each 750 ms image-presentation interval**, producing an (n_neurons, n_intervals) float32 matrix. If an interval happens to contain no ophys frame, the nearest frame's value is substituted. No normalisation, z-scoring, smoothing or neuron subselection is applied.

ii.
```python
def reduce_to_bins(values, timestamps, bin_edges):
    ...
    start_idx = np.searchsorted(timestamps, bin_edges[:-1], side="left")
    end_idx = np.searchsorted(timestamps, bin_edges[1:], side="left")
    ...
    reduced = np.empty((values.shape[0], n_bins), dtype=np.float32)
    for i in range(n_bins):
        lo, hi = start_idx[i], end_idx[i]
        if hi > lo:
            reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
        else:
            nearest = np.searchsorted(timestamps, centers[i], side="left")
            ...
            reduced[:, i] = values[:, nearest]
    return reduced
```
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
```

iii. The AI treats the SDK output as already fully preprocessed (motion correction, neuropil correction, dF/F, event extraction are all done in the Allen pipeline, as it confirmed by reading the SDK source and the methods text). The only added step is the interval aggregation, justified by the paper's statement that "We performed all of our behavioral analysis after assigning behavioral events to each image presentation interval. By image presentation interval we refer to the 750 ms interval beginning with each image presentation" (methods.txt line 205, read at steps 60/201).

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROI-level: the AI relies entirely on the SDK's own ROI validity flag, requested explicitly via `exclude_invalid_rois=True`. Experiments with zero surviving ROIs are dropped (`no_valid_rois`). No additional neuron-level criteria (SNR, event rate, etc.) are applied; neuron counts per session range 4–666. Trial-level, the all-zero-neural filter described in 1-e is the only activity-based rejection.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
if len(dataset.events) == 0:
    return None, {"skip_reason": "no_valid_rois"}
```

iii. `README.md`: "Neural signal: AllenSDK inferred calcium `events`, with invalid ROIs excluded by `BehaviorOphysExperiment.from_nwb_path(..., exclude_invalid_rois=True)`." Trajectory step 22: "I've verified that the SDK object excludes invalid ROIs by default"; the methods text also states that ROIs "deemed invalid by the ROI filtering step" are excluded from the released cell set, so the AI deferred to the pipeline's QC rather than adding its own.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The trial window is the SDK trial's `start_time` → `stop_time`. Inside it, bin boundaries are the **actual image-flash onset times** (`stimulus_presentations.start_time`) of that trial, with a final edge at `last_onset + 0.75 s`. Neural frames are assigned to bins by `np.searchsorted` on `ophys_timestamps`, so the ophys clock is what defines membership; all streams (neural, running, pupil, image label, change flag) share exactly the same edges, guaranteeing alignment. Metadata records `temporal_alignment_event = "Successive image-presentation interval onsets within each AllenSDK trial"`, `off_start = 0.0`, `off_end = None`.

ii.
```python
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate(
    [stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]]
)

neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
```

iii. Anchoring on flash onsets makes every timepoint a stimulus-locked unit, which is what the paper's analyses use, and it makes the change interval land exactly on one bin. The AI checked (step 200) that "a 750 ms post-change window can include the onset of the next flash, so treating the entire post window as one constant image label is too coarse", which is precisely why it switched from a fixed window to onset-anchored bins. Empirically the change flag lands at bin index 4–11 (mean 5.6) and never at index 0, i.e. every trial retains its pre-change flashes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — aggressive rebinning. The native ophys rate is ~31 Hz for single-plane and ~11 Hz for multiscope (`ophys_rate_hz` 10.7–31.0 in the session summary). All data are rebinned to **one timepoint per 750 ms image-presentation interval** (`time_bin_size: 750.0` ms), i.e. a ~23× downsampling for single-plane data. Trials are therefore only 10–17 timepoints long (mean 11.7) instead of ~250. The converter hard-refuses any other bin size. Note that bin widths are the *actual* inter-flash intervals (≈750 ms, up to one video frame of jitter), not exactly 750 ms, except the final bin which is exactly 750 ms.

ii.
```python
TIME_BIN_MS_DEFAULT = 750.0
IMAGE_INTERVAL_S = 0.75
...
if not math.isclose(args.time_bin_ms, TIME_BIN_MS_DEFAULT, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError(
        "This converter uses native 750 ms image-presentation intervals. "
        "Keep --time-bin-ms=750."
    )
```
```python
"time_bin_size": float(time_bin_ms),
"task_description": (
    "... Each timepoint is one native 750 ms image-presentation "
    "interval, and neural events, running, pupil, and stimulus labels are "
    "aggregated over those intervals."
),
```

iii. `CONVERSION_NOTES.md`: "The whitepaper and paper describe the task as 250 ms flashed images plus 500 ms gray. The paper's behavioral processing explicitly assigns events to each 750 ms image-presentation interval. AllenSDK provides trial boundaries and stimulus presentation onset times directly, so interval-based binning is the closest match to the references while still preserving trial structure." Trajectory step 204: "the paper's 'image presentation interval' is a labeled 750 ms block starting at each flash onset". A second, unstated motive follows from 1-c: uniform 750 ms bins are what make 31 Hz single-plane and 11 Hz multiscope sessions mutually commensurate, as the format spec requires equal bin sizes across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `dataset.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`: the `image_name` column, plus the `omitted` column. Omitted flashes (5% of presentations by design) are labelled `"gray"`. The trials table's `initial_image_name` / `change_image_name` are **not** used.

ii.
```python
stimulus_presentations = dataset.stimulus_presentations.copy()
stimulus_presentations = stimulus_presentations[
    stimulus_presentations["stimulus_block_name"]
    .fillna("")
    .str.contains("change_detection")
].copy()
stimulus_presentations = stimulus_presentations.sort_values("start_time")
```
```python
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

iii. Taking the label from the stimulus table means the label is the image that was physically on the monitor in that interval, including the pre-change repeats and the post-change repeats, and it lets omissions be represented honestly instead of being attributed to an image. The AI made this switch explicitly after finding that a constant post-change label was "too coarse if we want bin-wise stimulus identity 'during the non-grey screen'" (step 200) and that omissions can occur inside a trial (step 195). Restricting to the `change_detection` block excludes the gray-screen and fingerprint-movie epochs at the ends of the session.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The set of all image labels seen anywhere in the kept data (16 natural images plus `gray`) is sorted alphabetically and mapped to integers 0–16; `gray` sorts first and gets code 0. Each trial becomes an int64 vector of per-interval codes. The same mapping is reused for the sample dataset, and is exposed through `output_values[0]`.

ii.
```python
image_names = {NO_IMAGE_LABEL}
...
for trial_image_names in session["interval_image_names"]:
    image_names.update(trial_image_names)
...
image_name_to_idx = {name: idx for idx, name in enumerate(sorted(image_names))}
```
```python
image_identity = np.asarray(
    [
        image_name_to_idx[name]
        for name in session["interval_image_names"][trial_idx]
    ],
    dtype=np.int64,
)
```
```python
"output_values": [
    [name for name, _ in sorted(image_name_to_idx.items(), key=lambda x: x[1])],
    ...
```

iii. A single global, deterministic mapping keeps the categorical codes consistent across sessions and image sets (images A and B both appear in the local subset), which is required because `output_values` is a single dataset-level list. The notes enumerate the final 17-label set.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. By construction: the image label vector has exactly one entry per bin, and the bins are defined by the very stimulus presentations the labels come from. The label for interval *i* is the image presented at the onset that opens interval *i*, and the neural matrix column *i* is the mean event rate over the ophys frames in that same interval.

ii.
```python
trial_stim = stimulus_presentations[
    (stimulus_presentations["start_time"] >= trial_start - 1e-6)
    & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
].copy()
stim_start_times = trial_stim["start_time"].to_numpy(dtype=np.float64)
bin_edges = np.concatenate([stim_start_times, [stim_start_times[-1] + IMAGE_INTERVAL_S]])
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
...
T = neural_trial.shape[1]   # == len(trial_stim) == len(image label list)
```

iii. Deriving both the bin edges and the labels from the same `trial_stim` rows removes any possibility of off-by-one drift between stimulus labels and neural bins; the AI's sanity checks (`change flag` counts, omission positions) all passed on the final artifact. Verifying on the produced pickle: in every trial with a change, the image code at the change bin differs from the code at the preceding bin (0 violations in 9,081 checked trials).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean column of the same change-detection `stimulus_presentations` rows (cross-checked against the trials table's `go` / `catch` flags).

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
if bool(row["go"]) and int(change_flags.sum()) != 1:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
if bool(row["catch"]) and int(change_flags.sum()) != 0:
    session["sanity_change_flag_mismatch_count"] += 1
    continue
...
session["interval_change_flags"].append(change_flags.astype(bool).tolist())
```

iii. `CONVERSION_NOTES.md`: "Binary interval-wise output from the Allen stimulus table `is_change`. `1` only on true change intervals. Catch trials therefore contain no positive change interval." `is_change` is the SDK's own per-flash change annotation, and the AI verified (step 61) that catch trials carry `is_sham_change` rather than `is_change`, so sham changes are correctly excluded from the positive class.

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond casting the per-interval boolean to int64 (0/1) and stacking it as output row 1. Because a timepoint *is* one image-presentation interval, the positive class is exactly the single interval that begins with the changed image — "1 right after a change in image identity", spanning that flash and the following gray period.

ii.
```python
image_change = np.asarray(
    session["interval_change_flags"][trial_idx], dtype=np.int64
)
...
output_trial = np.vstack([image_identity, image_change, running_bins, pupil_bins, trial_outcome]).astype(np.int64)
```

iii. The AI's design goal was that "each timepoint is one native 750 ms image-presentation interval", which makes the change indicator trivially a one-hot over intervals; no window length has to be chosen by hand. Measured on the artifact: 42,669 positive intervals out of 567,895 (7.5%), one per go trial, zero for catch trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is natively binary, with `output_values[1] = ["no_change", "change"]`.

ii.
```python
"output_names": ["image_identity", "image_change", "running_speed_bin", "pupil_diameter_bin", "trial_outcome"],
"output_values": [
    [...images...],
    ["no_change", "change"],
    ...
]
```

iii. Implicit in the instruction ("Image change, binary variable").

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the flag vector comes from the same `trial_stim` rows that define the bin edges, so element *i* of the change flag corresponds to neural column *i*. The change bin is the interval starting at the change flash onset.

ii.
```python
change_flags = trial_stim["is_change"].fillna(False).to_numpy(dtype=bool)
...
change_idx = np.flatnonzero(change_flags)
session["sanity_change_interval_omission_count"] += int(omitted_flags[change_idx].sum())
if change_idx[0] > 0:
    session["sanity_pre_change_omission_count"] += int(omitted_flags[change_idx[0] - 1])
```

iii. The AI added the omission bookkeeping specifically to prove the alignment: the whitepaper states that the change image and the image immediately before it are never omitted, and the converter reports 0 omitted change intervals and 0 omitted pre-change intervals on the full dataset — an independent confirmation that the change index lands on the right interval.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, columns `speed` and `timestamps` (the SDK's filtered wheel-encoder speed in cm/s).

ii.
```python
running_speed = dataset.running_speed["speed"].to_numpy(dtype=np.float32)
running_timestamps = dataset.running_speed["timestamps"].to_numpy(dtype=np.float64)
running_valid = np.isfinite(running_speed) & np.isfinite(running_timestamps)
if running_valid.sum() == 0:
    return None, {"skip_reason": "missing_running_speed"}
running_speed = running_speed[running_valid]
running_timestamps = running_timestamps[running_valid]
```

iii. `CONVERSION_NOTES.md`: "Used AllenSDK `running_speed["speed"]`." The methods text describes the SDK's `running_processing` module (unwrapping, transient removal) as the authoritative running-speed computation, so the AI takes the SDK output as-is; the tutorial it read also plots `running_speed['speed']` against `timestamps`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are dropped; the remaining samples are averaged (`nanmean`) within each 750 ms image-presentation interval on the running-speed clock; the resulting per-interval means are then discretised into 5 global quantile bins whose edges are computed once over **all** kept intervals of all sessions.

ii.
```python
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```
```python
def compute_quantile_edges(values, nbins):
    percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    edges = np.quantile(values, percentiles)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def digitize_with_edges(values, edges):
    values = np.asarray(values, dtype=np.float32)
    return np.searchsorted(edges, values, side="right").astype(np.int64)
```
```python
all_running_values = np.concatenate(all_running_values).astype(np.float32)
running_edges = compute_quantile_edges(all_running_values, nbins=5)
...
running_bins = digitize_with_edges(session["running_cont"][trial_idx], running_edges)
```

iii. Averaging within the interval is the natural aggregation once the timebase is the image-presentation interval. Global (rather than per-session) quantiles are needed so that the single `output_values` list means the same thing in every session; the AI verified exact balance in the artifact ("running bins: 113,579 each"). The `np.nextafter` guard prevents degenerate/duplicate edges when the distribution is heavily zero-inflated (mice are stationary a large fraction of the time — the first edge is 0.0045 cm/s).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-population bins (quintiles) computed globally: interior edges at the 20th/40th/60th/80th percentiles of all interval-mean speeds, i.e. `[0.0045, 0.976, 16.37, 33.30]` cm/s; `output_values[2] = ["bin_0".."bin_4"]`.

ii.
```python
percentiles = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
edges = np.quantile(values, percentiles)
...
"running_speed_bin_edges": [float(x) for x in running_edges.tolist()],
```

iii. Directly from the instruction ("Running speed, discretized into five equal percentile bins"). Verified in the notes and reproducible from the artifact: 113,579 intervals in each of the 5 bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned onto the *same* `bin_edges` array as the neural data, using its own native timestamps; no interpolation onto the ophys clock is performed — instead both streams are reduced onto the common stimulus-interval grid. Trials where any running bin is empty *and* would be non-finite are dropped; an interval containing no running sample takes the nearest sample.

ii.
```python
neural_trial = reduce_to_bins(events_matrix, ophys_timestamps, bin_edges)
running_trial = reduce_to_bins(running_speed, running_timestamps, bin_edges)
```
```python
nearest = np.searchsorted(timestamps, centers[i], side="left")
nearest = min(max(nearest, 0), len(timestamps) - 1)
if nearest > 0 and abs(timestamps[nearest - 1] - centers[i]) < abs(timestamps[nearest] - centers[i]):
    nearest -= 1
reduced[i] = values[nearest]
```

iii. The SDK synchronises all behavioural streams to the same session clock (the AI checked the SDK's timestamp handling at steps 14–22), so reducing each stream onto the same wall-clock bin edges is sufficient for alignment and avoids a separate interpolation step. Running speed is sampled at ~60 Hz, so a 750 ms bin contains ~45 samples and the nearest-sample fallback is essentially never used.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking["pupil_width"]` with `dataset.eye_tracking["timestamps"]`. The AI does not reference `likely_blink` explicitly; it instead treats NaNs as missing — which is equivalent in effect, because the SDK's `filter_on_blinks` has already set `pupil_width` (and the other ellipse columns) to NaN on likely-blink frames.

ii.
```python
if len(dataset.eye_tracking) == 0:
    return None, {"skip_reason": "missing_eye_tracking"}

eye_timestamps = dataset.eye_tracking["timestamps"].to_numpy(dtype=np.float64)
pupil_width = fill_nan_by_time(
    dataset.eye_tracking["pupil_width"].to_numpy(),
    eye_timestamps,
)
if pupil_width is None:
    return None, {"skip_reason": "all_pupil_nan"}
```

iii. `CONVERSION_NOTES.md`: "Used AllenSDK `eye_tracking["pupil_width"]` as the pupil-size measure." This is the column the SDK tutorial the AI read uses for plotting pupil diameter (`plot_pupil` uses `pupil_sample['pupil_width']`), and it is a diameter (in pixels) rather than an area, matching the requested variable.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) NaN samples (blinks/tracking failures) are linearly interpolated in time from the surrounding valid samples — with constant fill if only one valid sample exists, and edge-value extrapolation at the ends; (2) the cleaned trace is averaged within each image-presentation interval; (3) the per-interval means are discretised into 5 global quantile bins exactly as for running speed.

ii.
```python
def fill_nan_by_time(values, timestamps):
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(timestamps)
    if valid.sum() == 0:
        return None
    if valid.sum() == 1:
        filled = np.empty_like(values)
        filled[:] = values[valid][0]
        return filled
    return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)
```
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
...
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
pupil_bins = digitize_with_edges(session["pupil_cont"][trial_idx], pupil_edges)
```

iii. `CONVERSION_NOTES.md`: "Missing values were linearly interpolated in timestamp space before interval binning." Interpolating before binning prevents blink gaps from turning whole intervals into NaN (which would have caused the entire trial to be discarded by the finiteness filter), while keeping the interpolation local in time.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global equal-population bins, interior edges `[36.79, 41.63, 46.21, 52.30]` (pixels); `output_values[3] = ["bin_0".."bin_4"]`. Bin occupancies in the artifact are exactly equal (113,579 each).

ii.
```python
pupil_edges = compute_quantile_edges(all_pupil_values, nbins=5)
...
"pupil_diameter_bin_edges": [float(x) for x in pupil_edges.tolist()],
```

iii. Same rationale as running speed, and required by the instruction ("Pupil diameter, discretized into five equal percentile bins"). Global edges are used so that the 5 categories are comparable across mice and sessions, at the cost of confounding between-animal differences in absolute pupil size with within-session arousal.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The same mechanism as running speed: the cleaned pupil trace is reduced onto the shared `bin_edges`, using the eye-tracking camera's own timestamps, giving one value per neural column. Any trial that still contains a non-finite pupil bin is dropped.

ii.
```python
pupil_trial = reduce_to_bins(pupil_width, eye_timestamps, bin_edges)
if (np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial))
        or np.any(~np.isfinite(pupil_trial))):
    continue
```

iii. As for running: the eye-tracking timestamps are already on the synchronised session clock, so binning onto common wall-clock edges is sufficient. The eye camera runs at ~60 Hz, well above the 1.33 Hz bin rate.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The mutually exclusive boolean columns `hit`, `miss`, `false_alarm`, `correct_reject` of the SDK trials table, checked in that fixed order. A trial matching none of them raises an error (never triggered, because the trial filter already removed aborted/auto-rewarded trials).

ii.
```python
TRIAL_OUTCOME_VALUES = ["hit", "miss", "false_alarm", "correct_reject"]
TRIAL_OUTCOME_TO_INT = {name: idx for idx, name in enumerate(TRIAL_OUTCOME_VALUES)}


def get_trial_outcome(row):
    if bool(row["hit"]):
        return TRIAL_OUTCOME_TO_INT["hit"]
    if bool(row["miss"]):
        return TRIAL_OUTCOME_TO_INT["miss"]
    if bool(row["false_alarm"]):
        return TRIAL_OUTCOME_TO_INT["false_alarm"]
    if bool(row["correct_reject"]):
        return TRIAL_OUTCOME_TO_INT["correct_reject"]
    raise ValueError("Trial does not have a valid decoder outcome.")
```

iii. These four labels are the canonical outcome taxonomy described in the whitepaper ("this trial structure leads to a sampling of 'GO' and 'CATCH' trials, that when combined with mouse responding, yields 'HIT', 'MISS', 'FALSE ALARM', and 'CORRECT REJECTION' trials"), and the AI confirmed the columns exist in the SDK trial table at steps 61–62. The hard `raise` is a deliberate assertion that the go/catch filter and the outcome flags are consistent.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is an integer 0–3 from the fixed mapping, broadcast as a constant across all intervals of the trial (so it is stored as a time-varying row even though it is static per trial), and placed in output row 4. Final counts: hit 15,217 / miss 27,452 / false_alarm 864 / correct_reject 5,122.

ii.
```python
trial_outcome = np.full(
    T, session["trial_outcomes"][trial_idx], dtype=np.int64
)
...
output_trial = np.vstack([
    image_identity, image_change, running_bins, pupil_bins, trial_outcome,
]).astype(np.int64)
...
"output_values": [..., TRIAL_OUTCOME_VALUES],
```

iii. The format spec asks for time-varying outputs "if at all possible" and requires all output rows of a trial to share `n_timepoints`; broadcasting a per-trial scalar satisfies both. The AI also tracks the outcome counts in metadata so the class imbalance is visible.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is explicit and mostly by exclusion, recorded in metadata:
- Experiment-level: empty eye-tracking table, zero valid ROIs, all-NaN pupil, no finite running samples, empty change-detection stimulus block, or <2 usable trials → the experiment is skipped and the reason is appended to `metadata["skipped_sessions"]` (3 skips, all `missing_eye_tracking`).
- Pupil NaNs (blinks/tracking dropouts) → linearly interpolated in time before binning.
- Running NaNs → those samples are removed before binning.
- Intervals containing no sample of a stream → filled with the temporally nearest sample.
- Any residual non-finite value in a binned trial, or an all-zero neural trial → the trial is dropped (4.7% of trials).
- Stimulus/trial-table inconsistencies (`is_change` count vs go/catch) → the trial is dropped and counted.
There is **no** `try/except` around session loading: an unexpected exception (e.g. a corrupt NWB, or a trial matching none of the four outcomes) aborts the entire run rather than skipping that experiment.

ii.
```python
if session is None:
    skipped.append({"ophys_experiment_id": int(experiment_id), **stats})
    continue
...
full_data["metadata"]["skipped_sessions"] = skipped
```
```python
return np.interp(timestamps, timestamps[valid], values[valid]).astype(np.float32)
```
```python
if (np.any(~np.isfinite(neural_trial)) or np.any(~np.isfinite(running_trial))
        or np.any(~np.isfinite(pupil_trial))):
    continue
if np.all(neural_trial == 0):
    continue
```

iii. The AI's stated principle is to exclude rather than impute anything that would become a decoder label: missing behaviour would otherwise have to be given a fabricated category. Silent trials were removed after the validator surfaced them ("those are not useful for decoding", step 86). Every exclusion is counted in `session_summary` / `skipped_sessions` and reported in `CONVERSION_NOTES.md` so the curation is auditable.

## 9-a. What are the most time-consuming steps of the code?

i. Reading and deserialising the 202 NWB files through `BehaviorOphysExperiment.from_nwb_path` dominates: the AI measured ~1–2 s per file and the full pass took on the order of 10 minutes, with the checkpoint cadence tracking file loads, not computation. The next largest costs are inside `load_session`: the per-trial boolean mask over the whole `stimulus_presentations` table (~300 trials × ~4,800 presentations per session) and the pure-Python per-bin loops in `reduce_to_bins` (~568k bins × 3 streams). Assembly, discretisation and pickling are comparatively cheap (though the 397 MB pickle write is non-trivial).

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(
    str(nwb_path), exclude_invalid_rois=True
)
```
```python
for trial_id, row in valid_trials.iterrows():
    trial_stim = stimulus_presentations[
        (stimulus_presentations["start_time"] >= trial_start - 1e-6)
        & (stimulus_presentations["start_time"] < trial_stop + 1e-6)
    ].copy()
```

iii. Trajectory step 105: "The runtime is dominated by repeated NWB deserialization rather than the actual binning"; step 100: "The main runtime is NWB loading through the AllenSDK, but the data volume after trial filtering and 750 ms binning is much smaller than the raw sessions." The AI never profiled the per-trial pandas masking.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- `reduce_to_bins`'s `for i in range(n_bins)` loop — with `start_idx`/`end_idx` already computed by `searchsorted`, the means could be obtained in one shot with `np.add.reduceat` (or a cumulative-sum difference) instead of a Python-level `np.nanmean` per bin per stream.
- The per-trial `stimulus_presentations` boolean mask + `.copy()` — a single `np.searchsorted` on the sorted `start_time` array (or a `groupby("trials_id")`) would replace an O(n_trials × n_presentations) scan with O(n_trials log n).
- The per-interval Python list comprehension that maps image names to codes could be a single `pd.Series.map` / factorize over the whole session.
None of these were vectorised; the per-experiment loop itself is inherently I/O-bound and could only be parallelised, not vectorised.

ii.
```python
for i in range(n_bins):
    lo = start_idx[i]
    hi = end_idx[i]
    if hi > lo:
        reduced[:, i] = np.nanmean(values[:, lo:hi], axis=1, dtype=np.float64)
```
```python
image_identity = np.asarray(
    [image_name_to_idx[name] for name in session["interval_image_names"][trial_idx]],
    dtype=np.int64,
)
```

iii. The AI's justification is implicit and is about relative cost: since "the runtime is dominated by repeated NWB deserialization rather than the actual binning" (step 105), it chose the straightforward per-bin implementation, which also makes the empty-bin nearest-sample fallback easy to express.

## 9-c. What processing does the code repeat multiple times?

i. The main repeat is that the whole assembly stage runs **twice**: `convert_sessions_to_dataset` is called once for all 199 sessions and again for the 11-session sample, so those 11 sessions are re-digitised, re-coded and re-stacked, and a second 9.5 MB pickle is written. Otherwise each NWB file is read exactly once, and the per-session extraction results are cached in memory and reused for both the global quantile-edge computation and the final assembly. Minor repeats: `.astype(np.float32)` is applied to arrays that are already float32, `reduce_to_bins` recomputes `centers` for every trial, and `np.isfinite` is evaluated over each binned trial three times.

ii.
```python
full_data = convert_sessions_to_dataset(sessions=sessions, ...)
...
sample_sessions = [sessions[i] for i in sample_indices]
sample_data = convert_sessions_to_dataset(sessions=sample_sessions, ...)
```
```python
neural_trials.append(neural_trial.astype(np.float32))  # already float32
```

iii. The duplication is intentional: the AI wanted a small artifact for fast format verification and decoder smoke tests ("sample_data.pkl: smaller dataset generated for quick checks"), and it reuses the *full* dataset's quantile edges and image mapping for the sample so the two are comparable. It is cheap relative to loading.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items that the decoder never uses:
- The entire `sample_data.pkl` branch (subset selection, second assembly, second pickle) — not part of the requested deliverable.
- `summarize_sessions` and the per-session diagnostic counters (`sanity_total_omitted_intervals`, `sanity_change_interval_omission_count`, `sanity_pre_change_omission_count`, `sanity_change_flag_mismatch_count`, `ophys_rate_hz`, …), plus `skipped_sessions`, all embedded into `metadata`.
- The final summary statistics block (re-concatenating every running/pupil/change row of the full dataset just to print bin counts).
- `input` is stored as an explicit `np.zeros((0, T))` array per trial, allocated per trial although the task specifies no decoder inputs.
- Loading the full `BehaviorOphysExperiment` object per file pulls in more than the six fields actually used.

ii.
```python
session_summary = summarize_sessions(session_stats, sessions)
full_data["metadata"]["session_summary"] = session_summary.to_dict(orient="records")
full_data["metadata"]["skipped_sessions"] = skipped
```
```python
running_bins = np.concatenate(
    [trial[2] for session_trials in full_data["output"] for trial in session_trials]
)
```
```python
input_trials.append(np.zeros((0, T), dtype=np.float32))
```

iii. These are documentation/QC artifacts rather than oversights: the AI explicitly wanted its notes to "report measured results instead of intermediate smoke-test numbers" (step 243) and to prove the omission/change-flag sanity properties against the whitepaper. The cost is small compared with NWB loading, though the metadata does inflate the pickle and the diagnostics are duplicated into both output files.
