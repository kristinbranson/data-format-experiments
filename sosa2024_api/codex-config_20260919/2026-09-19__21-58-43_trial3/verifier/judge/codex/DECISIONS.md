# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file under `/app/data/sub-*/*.nwb` in sorted path order, unless `--sample` is used. Each file is opened with `pynwb.NWBHDF5IO(..., load_namespaces=True)`, converted session-by-session, and then assembled into the final top-level dictionary.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    ...
    return files

with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. `CONVERSION_NOTES.md` says the conversion should cover all 152 provided NWBs, use deterministic path order, and use only the `pynwb` API rather than direct HDF5 access.

## 1-b. How are the data split into subjects (mice)?

i. The agent treats subject identity as the NWB subject ID for each loaded file, checks that it matches the subject encoded in the filename, and then builds the final `subjects` list from the converted sessions sorted numerically (`m3`, `m4`, ...).

ii.
```python
subject_from_path, day = session_key(path)
...
subject = nwb.subject.subject_id
if subject != subject_from_path:
    raise ValueError(f"Subject mismatch in {path}: {subject} vs {subject_from_path}")
...
subjects = sorted({x["subject"] for x in converted_sessions}, key=lambda x: int(x[1:]))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes say the dataset is organized as `sub-<mouse>` directories, but the agent additionally verifies the path-derived mouse against the NWB metadata before building `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is parsed from the filename (`sub-m##_ses-##`) and later formatted as `sub-{subject}_ses-{day:02d}` in metadata.

ii.
```python
def session_key(path: Path) -> tuple[str, int]:
    match = re.search(r"sub-(m\d+)_ses-(\d+)", path.name)
    ...
    return match.group(1), int(match.group(2))
...
session_id = f"sub-{subject}_ses-{day:02d}"
```

iii. `CONVERSION_NOTES.md` states that all 152 files are individual sessions and that session/day is encoded in the filename.

## 1-d. How are the data split into trials?

i. Trials are defined by paired `trial_start` and `teleport` events from the aligned behavior streams. The agent finds all indices where `trial_start > 0` and all indices where `teleport > 0`, checks that counts match and `stop > start`, and uses `[start:stop]` slices for every per-trial stream.

ii.
```python
b = load_behavior(behavior, common_length)
starts = np.flatnonzero(b["trial_start"] > 0)
stops = np.flatnonzero(b["teleport"] > 0)
if len(starts) != len(stops) or len(starts) < 2:
    raise ValueError(f"Invalid trial event counts in {path}: {len(starts)}/{len(stops)}")
if np.any(stops <= starts):
    raise ValueError(f"Non-positive trial interval in {path}")
...
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
```

iii. The notes justify direct `[start:stop]` slicing as the correct NWB interpretation of start-to-teleport intervals, arguing that the NWB event arrays are already explicit frame indices and should not get the legacy `-1` offset from the original code.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes trials if they have fewer than 2 samples, include frames where `scanning != 1`, contain non-finite required behavior values, or have a corrupt lick sensor (`>30%` of frames with `lick > 2`). It also requires at least 2 retained trials per session.

ii.
```python
if stop - start < 2:
    reason = "fewer than two samples"
elif not np.all(b["scanning"][start:stop] == 1):
    reason = "outside valid scanning period"
required = np.vstack((b["position"][start:stop], b["speed"][start:stop],
                      b["lick"][start:stop], timestamps[start:stop]))
if not np.all(np.isfinite(required)):
    reason = "non-finite required behavior"
lick_bad_fraction = float(np.mean(b["lick"][start:stop] > 2))
if lick_bad_fraction > 0.30:
    reason = "corrupt lick sensor (>30% frames with count >2)"
...
if len(neural_trials) < 2:
    raise ValueError(f"Fewer than two retained trials in {path}")
```

iii. `CONVERSION_NOTES.md` says the agent intentionally copied the paper's 30%-lick-corruption criterion, kept only valid synchronized frames, and did not impose a speed filter because stationary periods are required by the decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from NWB `ophys/Fluorescence`, `ophys/Neuropil`, and the segmentation table's `iscell` column. The stored NWB `Deconvolved` signal is not used.

ii.
```python
ophys = nwb.processing["ophys"]
fluorescence_container = ophys["Fluorescence"]
neuropil_container = ophys["Neuropil"]
segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]
...
iscell = np.asarray(segmentation["iscell"].data[:])
manual_ids = np.flatnonzero(iscell[:, 0] == 1)
```

iii. The notes explicitly say the NWB `Deconvolved` array is a non-reference Suite2p output and that the paper instead analyzes events reconstructed from raw fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. The agent first keeps only manually curated ROIs, loads matching fluorescence and neuropil traces across all plane response series, then recomputes dF/F trial-by-trial: `F - 0.7*Fneu + 0.7*mean(Fneu_trial)`, Gaussian smoothing with sigma 15, 300-sample min then max filters for the baseline, `(corrected - baseline)/abs(baseline)`, and a second Gaussian smooth with sigma 2. After the speed-correlation cell filter, it deconvolves each trial's dF/F with `dcnv.oasis(..., tau=0.7, fs=15.5078125)` and clips negative events to zero.

ii.
```python
corrected = f - 0.7 * fn + 0.7 * np.nanmean(fn, axis=1, keepdims=True)
baseline = gaussian_smooth(corrected, 15)
baseline = ndimage.minimum_filter1d(baseline, 300, axis=-1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
with np.errstate(divide="ignore", invalid="ignore"):
    trial_dff = (corrected - baseline) / np.abs(baseline)
trial_dff = gaussian_smooth(trial_dff, 2)
...
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
events[events < 0] = 0
```

iii. The notes say this was intended to recreate the paper's maximin dF/F and OASIS pipeline while using the synchronized ~15.5 Hz frame grid already present in the NWBs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two neural filters are applied: keep only manually curated `iscell[:, 0] == 1` ROIs, then exclude cells whose reconstructed dF/F is correlated with running speed at `r > 0.5`.

ii.
```python
iscell = np.asarray(segmentation["iscell"].data[:])
manual_ids = np.flatnonzero(iscell[:, 0] == 1)
...
correlations = speed_correlations(dff, b["speed"])
keep = ~(correlations > 0.5)
kept_ids = manual_ids[keep]
dff = dff[keep]
```

iii. `CONVERSION_NOTES.md` says these are the paper's curation steps and that place-cell filtering was intentionally not applied because the requested decoder has broader outputs than the paper's place-cell analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing the reconstructed session-level dF/F/events with each trial's `[start:stop]` interval, where `start` comes from `trial_start`.

ii.
```python
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
    ...
    trial_dff = dff[:, start:stop]
    ...
    neural_trials.append(events)
```

iii. The notes say the alignment event is `trial_start`, so no extra temporal shifting is needed beyond splitting the synchronized framewise arrays into trial intervals.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent fixes the output time bin size at `1000 / 15.5078125` ms and does not rebin or resample the aligned framewise data. If neural and behavior lengths differ slightly, all streams are cropped to a shared `common_length`.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                    [x.data.shape[0] for x in neuropil_series])
...
"time_bin_size": TIME_BIN_MS,
```

iii. The notes state that the behavior timestamps are already synchronized to one neural row per frame and should therefore be kept at native ~64.48 ms resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The agent derives this input from the behavior timestamps attached to the `position` time series, which are loaded into `b["timestamps"]`.

ii.
```python
def load_behavior(behavior, common_length: int) -> dict[str, np.ndarray]:
    ...
    arrays["timestamps"] = np.asarray(behavior["position"].timestamps[:common_length])
    return arrays
```

iii. The notes say the behavioral streams share a common aligned frame clock; the agent chose the `position` timestamps as the authoritative session timebase.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Within each retained trial, the first timestamp is subtracted from all timestamps so the row begins at zero seconds from trial start.

ii.
```python
time_from_start = timestamps[start:stop] - timestamps[start]
...
inputs = np.vstack((
    time_from_start,
    ...
)).astype(np.float32)
```

iii. The notes describe this as direct trial-start alignment on the native synchronized timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `start:stop` slice is used for timestamps and neural data, after all streams are truncated to a shared `common_length`. This yields one time value per neural sample.

ii.
```python
common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                    [x.data.shape[0] for x in neuropil_series])
...
time_from_start = timestamps[start:stop] - timestamps[start]
...
trial_dff = dff[:, start:stop]
...
if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError(f"Alignment shape mismatch in {path}, trial {trial_idx}")
```

iii. The notes say the NWBs already provide behavior and imaging on a common frame grid, with only minor trailing-length mismatches that should be cropped away.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii.
```python
environment = mode_value(b["environment"][start:stop])
```

iii. The notes identify `environment` as the synchronized behavior stream that encodes ENV1 vs ENV2 and state that day-8 switches occur at trial index 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent takes the modal `environment` value within the trial and repeats that value across all timepoints in the trial input matrix.

ii.
```python
environment = mode_value(b["environment"][start:stop])
...
np.full(stop - start, environment),
```

iii. The notes say this row is intended to be a per-trial contextual variable rather than a fluctuating time series.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The agent derives trial number from the synchronized behavior `trial number` time series, not from the Python loop index.

ii.
```python
trial_number = mode_value(b["trial number"][start:stop])
```

iii. The notes say the stored trial number is treated as the native zero-based within-session trial identifier and is repeated as decoder context.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The modal `trial number` value within the current trial interval is computed and then repeated across every timepoint of that trial's input matrix.

ii.
```python
trial_number = mode_value(b["trial number"][start:stop])
...
np.full(stop - start, trial_number),
```

iii. The notes frame this as preserving the source trial identity while fitting the uniform 2-D per-trial array format.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The agent derives previous outcome from `behavior["Reward"].timestamps[:]`, together with the current session's trial boundaries and behavior timestamps.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:])
timestamps = b["timestamps"]
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
```

iii. The notes say reward outcome should come from in-trial reward deliveries only, with three inter-trial reward events intentionally left unmapped.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes a binary current-trial outcome for every source trial, then for trial `i` uses `outcomes[i-1]` as the previous-outcome input, with the first trial forced to zero. That scalar is repeated across all timepoints in the current trial.

ii.
```python
previous_outcome = int(outcomes[trial_idx - 1]) if trial_idx > 0 else 0
...
np.full(stop - start, previous_outcome),
```

iii. The notes justify using source-trial order so that excluding a corrupt lick trial does not change the true predecessor relationship for later trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavior `position` time series and an active reward-zone label inferred from the NWB session identifier. `scene_zone_labels()` extracts one or two zone letters from `nwb.identifier`; `zones_by_trial()` assigns them across trials, switching after trial 30 when needed.

ii.
```python
scene = scene_from_identifier(nwb.identifier)
zones = zones_by_trial(scene, len(starts))
...
position = b["position"][start:stop]
distance = reward_distance(position, zone)
```

iii. The notes say the reward-zone schedule is recoverable from session identifiers and matches the paper's `get_reward_zones` logic better than using the noisy framewise `reward_zone` stream to infer labels.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest part of the active reward zone: negative before the zone start, zero anywhere inside the zone, and positive after the zone end.

ii.
```python
def reward_distance(position: np.ndarray, zone: str) -> np.ndarray:
    start, stop = ZONE_BOUNDS[zone]
    return np.where(position < start, position - start,
                    np.where(position > stop, position - stop, 0.0))
```

iii. `CONVERSION_NOTES.md` says this was chosen because the requested decoder output is distance to any point in the reward zone, not the paper's reward-relative coordinate centered on zone start.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 classes with explicit inclusive/exclusive thresholds matching the task specification: `<-50`, `[-50,-10]`, `(-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii.
```python
def discretize_reward_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int16)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. The notes say the boundaries were unit-tested explicitly because exact cut-point handling matters for the decoder evaluation.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The position values and the trial-specific zone label are both evaluated on the exact same `[start:stop]` frame indices used for the neural trial matrix, so one distance class is produced per neural sample.

ii.
```python
position = b["position"][start:stop]
distance = reward_distance(position, zone)
...
trial_dff = dff[:, start:stop]
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
```

iii. The notes describe all time-varying inputs and outputs as frame-aligned to the same native imaging/behavior clock.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii.
```python
position = b["position"][start:stop]
```

iii. The notes say valid track position spans roughly 0 to 450 cm and that the task requests absolute corridor position rather than reward-relative position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent takes the per-trial `position` slice and discretizes it into 5 corridor bins with a custom thresholding function.

ii.
```python
outputs = np.vstack((
    ...,
    discretize_position(position),
    ...
)).astype(np.int16, copy=False)
```

iii. The notes say this keeps the requested output on the original frame grid while translating the continuous position signal into categorical classes for decoding.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into 5 equal-width bins over the 450 cm track: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int16)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```

iii. The notes say the cut points were boundary-tested to ensure exact treatment of `90`, `180`, `270`, and `360`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position uses the same per-trial `start:stop` frame interval as the neural data and the rest of the trial inputs/outputs.

ii.
```python
position = b["position"][start:stop]
...
trial_dff = dff[:, start:stop]
```

iii. The notes say behavior and neural rows are already synchronized at imaging-frame resolution, so shared slicing is the intended alignment mechanism.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii.
```python
lick = b["lick"][start:stop]
```

iii. The notes describe this source as cumulative per-frame lick counts in the aligned behavior stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick counts are binarized: `lick > 0` becomes class `1`, otherwise `0`.

ii.
```python
(lick > 0).astype(np.int16),
```

iii. The notes say binary lick presence is the decoder target, while trials with gross lick-sensor corruption are removed entirely.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick row is produced from the same `[start:stop]` frame slice used for the neural trial matrix, so it is sample-aligned without interpolation.

ii.
```python
lick = b["lick"][start:stop]
...
trial_dff = dff[:, start:stop]
```

iii. The notes say all retained variables are already synchronized on the common imaging-frame clock.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the NWB session identifier (`nwb.identifier`), whose scene tokens encode one or two active reward zones. Trial boundaries determine when the session changes zone if it is a switch session.

ii.
```python
scene = scene_from_identifier(nwb.identifier)
zones = zones_by_trial(scene, len(starts))
```

iii. The notes say the framewise `reward_zone` series signals zone entry but does not by itself give the desired per-trial A/B/C label as cleanly as the identifier schedule does.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent parses A/B/C labels from the identifier. If only one label is present, that zone is used for every trial; if two labels are present, the first 30 trials get the first label and later trials get the second. The chosen label is then encoded as `0/1/2` and repeated over the trial.

ii.
```python
def zones_by_trial(scene: str, n_trials: int) -> list[str]:
    labels = scene_zone_labels(scene)
    if len(labels) == 1:
        return [labels[0]] * n_trials
    ...
    return [labels[0]] * 30 + [labels[1]] * (n_trials - 30)
...
np.full(stop - start, "ABC".index(zone), dtype=np.int16),
```

iii. The notes say this directly mirrors the task design described in the paper and was validated against known switch timing.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `behavior["Reward"].timestamps[:]` together with each trial's `start` and `stop` timestamps.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:])
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
```

iii. The notes say the desired label is whether reward delivery occurred within the current start-to-teleport interval.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial-level binary outcome is computed from whether any reward timestamp falls inside the trial interval. That binary value is then repeated across every timepoint in the trial output matrix.

ii.
```python
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
...
np.full(stop - start, outcomes[trial_idx], dtype=np.int16),
```

iii. `CONVERSION_NOTES.md` says the agent intentionally leaves three reward events outside all start-to-teleport intervals unmapped rather than leaking them into adjacent trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles mismatched stream lengths by truncating everything to a shared `common_length`; rejects files with missing fluorescence/neuropil series, malformed `iscell`, invalid trial counts, or no curated cells; excludes trials with non-finite required behavior or corrupt licks; errors if all cells are filtered out or fewer than two trials remain; and runs a final validator over shapes, finiteness, and output class ranges.

ii.
```python
if not fluorescence_series or not neuropil_series:
    raise ValueError(f"Missing fluorescence or neuropil response series in {path}")
common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                    [x.data.shape[0] for x in neuropil_series])
...
if not np.all(np.isfinite(required)):
    reason = "non-finite required behavior"
...
if len(neural_trials) < 2:
    raise ValueError(f"Fewer than two retained trials in {path}")
...
validate_converted(data)
```

iii. The notes describe this as strict synchronization/shape checking plus targeted trial exclusion, with reward-zone identity obtained from scene metadata so no special imputation of missing `reward_zone` frames is needed.

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are reading large NWB arrays from disk, loading selected ROI series across planes, computing dF/F with Gaussian and min/max filters across all curated cells and trial windows, running speed-correlation filtering, performing per-trial OASIS deconvolution, and finally writing the large pickle.

ii.
```python
fluorescence = load_selected_roi_series(
    fluorescence_container, manual_ids, common_length, len(segmentation))
neuropil = load_selected_roi_series(
    neuropil_container, manual_ids, common_length, len(segmentation))
dff, plot_example = calculate_reference_dff(fluorescence, neuropil, starts, stops)
correlations = speed_correlations(dff, b["speed"])
...
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
...
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly say that loading ~92 GB of NWB assets and doing cell-by-frame filtering/deconvolution are the dominant costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining non-vectorized loops are the trial loop in `calculate_reference_dff`, the trial loop in `convert_session`, and the session loop in `main`. The agent already vectorized speed correlations and thresholding/discretization, so the remaining loops are mostly around variable-length trial slicing and repeated OASIS calls.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
    ...

for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
    ...

for idx, path in enumerate(files):
    ...
```

iii. The notes say vectorization was added where straightforward, but per-trial processing was kept because the data are organized as variable-length trial intervals.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly slices the same session-level arrays per trial to build inputs, outputs, and neural trials. It first computes a full-session dF/F matrix and then revisits each trial again for deconvolution and packaging. For plotting, it also copies example traces and trial payloads for the first retained trial.

ii.
```python
dff, plot_example = calculate_reference_dff(fluorescence, neuropil, starts, stops)
...
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
    ...
    position = b["position"][start:stop]
    speed = b["speed"][start:stop]
    lick = b["lick"][start:stop]
    ...
    trial_dff = dff[:, start:stop]
```

iii. The notes emphasize that the agent removed the human solution's separate whole-dataset survey pass, but within each session it still does repeated per-trial slicing because the output format itself is organized by trial.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds `plot_example` and optional processing plots for visualization, records extensive `session_meta` bookkeeping, runs discretizer self-tests, and computes dF/F for all manually curated cells before discarding the speed-correlated ones. It also loads and checks some behavior/QC streams (`scanning`, `reward_zone`, `autoreward`) that are not saved as decoder variables.

ii.
```python
dff, plot_example = calculate_reference_dff(fluorescence, neuropil, starts, stops)
correlations = speed_correlations(dff, b["speed"])
keep = ~(correlations > 0.5)
...
if show_processing and plot_payload is not None:
    ...
    plot_path = build_processing_plot(...)
...
session_meta = {
    ...
    "excluded_trials": excluded_trials,
    ...
}
...
check_discretizer_boundaries()
```

iii. The notes present most of this as validation/provenance work rather than decoder input preparation; it is useful for auditing but not consumed by downstream model training.
