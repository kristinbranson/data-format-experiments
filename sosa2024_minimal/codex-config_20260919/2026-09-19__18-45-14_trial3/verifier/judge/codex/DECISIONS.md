# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file under `data/sub-*/*.nwb`, sorts them by parsed mouse/session numbers, and converts each file as one session. It uses direct HDF5 access via `h5py` rather than `pynwb`.

ii.
```python
def build_dataset(data_root: Path) -> dict:
    files = sorted(data_root.glob("sub-*/*.nwb"), key=natural_key)
    ...
    for number, path in enumerate(files, start=1):
        session_neural, session_input, session_output, info = convert_session(path)
```
```python
def convert_session(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    with h5py.File(path, "r") as h5:
        behavior = h5["processing/behavior/BehavioralTimeSeries"]
```

iii. In the trajectory, the agent first established that the source data were full NWB session files and that every recording should be retained as a session. It explicitly chose direct HDF5 access after inspecting the NWB layout.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the `sub-m*` filenames/directories. The output `subjects` list is the sorted unique set of parsed mouse IDs, and each session gets a `subject_idx` entry via lookup.

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    return int(match.group(1)), int(match.group(2))
...
subjects = sorted({f"m{natural_key(path)[0]}" for path in files},
                  key=lambda value: int(value[1:]))
subject_lookup = {subject: index for index, subject in enumerate(subjects)}
```

iii. The trajectory shows the agent treating each NWB filename as encoding mouse/day identity and preserving that organization in the converted dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session.

ii.
```python
for number, path in enumerate(files, start=1):
    session_neural, session_input, session_output, info = convert_session(path)
    neural.append(session_neural)
```

iii. The agent stated that “a session is one NWB recording (one mouse/day)” and kept all 152 recordings as sessions.

## 1-d. How are the data split into trials?

i. Trials are sliced from each `trial_start` sample up to, but not including, the matching `teleport` sample. The agent uses the positive samples of `trial_start` and `teleport` directly as boundary indices.

ii.
```python
starts = np.flatnonzero(behavior["trial_start/data"][:] > 0)
stops = np.flatnonzero(behavior["teleport/data"][:] > 0)
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial boundaries in {path}")
...
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. In the trajectory, the agent said trials should run from `trial_start` to `teleport`, excluding the teleport period to match the main paper analyses.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes trials with the paper’s “stuck lick sensor” signature: more than 30% of samples in the trial have `lick > 2`. It does not implement the human reference’s short-trial `< 50` timepoint filter. It also raises an error if a session ends up with fewer than two retained trials.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
bad_lick_sensor = np.mean(trial_lick > 2) > LICK_ERROR_FRACTION
...
if bad_lick_sensor:
    excluded_lick_trials += 1
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"Fewer than two retained trials in {path}")
```

iii. The trajectory explicitly justifies this as reproducing the paper’s 81 lick-detector failures and argues that those trials must be excluded because lick is a required categorical decoder target.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from the raw fluorescence traces in `processing/ophys/Fluorescence` and neuropil traces in `processing/ophys/Neuropil`, restricted to curated ROIs from `ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
...
iscell_table = h5["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][:, 0]
iscell = np.asarray(iscell_table == 1)
```

iii. The trajectory shows the agent initially considering the NWB `Deconvolved` field, then rejecting it after checking that rerunning the paper’s pipeline produced a different signal scale. Its final justification was that the paper’s analysis signal must be recomputed from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. For each trial, the agent subtracts `0.7 * neuropil`, adds back the trial mean neuropil, computes a maximin baseline on that trial, forms dF/F, smooths with a Gaussian (`sigma=2` samples), replaces non-finite values with zero, and deconvolves with `suite2p.extraction.dcnv.oasis` using a fixed `FRAME_RATE_HZ = 15.5078125` and `tau = 0.7`. It does this trial-by-trial and does not implement the reference code’s `keep_teleports` handling or per-plane `frame_rate / n_planes` logic.

ii.
```python
def compute_dff_and_events(f: np.ndarray, f_neu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrected = f - NEUROPIL_COEF * f_neu
    corrected += NEUROPIL_COEF * np.mean(f_neu, axis=1, keepdims=True)

    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    baseline = maximum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1)
    if not np.isfinite(dff).all():
        dff = np.nan_to_num(dff, copy=False)
    events = dcnv.oasis(dff, 2000, CALCIUM_TAU_S, FRAME_RATE_HZ)
```

iii. The trajectory says the agent wanted to preserve the paper’s dF/F and OASIS pipeline but simplified it to a fixed effective frame rate and per-trial processing, arguing that all sessions share the same native aligned sample spacing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent first keeps only manually curated `iscell` ROIs, then excludes putative interneurons whose session-wide dF/F-to-speed correlation exceeds `0.5`.

ii.
```python
iscell = np.asarray(iscell_table == 1)
...
speed_correlation = correlation_from_sums(n_samples, sx, sx2, sy, sy2, sxy)
keep_neuron = speed_correlation <= 0.5
neural_trials = [np.ascontiguousarray(events[keep_neuron], dtype=np.float32)
                 for events in trial_events]
```

iii. The trajectory explicitly says these are the paper’s curation rules: keep manual `iscell` ROIs and remove speed-correlated putative interneurons. It also notes that the interneuron test should use all valid neural trials, even trials later excluded for bad lick data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial on the same `[start:stop)` window defined from `trial_start` and `teleport`.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    neural_trials = [np.ascontiguousarray(events[keep_neuron], dtype=np.float32)
                     for events in trial_events]
```

iii. The trajectory describes the alignment event as the start of the trial at corridor entry and states that no extra temporal shifting is required beyond splitting trials on those boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native aligned sampling with a fixed bin size of `1000 / 15.5078125 = 64.48 ms`. No temporal rebinning is applied.

ii.
```python
FRAME_RATE_HZ = 15.5078125
...
"time_bin_size": 1000.0 / FRAME_RATE_HZ,
```

iii. In the trajectory, the agent argues that dual-plane metadata are confusing but that the effective aligned sample spacing is the same paper-level `~15.5 Hz` for all sessions, so the conversion should retain that native grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral `position/timestamps` array.

ii.
```python
timestamps = behavior["position/timestamps"][:]
...
time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. The trajectory says the behavioral streams are already aligned, so one consistent behavioral timestamp vector is sufficient for time-from-start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp of that trial is subtracted from every timestamp in the trial.

ii.
```python
time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. The agent gave no extra specialized justification beyond aligning all trial-local time vectors to zero at trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same trial slice indices as the neural data.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
    ...
    trial_events.append(events)
```

iii. The trajectory repeatedly states that the NWB arrays are already sample-aligned across behavior and neural streams, so shared trial indices are enough.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The agent derives environment type primarily from the scene string in the NWB `identifier`, with `environment/data` used only as a consistency check.

ii.
```python
identifier = h5["identifier"][()].decode()
scene = identifier.rsplit("/", 1)[-1]
environments, zones = scene_conditions(scene, len(starts))
...
env_stream = behavior["environment/data"][:]
stream_environment = int(round(float(np.median(env_stream[start:stop]))))
if stream_environment != int(environments[trial]):
    raise ValueError(...)
```

iii. The trajectory shows the agent validating that the scene naming patterns and switch schedules matched the environment stream for all trials, then using the parsed scene schedule as the primary source.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses the scene name with regexes. Static scenes stay constant; switch scenes change after trial 30; the resulting environment code (`0` or `1`) is then repeated across all timepoints in the trial.

ii.
```python
def scene_conditions(scene: str, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    static = re.fullmatch(r"Env([12])_Location([ABC])", scene)
    same_env_switch = re.fullmatch(r"Env([12])_Location([ABC])_to_([ABC])", scene)
    env_switch = re.fullmatch(r"Env([12])_([ABC])_to_Env([12])_([ABC])", scene)
    ...
```
```python
environment = float(environments[trial])
inputs = np.vstack((
    time_from_start,
    np.full(len(pos), environment, dtype=np.float32),
```

iii. The trajectory justifies this by saying the scene metadata cleanly encode the task condition, and that the direct behavior stream check passed.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The agent derives trial number from the stored behavioral `trial number/data` stream, taking the within-trial median.

ii.
```python
trial_number_stream = behavior["trial number/data"][:]
...
trial_number = float(np.median(trial_number_stream[start:stop]))
```

iii. No long rationale is given in the trajectory; the implementation implies the agent considered the stored trial-number stream to be already correct within each trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-trial median is taken once and then broadcast over all timepoints in the trial.

ii.
```python
trial_number = float(np.median(trial_number_stream[start:stop]))
...
np.full(len(pos), trial_number, dtype=np.float32),
```

iii. The trajectory contains no separate justification beyond treating trial number as a per-trial constant.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from `Reward/timestamps` and `reward_zone/data`. The agent first computes a binary outcome for every original trial, then uses the previous trial’s value.

ii.
```python
rzone_stream = behavior["reward_zone/data"][:]
reward_times = behavior["Reward/timestamps"][:]
...
left = np.searchsorted(reward_times, timestamps[start], side="left")
right = np.searchsorted(reward_times, timestamps[stop], side="left")
outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. The trajectory says reward outcome should require both a reward delivery timestamp and an active reward-zone sample, and that previous outcome should refer to the immediately preceding original trial, not merely the preceding retained trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each original trial, the agent marks outcome `1` if a reward timestamp falls within the trial and `reward_zone` is active somewhere in that trial. The previous trial’s binary outcome is then repeated across the current trial; the first trial gets `0`.

ii.
```python
outcomes = np.zeros(len(starts), dtype=np.int8)
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
...
previous_outcome = float(outcomes[trial - 1]) if trial > 0 else 0.0
```

iii. The trajectory explicitly justifies using original-trial outcomes so that excluded lick-failure trials still count when defining “previous trial.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from `position/data` plus reward-zone identity inferred from the scene identifier, using the hard-coded zone bounds in `ZONE_BOUNDS`.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
zone = str(zones[trial])
zone_start, zone_stop = ZONE_BOUNDS[zone]
outputs = np.vstack((
    discretize_distance(pos, zone_start, zone_stop),
```

iii. In the trajectory, the agent chose scene-based reward-zone labels after verifying that scene parsing matched the observed session structure. It did not use the human reference’s Viterbi-style inference from `reward_zone` activity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent computes signed distance to the nearest reward-zone boundary: negative before the zone, zero inside, positive after the zone. It then discretizes that distance into seven categories.

ii.
```python
def discretize_distance(position: np.ndarray, start: float, stop: float) -> np.ndarray:
    distance = np.where(position < start, position - start,
                        np.where(position > stop, position - stop, 0.0))
    ...
```

iii. The trajectory explicitly states this sign convention and says it was chosen to match the decoder target specification.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The categories are implemented manually with thresholds `(-inf, -50)`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `(50, inf)`.

ii.
```python
result[distance < -50] = 0
result[(distance >= -50) & (distance < -10)] = 1
result[(distance >= -10) & (distance < 0)] = 2
result[distance == 0] = 3
result[(distance > 0) & (distance <= 10)] = 4
result[(distance > 10) & (distance <= 50)] = 5
result[distance > 50] = 6
```

iii. The trajectory cites the task specification here rather than any paper-specific processing.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It uses the same per-trial `[start:stop)` slice as neural activity.

ii.
```python
for trial in retained_trial_indices:
    start, stop = starts[trial], stops[trial]
    pos = np.asarray(position[start:stop], dtype=np.float32)
    ...
    discretize_distance(pos, zone_start, zone_stop)
```

iii. The agent’s general alignment rationale was that behavioral and neural arrays are already synchronized sample-by-sample in the NWB files.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `position/data`.

ii.
```python
position = behavior["position/data"][:]
...
pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. The trajectory treats position as a direct behavioral stream on the common time grid.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent takes the raw position trace within each trial and discretizes it into five bins with `np.digitize`.

ii.
```python
np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. The trajectory gives no extra justification beyond matching the decoder specification.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are `90`, `180`, `270`, and `360` cm, yielding five bins.

ii.
```python
np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. The trajectory follows the task’s requested equal-width 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned with neural activity by using the same trial indices.

ii.
```python
start, stop = starts[trial], stops[trial]
pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. The agent’s alignment justification is the same as for other behavioral traces: shared sample indexing in the NWB arrays.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `lick/data`.

ii.
```python
lick = behavior["lick/data"][:]
...
(lick[start:stop] > 0).astype(np.int8),
```

iii. The trajectory treats lick as a behavioral channel already aligned to the neural grid.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick values are binarized: values greater than zero become `1`, otherwise `0`.

ii.
```python
(lick[start:stop] > 0).astype(np.int8),
```

iii. The trajectory justifies excluding failed lick-sensor trials, but for retained trials it simply binarizes lick presence.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same per-trial indices as the neural data.

ii.
```python
start, stop = starts[trial], stops[trial]
...
(lick[start:stop] > 0).astype(np.int8),
```

iii. The agent’s general alignment argument is that all retained streams share the same underlying trial slices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the scene identifier parsed into zone labels, not from direct trial-wise inference on `reward_zone/data`.

ii.
```python
scene = identifier.rsplit("/", 1)[-1]
environments, zones = scene_conditions(scene, len(starts))
...
zone_to_class = {"A": 0, "B": 1, "C": 2}
np.full(len(pos), zone_to_class[zone], dtype=np.int8),
```

iii. The trajectory indicates the agent trusted the scene metadata once it had checked that the encoded schedule matched the behavioral environment stream.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into a per-trial zone schedule, including switches after trial 30 where encoded, then mapped from `A/B/C` to `0/1/2` and repeated across the whole trial.

ii.
```python
elif same_env_switch:
    environments = np.full(n_trials, int(same_env_switch.group(1)) - 1, dtype=np.int8)
    zones = np.where(np.arange(n_trials) < 30,
                     same_env_switch.group(2), same_env_switch.group(3))
...
np.full(len(pos), zone_to_class[zone], dtype=np.int8),
```

iii. The agent’s justification in the trajectory is that the scene metadata already encode the zone schedule deterministically, making extra inference unnecessary.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, combined with `reward_zone/data` as an additional requirement that the reward zone be active during the trial.

ii.
```python
reward_times = behavior["Reward/timestamps"][:]
rzone_stream = behavior["reward_zone/data"][:]
...
outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. The trajectory explicitly says the agent matched `behavior.get_trial_types` by requiring both a reward delivery timestamp and reward-zone occupancy.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the agent checks whether any reward timestamp falls within the trial’s time bounds and whether `reward_zone` is active during that trial. The resulting binary value is then repeated across all timepoints in the trial.

ii.
```python
left = np.searchsorted(reward_times, timestamps[start], side="left")
right = np.searchsorted(reward_times, timestamps[stop], side="left")
outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
...
np.full(len(pos), outcomes[trial], dtype=np.int8),
```

iii. The trajectory’s justification is that this matches the source repository’s trial-type logic more closely than checking reward timestamps alone.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly validates and fails fast rather than repairing data. It raises on invalid trial boundaries, raises if scene-derived environment disagrees with the environment stream, replaces non-finite dF/F values with zero, excludes bad lick-sensor trials, and raises if too few trials remain. It does not implement the human reference’s cropping of neural/behavior length mismatches or special handling for missing reward-zone observations.

ii.
```python
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial boundaries in {path}")
...
if stream_environment != int(environments[trial]):
    raise ValueError(f"Scene/environment mismatch in {path}, trial {trial}")
...
if not np.isfinite(dff).all():
    dff = np.nan_to_num(dff, copy=False)
...
if len(neural_trials) < 2:
    raise ValueError(f"Fewer than two retained trials in {path}")
```

iii. The trajectory focuses on data cleanliness checks and documented lick-sensor exclusions, but it does not describe any broader imputation or repair strategy.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeated HDF5 reads of fluorescence and neuropil for every trial and plane, repeated dF/F plus OASIS deconvolution for every retained trial, and the full-session conversion over all 152 NWB files.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
    f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
    dff, events = compute_dff_and_events(f, f_neu)
```

iii. The trajectory confirms this indirectly: the agent repeatedly discusses full-dataset runtime, per-session conversion cost, and finishing a long end-to-end 152-session conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session`, the per-plane loop in `read_curated_trial`, and parts of the per-trial output assembly could be vectorized or batched. The current implementation repeatedly slices and processes each trial independently.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
for plane_name in sorted(series.keys(), key=lambda value: int(value.removeprefix("plane"))):
    ...
for trial in retained_trial_indices:
    ...
```

iii. The trajectory does not call this out explicitly, but the implementation shows a deliberately simple trial-wise structure chosen for tractability and correctness.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly reads contiguous slabs from both fluorescence and neuropil for every trial, recomputes dF/F and OASIS on every trial separately, and repeatedly slices the same behavioral arrays trial-by-trial when assembling inputs and outputs.

ii.
```python
f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
dff, events = compute_dff_and_events(f, f_neu)
...
pos = np.asarray(position[start:stop], dtype=np.float32)
spd = np.asarray(speed[start:stop], dtype=np.float32)
```

iii. The trajectory’s discussion of runtime and memory indicates the agent knowingly accepted repeated per-trial work to keep peak memory manageable.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes full dF/F traces only to derive speed-correlation interneuron masks and then discards those dF/F arrays. It also performs environment-stream consistency checks and stores extensive per-session metadata that the downstream decoder does not need for training.

ii.
```python
dff, events = compute_dff_and_events(f, f_neu)
...
dff64 = np.asarray(dff, dtype=np.float64)
sx += np.sum(dff64, axis=1)
...
info = {
    "source_file": ...,
    "scene": scene,
    ...
}
```

iii. The trajectory does not explicitly label these as unnecessary, but it shows the agent prioritizing auditability and source-faithful curation even when some intermediate products are not preserved in the final decoder inputs.
