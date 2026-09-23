# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from all `sub-*` subdirectories under the data root using `pathlib.Path.glob("sub-*/*.nwb")`. Files are sorted by mouse and session number using a regex-based natural sort key. Each NWB file is opened with `h5py` (not `pynwb`). All behavioral and neural data streams are read directly from the HDF5 hierarchy.

ii.
```python
files = sorted(data_root.glob("sub-*/*.nwb"), key=natural_key)
# ...
with h5py.File(path, "r") as h5:
    behavior = h5["processing/behavior/BehavioralTimeSeries"]
```

iii. The agent stated: "I'll trace the paper's own preprocessing pipeline first, then map its trial-aligned arrays into the required schema." The agent systematically listed all files and explored the NWB structure before writing the conversion code.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file paths by parsing the `sub-m<N>` pattern with regex, then deduplicated and sorted numerically. Subject IDs are stored as strings like `"m3"`, `"m4"`, etc.

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    return int(match.group(1)), int(match.group(2))

subjects = sorted({f"m{natural_key(path)[0]}" for path in files},
                  key=lambda value: int(value[1:]))
```

iii. The subject ID is also read from inside the NWB file (`h5["general/subject/subject_id"]`) and used for the session info metadata, but the subject list itself is derived from file paths.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by parsing the `ses-<N>` part of the filename.

ii.
```python
files = sorted(data_root.glob("sub-*/*.nwb"), key=natural_key)
for number, path in enumerate(files, start=1):
    session_neural, session_input, session_output, info = convert_session(path)
```

iii. The agent recognized that each NWB file represents one mouse/day recording session.

## 1-d. How are the data split into trials?

i. Trial starts are identified by `trial_start > 0` and trial ends by `teleport > 0`. Trials span from `trial_start` (inclusive) to `teleport` (exclusive).

ii.
```python
starts = np.flatnonzero(behavior["trial_start/data"][:] > 0)
stops = np.flatnonzero(behavior["teleport/data"][:] > 0)
# ...
for trial, (start, stop) in enumerate(zip(starts, stops)):
    # data[start:stop] for each trial
```

iii. The agent's docstring states: "Trials run from each `trial_start` sample up to (not including) its matching `teleport` sample." Note: the AI uses `teleport > 0` (all positive teleport samples), while the reference uses teleport onset detection (`(teleport[1:] > 0) & (teleport[:-1] <= 0)`). This means the AI's `stops` will include ALL positive teleport samples, not just the first one of each teleport period, potentially causing issues if teleport remains positive for multiple samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick sensor quality. A trial is excluded if more than 30% of its samples have a cumulative lick count > 2, which the paper identifies as 81 stuck lick-sensor trials. No minimum trial length filter is applied.

ii.
```python
LICK_ERROR_FRACTION = 0.30
# ...
trial_lick = lick[start:stop]
bad_lick_sensor = np.mean(trial_lick > 2) > LICK_ERROR_FRACTION
# ...
if bad_lick_sensor:
    excluded_lick_trials += 1
    continue
```

iii. The agent stated: "I've now reproduced the paper's trial-level quality rule exactly: 81 of 12,216 trials have the documented stuck lick-sensor signature (>30% of samples with cumulative lick count >2). Because lick is a required categorical target and cannot be represented as NaN, those 81 trials will be excluded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `Fluorescence` and `Neuropil` series in the NWB file, not from the pre-computed `Deconvolved` field. The AI reimplements the paper's own dF/F and deconvolution pipeline.

ii.
```python
f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
dff, events = compute_dff_and_events(f, f_neu)
```

iii. The agent's docstring states: "Neural activity is recomputed from ROI and neuropil fluorescence with the paper's pipeline."

## 2-b. How is the `neural` data processed?

i. The processing follows the paper's pipeline: (1) neuropil subtraction with coefficient 0.7, (2) add back trial-mean neuropil, (3) Gaussian smoothing with sigma=15 for baseline, (4) 300-sample minimum filter then maximum filter (maximin baseline), (5) dF/F = (corrected - baseline) / |baseline|, (6) Gaussian smoothing with sigma=2, (7) OASIS deconvolution with tau=0.7 at 15.5 Hz frame rate.

However, the AI processes each trial independently (reads trial slabs individually), whereas the reference processes the full session with the paper's `dff()` function which handles keep_teleports logic to allow the baseline window to span inter-trial periods on certain sessions.

ii.
```python
def compute_dff_and_events(f: np.ndarray, f_neu: np.ndarray):
    corrected = f - NEUROPIL_COEF * f_neu
    corrected += NEUROPIL_COEF * np.mean(f_neu, axis=1, keepdims=True)
    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    baseline = maximum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1)
    events = dcnv.oasis(dff, 2000, CALCIUM_TAU_S, FRAME_RATE_HZ)
    return dff, np.asarray(events, dtype=np.float32)
```

iii. The agent stated in the docstring: "0.7 neuropil subtraction, trial-local 20 s maximin baseline, dF/F, Gaussian smoothing (2 imaging samples), and Suite2p OASIS deconvolution with a 0.7 s calcium time constant."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) only manually curated ROIs (`iscell == 1`) are retained, (2) putative interneurons with dF/F-speed Pearson correlation > 0.5 are excluded. The interneuron test uses all valid neural trials including bad-lick trials (corrected in a late revision).

ii.
```python
iscell_table = h5["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][:, 0]
iscell = np.asarray(iscell_table == 1)
# ...
speed_correlation = correlation_from_sums(n_samples, sx, sx2, sy, sy2, sxy)
keep_neuron = speed_correlation <= 0.5
neural_trials = [np.ascontiguousarray(events[keep_neuron], dtype=np.float32)
                 for events in trial_events]
```

iii. The agent stated: "One final audit caught a subtle curation-order issue: bad-lick trials must be excluded from the decoder, but their valid neural activity should still contribute to the paper's session-wide dF/F-speed interneuron test."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing each trial from the `trial_start` index to the `teleport` index. No additional shifting is needed since the trial start IS the alignment event.

ii.
```python
f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
```

iii. The alignment is implicit in the trial slicing approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native sampling rate of ~15.5 Hz (64.48 ms per frame) is preserved. A fixed frame rate constant `FRAME_RATE_HZ = 15.5078125` is used.

ii.
```python
FRAME_RATE_HZ = 15.5078125
# ...
"time_bin_size": 1000.0 / FRAME_RATE_HZ,
```

iii. The agent's docstring states: "Native aligned samples are retained (~15.5 Hz, 64.48 ms); trials may consequently have different durations."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the behavior data.

ii.
```python
timestamps = behavior["position/timestamps"][:]
# ...
time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. The timestamps are read from the position series, which shares the same time base as all other behavioral variables.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in the trial.

ii.
```python
time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. Standard approach to compute time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same sample indices within each trial (both indexed by `start:stop`), so no additional alignment is needed.

ii.
```python
# Same start:stop used for both neural and behavioral data
f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. Implicit alignment through shared indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the NWB file's `identifier` field, which contains the scene/condition name (e.g., "Env1_LocationA", "Env1_LocationA_to_B", "Env1_A_to_Env2_B"). This is parsed by `scene_conditions()` to determine environment type. The `environment` data stream is used only as a cross-check.

ii.
```python
identifier = h5["identifier"][()].decode()
scene = identifier.rsplit("/", 1)[-1]
environments, zones = scene_conditions(scene, len(starts))
# ...
stream_environment = int(round(float(np.median(env_stream[start:stop]))))
if stream_environment != int(environments[trial]):
    raise ValueError(f"Scene/environment mismatch in {path}, trial {trial}")
```

iii. The agent parses the scene name to determine environment and handles three patterns: static conditions, same-environment switches, and cross-environment switches (with switch at trial 30).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed with regex patterns to extract environment type (0 or 1, corresponding to ENV1/ENV2). On switch days, the first 30 trials use the pre-switch environment and subsequent trials use the post-switch environment. The value is constant across timepoints within a trial.

ii.
```python
def scene_conditions(scene: str, n_trials: int):
    static = re.fullmatch(r"Env([12])_Location([ABC])", scene)
    same_env_switch = re.fullmatch(r"Env([12])_Location([ABC])_to_([ABC])", scene)
    env_switch = re.fullmatch(r"Env([12])_([ABC])_to_Env([12])_([ABC])", scene)
    # ...
    if env_switch:
        environments = np.where(np.arange(n_trials) < 30,
                                int(env_switch.group(1)) - 1,
                                int(env_switch.group(3)) - 1).astype(np.int8)
```

iii. The scene parsing approach derives environment from metadata rather than from the per-sample behavioral stream.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral data stream stored in the NWB file.

ii.
```python
trial_number_stream = behavior["trial number/data"][:]
# ...
trial_number = float(np.median(trial_number_stream[start:stop]))
```

iii. The AI uses the stored trial number variable, taking the median value within each trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median of the `trial number` stream values within the trial's time range is taken as the trial number. The value is constant across all timepoints within a trial.

ii.
```python
trial_number = float(np.median(trial_number_stream[start:stop]))
# ...
np.full(len(pos), trial_number, dtype=np.float32),
```

iii. Using the median handles any edge effects at trial boundaries.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` and `reward_zone/data`. A trial is considered rewarded if both a reward delivery timestamp falls within the trial's time range AND the `reward_zone` stream is active (>0) during the trial.

ii.
```python
reward_times = behavior["Reward/timestamps"][:]
rzone_stream = behavior["reward_zone/data"][:]
# ...
outcomes = np.zeros(len(starts), dtype=np.int8)
for trial, (start, stop) in enumerate(zip(starts, stops)):
    left = np.searchsorted(reward_times, timestamps[start], side="left")
    right = np.searchsorted(reward_times, timestamps[stop], side="left")
    outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. The agent stated: "Reward outcome requires both a reward delivery timestamp and an active reward-zone sample, matching behavior.get_trial_types."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are computed for ALL original trials before any quality filtering. The previous trial outcome for trial `t` is `outcomes[t-1]`. For the first trial, the value is 0. This refers to the immediately preceding original trial (not merely the preceding retained trial). The value is constant per trial.

ii.
```python
previous_outcome = float(outcomes[trial - 1]) if trial > 0 else 0.0
# ...
np.full(len(pos), previous_outcome, dtype=np.float32),
```

iii. The agent stated: "Previous outcome refers to the immediately preceding original trial (not merely the preceding retained trial)."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral data stream and the reward zone boundaries determined by the `scene_conditions()` function (parsed from the NWB file identifier). The reward zone bounds are hard-coded as `ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}`.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
# ...
zone = str(zones[trial])
zone_start, zone_stop = ZONE_BOUNDS[zone]
```

iii. The agent determines which reward zone is active for each trial from the scene name parsing, with the switch happening at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before the zone, zero inside the zone, positive after. The distance is to the nearest zone boundary.

ii.
```python
def discretize_distance(position, start, stop):
    distance = np.where(position < start, position - start,
                        np.where(position > stop, position - stop, 0.0))
    result = np.empty(distance.shape, dtype=np.int8)
    result[distance < -50] = 0
    result[(distance >= -50) & (distance < -10)] = 1
    result[(distance >= -10) & (distance < 0)] = 2
    result[distance == 0] = 3
    result[(distance > 0) & (distance <= 10)] = 4
    result[(distance > 10) & (distance <= 50)] = 5
    result[distance > 50] = 6
    return result
```

iii. The computation and discretization are combined in one function.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean indexing: <-50, [-50,-10), [-10,0), ==0, (0,10], (10,50], >50.

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

iii. The bin edges match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same sample indices (`start:stop`) are used for both neural and position data within each trial, so no additional alignment is needed.

ii.
```python
pos = np.asarray(position[start:stop], dtype=np.float32)
# Same start:stop as neural data
```

iii. Implicit alignment through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral data stream.

ii.
```python
position = behavior["position/data"][:]
# ...
pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. Direct use of the stored position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond discretization. Position is directly digitized into 5 bins using edges [90, 180, 270, 360].

ii.
```python
np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. The bin edges correspond to 5 equal-sized 90 cm bins spanning the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized using `np.digitize` with edges [90, 180, 270, 360], producing bins 0-4. `np.digitize` returns 0 for values below the first edge and N for values above the last edge, directly producing the 0-indexed categories.

ii.
```python
np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. Note: `np.digitize` with edges `[90, 180, 270, 360]` returns values in range [0, 4], which matches the required 5 bins. This differs from the reference which uses `[-inf, 90, 180, 270, 360, inf]` with a `-1` correction.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same sample indices as the neural data within each trial.

ii. Same indexing as neural: `pos = np.asarray(position[start:stop], ...)`

iii. Implicit alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral data stream.

ii.
```python
lick = behavior["lick/data"][:]
# ...
(lick[start:stop] > 0).astype(np.int8),
```

iii. Direct use of the stored lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any lick value > 0 is mapped to 1, otherwise 0.

ii.
```python
(lick[start:stop] > 0).astype(np.int8),
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same sample indices as the neural data within each trial.

ii. `lick[start:stop]` uses the same indices as neural data.

iii. Implicit alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` field (scene name), parsed by `scene_conditions()`. NOT derived from the `reward_zone` behavioral data stream.

ii.
```python
identifier = h5["identifier"][()].decode()
scene = identifier.rsplit("/", 1)[-1]
environments, zones = scene_conditions(scene, len(starts))
# ...
zone = str(zones[trial])
zone_to_class = {"A": 0, "B": 1, "C": 2}
np.full(len(pos), zone_to_class[zone], dtype=np.int8),
```

iii. The scene name encodes both the environment and reward zone location. On switch days, the switch occurs at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed with regex to extract the zone letter (A, B, or C). On switch days, the first 30 trials get the pre-switch zone and subsequent trials get the post-switch zone. Zone letters are mapped to integers: A=0, B=1, C=2.

ii.
```python
def scene_conditions(scene, n_trials):
    # Parses patterns like "Env1_LocationA", "Env1_LocationA_to_B", "Env1_A_to_Env2_B"
    if same_env_switch:
        zones = np.where(np.arange(n_trials) < 30,
                         same_env_switch.group(2), same_env_switch.group(3))
```

iii. The switch at trial 30 is hard-coded based on the paper's experimental protocol.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` and `reward_zone/data`. A trial is rewarded if both a reward timestamp falls within the trial time range AND the reward zone was active during the trial.

ii.
```python
outcomes = np.zeros(len(starts), dtype=np.int8)
for trial, (start, stop) in enumerate(zip(starts, stops)):
    left = np.searchsorted(reward_times, timestamps[start], side="left")
    right = np.searchsorted(reward_times, timestamps[stop], side="left")
    outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. The agent stated this matches `behavior.get_trial_types` from the paper's code.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are searched to find if any fall within the trial's time range (using `searchsorted`). The trial is marked as rewarded only if both a reward timestamp is present AND the reward zone was active. The value is constant per trial.

ii.
```python
outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
# ...
np.full(len(pos), outcomes[trial], dtype=np.int8),
```

iii. The dual condition (reward timestamp + active reward zone) follows the paper's `get_trial_types` logic.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Lick sensor failures**: 81 trials with stuck lick sensors (>30% of samples with lick count >2) are excluded.
- **NaN/Inf in dF/F**: If non-finite values appear in dF/F (from degenerate baselines), they are replaced with zero via `np.nan_to_num`.
- **Trial boundary validation**: An assertion checks that the number of starts equals the number of stops and all stops are after their corresponding starts.
- **Environment cross-check**: The parsed environment from the scene name is validated against the stored environment stream.

ii.
```python
if not np.isfinite(dff).all():
    dff = np.nan_to_num(dff, copy=False)
# ...
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial boundaries in {path}")
```

iii. These are defensive checks found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Reading NWB/HDF5 data** for each trial individually (I/O bound) - the code reads fluorescence and neuropil data trial-by-trial rather than loading the full session array once.
2. **dF/F computation and OASIS deconvolution** for each trial independently.
3. **Pickle serialization** of the full dataset.

ii. N/A

iii. Reading trial slabs from HDF5 is repeated for every trial, which is less efficient than reading the full session array once.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial, reading neural data from disk and computing dF/F independently for each trial. The interneuron correlation computation is already vectorized using streaming sufficient statistics. The discretization operations (distance, position, speed) could be applied to full-session arrays before splitting.

ii. N/A

iii. The per-trial disk read is the main bottleneck that could be improved by reading full-session arrays.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file only once (no survey step), so there is no repeated file reading. However, the `read_curated_trial` function is called twice per trial (once for Fluorescence, once for Neuropil), each time re-computing the plane-based ROI selection logic.

ii. N/A

iii. The per-trial reading approach means disk I/O is repeated many times rather than loading the full session array once.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `dff` (the smoothed dF/F trace) for each trial but only uses the `events` (deconvolved signal) for the final output. The `dff` is needed for the interneuron correlation check but is otherwise discarded. The `trial_number_stream` is read but only the median is used per trial, which is essentially just the stored trial number.

ii.
```python
dff, events = compute_dff_and_events(f, f_neu)
# dff is used for interneuron correlation but events are what goes into neural output
```

iii. The dF/F must be computed as an intermediate step for deconvolution, so this is not truly unnecessary.
