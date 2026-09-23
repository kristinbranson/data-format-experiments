# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files under the data root by globbing for `sub-*/*_behavior+ophys.nwb`, sorts them by natural key (subject number, session number), and opens each with `h5py`. All subjects, sessions, and trials within each NWB file are included.

ii.
```python
def build_dataset(data_root: Path, max_sessions: int | None = None) -> dict:
    files = sorted(data_root.glob("sub-*/*_behavior+ophys.nwb"), key=_natural_key)
    ...
    for index, path in enumerate(files, start=1):
        arrays, info = convert_session(path)
```

```python
def convert_session(path: Path) -> tuple[dict, dict]:
    with h5py.File(path, "r") as nwb:
        behavior = nwb[BEHAVIOR]
        starts, stops = _trial_bounds(behavior)
        ...
```

iii. The AI identified 152 sessions from 11 mice in the data release. It uses `h5py` rather than `pynwb` to load NWB files. The glob pattern captures all NWB files across all subject directories.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. A unique sorted list of subject IDs is built after processing all sessions.

ii.
```python
subject = _decode_scalar(nwb["general/subject/subject_id"][()])
...
subjects = sorted(set(session_subjects), key=lambda value: int(re.search(r"\d+", value).group()))
subject_lookup = {subject: index for index, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[s] for s in session_subjects], dtype=np.int64)
```

iii. Subject IDs are read from the NWB metadata, ensuring correct identification even if directory names were inconsistent.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted by (subject number, session number) via `_natural_key`.

ii.
```python
def _natural_key(path: Path):
    subject = re.search(r"sub-m(\d+)", str(path))
    session = re.search(r"ses-(\d+)", path.name)
    return (int(subject.group(1)), int(session.group(1)))

files = sorted(data_root.glob("sub-*/*_behavior+ophys.nwb"), key=_natural_key)
```

iii. The AI noted the NWB files are organized one-per-session and sorted them for deterministic ordering.

## 1-d. How are the data split into trials?

i. Trial boundaries are found by detecting nonzero values in `trial_start` (for starts) and nonzero values in `teleport` (for stops). The on-track interval from `start` to `stop` (exclusive) is one trial.

ii.
```python
def _trial_bounds(behavior) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(_read_behavior(behavior, "trial_start") > 0)
    stops = np.flatnonzero(_read_behavior(behavior, "teleport") > 0)
    if len(starts) != len(stops):
        raise ValueError(f"Unmatched trial starts ({len(starts)}) and teleports ({len(stops)})")
    if np.any(stops <= starts):
        raise ValueError("A teleport did not follow its corresponding trial start")
    return starts, stops
```

iii. The AI uses `trial_start` and `teleport` signals to define trials. It validates that starts and stops are properly paired and ordered.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using the paper's faulty lick sensor criterion: trials where >30% of frame samples have cumulative lick count >2 are excluded. There is no minimum trial length filter.

ii.
```python
def _bad_lick_trials(lick: np.ndarray, starts: np.ndarray,
                     stops: np.ndarray) -> np.ndarray:
    """Paper criterion: >30% of frame samples have cumulative lick count >2."""
    return np.asarray([
        np.mean(lick[start:stop] > 2) > 0.30
        for start, stop in zip(starts, stops)
    ], dtype=bool)
```

```python
bad_lick = _bad_lick_trials(lick, starts, stops)
...
for trial, (start, stop) in enumerate(zip(starts, stops)):
    if bad_lick[trial]:
        continue
```

iii. The AI identified 81 faulty-lick trials across the dataset using the paper's published criterion. It stated this ensures valid lick targets for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Fluorescence` and `Neuropil` response series in the NWB ophys processing module. The AI explicitly did NOT use the NWB `Deconvolved` field.

ii.
```python
fluorescence_group = nwb[f"{OPHYS}/Fluorescence"]
neuropil_group = nwb[f"{OPHYS}/Neuropil"]
for plane_name in sorted(fluorescence_group.keys()):
    ...
    plane_specs.append((
        fluorescence_group[f"{plane_name}/data"],
        neuropil_group[f"{plane_name}/data"],
        local_keep,
    ))
```

iii. The AI recognized that the NWB Deconvolved field is suite2p's own deconvolution, not the paper's signal, and instead recomputed events from raw fluorescence and neuropil traces.

## 2-b. How is the `neural` data processed?

i. The AI performs per-trial processing: neuropil subtraction (0.7 coefficient) with trial-mean neuropil restoration, maximin baseline (Gaussian smoothing sigma=15, 300-sample min/max filters), dF/F computation, 2-sample Gaussian smoothing, and OASIS deconvolution (tau=0.7). However, the processing is done trial-by-trial (each trial independently) rather than using the paper's full-session windowing with `keep_teleports` logic.

ii.
```python
def _dff_and_events(fluorescence: np.ndarray, neuropil: np.ndarray,
                    rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    corrected = fluorescence - 0.7 * neuropil
    corrected += 0.7 * neuropil.mean(axis=1, keepdims=True)
    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, 300, axis=1)
    baseline = maximum_filter1d(baseline, 300, axis=1)
    dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1).astype(np.float32, copy=False)
    events = dcnv.oasis(dff, 2000, 0.7, rate_hz)
    return dff, events
```

Called per trial:
```python
for start, stop in zip(starts, stops):
    fluorescence = np.concatenate([...])
    neuropil = np.concatenate([...])
    dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
```

iii. The AI described this as "paper-matched neural preprocessing." The parameters match the paper. However, the AI does NOT handle the `keep_teleports` sessions (where the laser was not blanked between trials, allowing the baseline window to span inter-trial intervals).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only manually curated ROIs (`iscell == True`) are retained. (2) Putative interneurons with dF/F-speed Pearson correlation > 0.5 are excluded. The speed correlation is computed session-wide using running sums across all trials.

ii.
```python
is_cell = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation/iscell"][:, 0].astype(bool)
roi_indices = np.flatnonzero(is_cell)
...
speed_correlation = np.divide(numerator, denominator, ...)
putative_interneuron = speed_correlation > 0.5
keep_neuron = ~putative_interneuron
```

iii. The AI correctly identified both the paper's manual curation step and the interneuron exclusion criterion (r > 0.5 with running speed).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Each trial's neural data starts at the `trial_start` index and ends at the `teleport` index, requiring no additional alignment beyond the trial boundary extraction.

ii.
```python
for start, stop in zip(starts, stops):
    fluorescence = np.concatenate([
        np.asarray(f_data[start:stop, :], ...)
        ...
    ])
```

iii. The AI correctly aligns to trial start as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the original sampling rate (~15.5 Hz, 64.48 ms bins). No temporal rebinning is applied. The rate is computed from the median of behavior timestamp differences.

ii.
```python
NOMINAL_RATE_HZ = 15.5078125
...
def _sampling_rate(behavior) -> float:
    timestamps = np.asarray(behavior["position/timestamps"])
    rate = float(1.0 / np.median(np.diff(timestamps)))
    if not np.isclose(rate, NOMINAL_RATE_HZ, rtol=0, atol=1e-4):
        raise ValueError(f"Unexpected aligned sampling rate: {rate} Hz")
    return rate
```

```python
"time_bin_size": 1000.0 / NOMINAL_RATE_HZ,
```

iii. The AI verified that all sessions share the same effective sampling rate of 15.5078125 Hz, accounting for two-plane sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the trial's timepoint index and the sampling rate. It does NOT use stored timestamps directly.

ii.
```python
input_trial = np.vstack([
    np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
    ...
])
```

iii. The AI computes time as sample index divided by the sampling rate, rather than using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A time array is generated as `np.arange(timepoints) / rate_hz`, producing time in seconds from trial start (0, dt, 2*dt, ...).

ii.
```python
np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
```

iii. This is a simple computation from sample index and rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same sample indices (start:stop), so alignment is automatic.

ii. The same `start, stop` trial bounds index both neural and behavioral data.

iii. Neural and behavioral data share the same time axis in the NWB file.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = _read_behavior(behavior, "environment")
...
trial_env = int(np.rint(np.median(environment[start:stop])))
```

iii. The environment variable records the VR environment type at each timepoint.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial median of the `environment` variable is taken and rounded to the nearest integer. This value is broadcast to all timepoints in the trial.

ii.
```python
trial_env = int(np.rint(np.median(environment[start:stop])))
...
np.full(timepoints, trial_env, dtype=np.float32),
```

iii. Taking the median handles any edge effects at trial boundaries. Environment is constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series stored in the NWB file.

ii.
```python
trial_number = _read_behavior(behavior, "trial number")
...
source_trial_number = float(np.median(trial_number[start:stop]))
```

iii. The AI uses the NWB's stored trial number rather than a loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The per-trial median of the `trial number` variable is taken and used as a float, broadcast to all timepoints.

ii.
```python
source_trial_number = float(np.median(trial_number[start:stop]))
...
np.full(timepoints, source_trial_number, dtype=np.float32),
```

iii. Using the median ensures robustness to any noise at trial boundaries.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and behavior `position/timestamps`. Trial outcomes are computed by checking if any reward timestamp falls within the trial's time range.

ii.
```python
def _trial_outcomes(reward_timestamps: np.ndarray, timestamps: np.ndarray,
                    starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
        right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
        outcomes[trial] = right > left
    return outcomes
```

iii. Reward timestamps are not aligned with behavior timestamps so searchsorted is used.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's outcome (0 or 1) is used. For the first trial, it is set to 0. The value is constant across all timepoints.

ii.
```python
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
...
np.full(timepoints, previous_outcome, dtype=np.float32),
```

iii. Standard lookback to the previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and the inferred reward zone location. The zone is identified from the `reward_zone` behavior signal and `position` using a per-condition modal assignment strategy.

ii.
```python
zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
...
def _distance_class(position: np.ndarray, zone: int) -> np.ndarray:
    start, end = ZONE_STARTS[zone], ZONE_ENDS[zone]
    distance = np.where(position < start, position - start,
                        np.where(position > end, position - end, 0.0))
```

iii. The AI uses the paper's reward zone boundaries (A: 80-130, B: 200-250, C: 320-370 cm).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance is computed: negative before the zone, 0 inside, positive after. This is then discretized into 7 categories.

ii.
```python
distance = np.where(position < start, position - start,
                    np.where(position > end, position - end, 0.0))
return np.select(
    [distance < -50, distance < -10, distance < 0, distance == 0,
     distance <= 10, distance <= 50],
    [0, 1, 2, 3, 4, 5], default=6,
).astype(np.int8)
```

iii. The signed distance and bin edges match the instruction specification.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using `np.select` with conditions: `< -50`, `< -10`, `< 0`, `== 0`, `<= 10`, `<= 50`, else `> 50`.

ii.
```python
return np.select(
    [distance < -50, distance < -10, distance < 0, distance == 0,
     distance <= 10, distance <= 50],
    [0, 1, 2, 3, 4, 5], default=6,
).astype(np.int8)
```

iii. The bin edges match the instructions' specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same start:stop indices used for both neural and behavioral data, so alignment is automatic.

ii. `position[start:stop]` uses the same indices as the neural data extraction.

iii. Both streams share the same time axis.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = _read_behavior(behavior, "position")
...
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. The position variable directly records the animal's corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond discretization. Position is discretized into 5 bins with edges at 90, 180, 270, 360 cm.

ii.
```python
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. The 5 equal-sized bins span the 450 cm track (90 cm each).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with bin edges `[90, 180, 270, 360]` produces bins 0-4.

ii.
```python
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. Note that `np.digitize` with edges `[90, 180, 270, 360]` returns values 0 through 4 (0 for < 90, 1 for 90-180, etc.), which matches the specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same start:stop indices as neural data.

ii. `position[start:stop]` indexed by the same trial bounds.

iii. Automatic alignment via shared time axis.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = _read_behavior(behavior, "lick")
...
lick_binary = (lick > 0).astype(np.int8)
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any value > 0 is mapped to 1, otherwise 0.

ii.
```python
lick_binary = (lick > 0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same start:stop indices as neural data.

ii. `lick[start:stop]` indexed by the same trial bounds.

iii. Automatic alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavior time series, using the `_reward_zone_labels` function. The function uses the median position when the reward zone signal is nonzero to determine which zone (A/B/C) is active, then takes the mode within each experimental condition (pre-switch vs post-switch on switch days).

ii.
```python
def _reward_zone_labels(position, rzone, starts, stops, experiment_day):
    observed = np.full(len(starts), -1, dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        mask = rzone[start:stop] > 0
        if np.any(mask):
            event_position = float(np.median(position[start:stop][mask]))
            observed[trial] = int(np.argmin(np.abs(ZONE_CENTERS - event_position)))
    boundaries = [0, min(30, len(starts)), len(starts)] if experiment_day in SWITCH_DAYS else [0, len(starts)]
    labels = np.empty(len(starts), dtype=np.int8)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        valid = observed[left:right]
        valid = valid[valid >= 0]
        labels[left:right] = np.bincount(valid, minlength=3).argmax()
    return labels
```

iii. The AI uses median event position matched to zone centers, then modal assignment per condition block, accounting for switch days.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. See 10-a. The per-trial observed zone is determined from median position when `reward_zone > 0`, matched to the closest zone center. Then the mode within each condition block (pre-switch or post-switch) is assigned to all trials in that block. Trials without reward zone activity (omission trials) inherit the block's mode. The output is encoded as 0=A, 1=B, 2=C.

ii.
```python
output_trial = _output_matrix(
    position[start:stop], speed[start:stop], lick[start:stop],
    int(zones[trial]), int(outcomes[trial]),
)
...
np.full(timepoints, zone, dtype=np.int8),
```

iii. The condition-block approach handles reward omission trials cleanly.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and behavior timestamps.

ii.
```python
outcomes = _trial_outcomes(
    np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
)
```

iii. Reward events have their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time range using `searchsorted`. Output is 1 if rewarded, 0 otherwise. The value is constant for all timepoints in the trial.

ii.
```python
def _trial_outcomes(reward_timestamps, timestamps, starts, stops):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
        right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
        outcomes[trial] = right > left
    return outcomes
```

iii. Using searchsorted on reward timestamps against the trial's time range.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several checks and handling strategies:
- **Trial start/teleport mismatch**: An error is raised if the counts don't match or if a teleport precedes its start.
- **Missing reward zone data**: Trials where `reward_zone` is never active get assigned the modal zone from their condition block.
- **Non-finite neural activity**: An error is raised if dF/F or events contain non-finite values after processing.
- **No curated cells**: An error is raised if a session has no `iscell` ROIs.
- **All interneurons**: An error is raised if every cell is classified as an interneuron.

ii.
```python
if len(starts) != len(stops):
    raise ValueError(...)
if np.any(stops <= starts):
    raise ValueError(...)
if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(events)):
    raise ValueError(...)
if len(roi_indices) == 0:
    raise ValueError(...)
if not np.any(keep_neuron):
    raise ValueError(...)
```

iii. The AI uses strict error-raising rather than graceful degradation for most edge cases.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with `h5py` (I/O bound, large files)
2. Computing dF/F and OASIS deconvolution per trial (CPU-bound, runs scipy filters and dcnv.oasis)
3. Computing speed correlation for interneuron screening (involves all-trial accumulation)
4. Writing the final pickle file (large dataset)

ii. N/A

iii. The AI noted the dataset is ~87 GB and the final pickle is 8.9 GB.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for extracting fluorescence/neuropil, computing dF/F, and accumulating speed correlations iterates over every trial. The dF/F computation itself is already vectorized over neurons within each trial. The `_trial_outcomes` loop could potentially be vectorized with a single searchsorted call.

ii. N/A

iii. Trial-level looping is somewhat necessary due to variable trial lengths.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file only once, in a single pass. It does not have a separate survey step. All neural processing, trial filtering, and data extraction happen in one function call per session.

ii. N/A

iii. This is more efficient than the reference which loads each file twice (survey + conversion).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes dF/F for all trials (including bad-lick trials) in order to perform the session-wide interneuron speed-correlation screen. The dF/F values from excluded trials are computed but ultimately only contribute to the correlation calculation. The dF/F itself is discarded; only events are kept.

ii.
```python
# Process all trials for the paper's session-wide speed-correlation
# interneuron screen, including trials whose lick sensor was faulty.
for start, stop in zip(starts, stops):
    ...
    dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
```

iii. This is necessary for correct interneuron screening but means dF/F computation happens for trials that are later excluded.
