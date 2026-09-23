# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `sub-*/*.nwb` under the data root. Each file is opened with `pynwb.NWBHDF5IO` and the behavior and ophys processing modules are accessed. All 152 NWB files across 11 subjects are processed.

ii.
```python
DATA_ROOT = Path("/app/data")
def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    ...
    return files
...
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    behavior = nwb.processing["behavior"]["BehavioralTimeSeries"]
    ophys = nwb.processing["ophys"]
```

iii. The AI used `Path.glob` to find all NWB files and `pynwb` to load them. The CONVERSION_NOTES confirm 152 sessions from 11 mice were processed, matching the paper.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from the NWB file's `subject.subject_id` field and also parsed from the filename. A consistency check verifies they match. Unique subjects are collected and sorted.

ii.
```python
def session_key(path: Path) -> tuple[str, int]:
    match = re.search(r"sub-(m\d+)_ses-(\d+)", path.name)
    return match.group(1), int(match.group(2))
...
subject = nwb.subject.subject_id
if subject != subject_from_path:
    raise ValueError(...)
...
subjects = sorted({x["subject"] for x in converted_sessions}, key=lambda x: int(x[1:]))
```

iii. The AI verified subject IDs from both the filename and NWB metadata, ensuring consistency.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session number is parsed from the filename pattern `ses-<NN>`.

ii.
```python
def session_key(path: Path) -> tuple[str, int]:
    match = re.search(r"sub-(m\d+)_ses-(\d+)", path.name)
    return match.group(1), int(match.group(2))
```

iii. One session per NWB file, matching the data organization.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified using the `trial_start` and `teleport` behavior time series. Trial starts are frames where `trial_start > 0`. Trial ends are frames where `teleport > 0`. The AI uses `np.flatnonzero(b["teleport"] > 0)` for trial end detection.

ii.
```python
b = load_behavior(behavior, common_length)
starts = np.flatnonzero(b["trial_start"] > 0)
stops = np.flatnonzero(b["teleport"] > 0)
if len(starts) != len(stops) or len(starts) < 2:
    raise ValueError(...)
```

iii. The AI validates that the number of starts equals the number of stops and that all intervals are positive. The teleport detection differs from the reference (which detects rising edges of teleport), but produces the same result because teleport is a single-frame event in this dataset.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded for: (1) fewer than 2 samples, (2) scanning not equal to 1, (3) non-finite required behavior variables, and (4) corrupt lick sensor (>30% of frames with cumulative lick count > 2, matching the paper's documented criterion). Exactly 81 trials are removed by the lick criterion across the full dataset.

ii.
```python
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
    reason = None
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
```

iii. The lick corruption filter directly implements the paper's documented criterion: "n = 81 out of 12,376 trials removed" when ">30% of frames have cumulative lick count > 2". The scanning and finite-behavior checks are additional safety checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) series in the ophys processing module, NOT the NWB `Deconvolved` field.

ii.
```python
fluorescence_container = ophys["Fluorescence"]
neuropil_container = ophys["Neuropil"]
...
fluorescence = load_selected_roi_series(
    fluorescence_container, manual_ids, common_length, len(segmentation))
neuropil = load_selected_roi_series(
    neuropil_container, manual_ids, common_length, len(segmentation))
```

iii. The AI correctly identified that the NWB `Deconvolved` field is suite2p's own deconvolution, not the signal the paper analyzes. The CONVERSION_NOTES document this finding in Step 4.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's dF/F pipeline: (1) subtract 0.7 * neuropil, (2) add back per-trial neuropil mean, (3) compute maximin baseline (Gaussian smooth sigma=15, then 300-sample min/max filters), (4) compute dF/F as `(corrected - baseline) / |baseline|`, (5) smooth with Gaussian sigma=2, (6) deconvolve with OASIS (tau=0.7, rate=15.5078125 Hz). However, the AI does NOT use the `teleport_sessions` metadata to extend the baseline window for sessions where the laser was not blanked between trials. It always restricts the dF/F computation to within-trial periods only.

ii.
```python
def calculate_reference_dff(fluorescence, neuropil, trial_starts, trial_ends):
    for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
        f = fluorescence[:, start:stop].astype(np.float64, copy=True)
        fn = neuropil[:, start:stop].astype(np.float64, copy=False)
        corrected = f - 0.7 * fn + 0.7 * np.nanmean(fn, axis=1, keepdims=True)
        baseline = gaussian_smooth(corrected, 15)
        baseline = ndimage.minimum_filter1d(baseline, 300, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
        trial_dff = (corrected - baseline) / np.abs(baseline)
        trial_dff = gaussian_smooth(trial_dff, 2)
        dff[:, start:stop] = trial_dff
...
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
events[events < 0] = 0
```

iii. The AI's CONVERSION_NOTES Step 4 correctly discusses recomputing reference dF/F from F and Fneu. However, the code does not incorporate the `teleport_sessions` metadata that the reference code uses (from `src/reward_relative/teleport_metadata.py`) to extend the baseline window for specific animals and days where imaging continued during teleport. The AI also clips negative OASIS events to 0, which the reference does not.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) manual `iscell` curation from the PlaneSegmentation table, (2) putative interneuron exclusion when dF/F-speed Pearson correlation > 0.5.

ii.
```python
iscell = np.asarray(segmentation["iscell"].data[:])
manual_ids = np.flatnonzero(iscell[:, 0] == 1)
...
correlations = speed_correlations(dff, b["speed"])
keep = ~(correlations > 0.5)
```

iii. Both filters match the paper's methods: manual Suite2p curation and speed-correlation interneuron exclusion at r > 0.5. The AI correctly uses the global segmentation table rather than per-plane ROI tables for iscell.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to trial start. Each trial's neural data is sliced from the trial start index to the trial end (teleport) index. No additional temporal shift is needed since neural and behavioral data are already synchronized at imaging-frame resolution.

ii.
```python
trial_dff = dff[:, start:stop]
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
neural_trials.append(events)
```

iii. The alignment is implicitly correct since both neural and behavioral data share the same frame indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate is used: ~64.48 ms (1000/15.5078125). No rebinning is applied. The AI hardcodes `FRAME_RATE_HZ = 15.5078125` and `TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ`.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The rate matches the paper's documented ~15.5 Hz imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior timestamps array (specifically `behavior["position"].timestamps`).

ii.
```python
arrays["timestamps"] = np.asarray(behavior["position"].timestamps[:common_length])
...
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. Behavior timestamps are the same across all behavior time series in the NWB file.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the trial is subtracted from all timestamps within the trial.

ii.
```python
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. Simple subtraction to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same frame indices within each trial, so the same `[start:stop]` slice ensures alignment. No additional alignment is needed.

ii.
```python
# Both use the same start:stop indexing
time_from_start = timestamps[start:stop] - timestamps[start]
trial_dff = dff[:, start:stop]
```

iii. Data rates are the same for neural and behavior (~15.5 Hz), and all are synchronized at the imaging frame level.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = mode_value(b["environment"][start:stop])
```

iii. The `environment` variable encodes ENV1 (0) vs ENV2 (1).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The mode (most frequent value) of the environment variable within the trial is computed and broadcast as a constant value across all timepoints.

ii.
```python
environment = mode_value(b["environment"][start:stop])
inputs = np.vstack((
    ...
    np.full(stop - start, environment),
    ...
))
```

iii. Environment is constant within a trial, so mode gives the same result as using the raw values directly. This is a per-trial input as specified.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` behavior time series (the stored trial index).

ii.
```python
trial_number = mode_value(b["trial number"][start:stop])
```

iii. The AI uses the stored NWB trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The mode of the NWB `trial number` variable within the trial is computed and broadcast as a constant value. This preserves the original trial numbering from the experiment.

ii.
```python
trial_number = mode_value(b["trial number"][start:stop])
inputs = np.vstack((
    ...
    np.full(stop - start, trial_number),
    ...
))
```

iii. The mode computation is used because the NWB trial number is defined per-sample and may have edge effects at trial boundaries. The value is broadcast as a per-trial constant.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward delivery timestamps are compared to each trial's [start, stop) interval to determine trial outcomes.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:])
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
```

iii. The AI checks whether any reward delivery timestamp falls within each trial's time interval.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the outcome of the immediately preceding source trial is used. For the first trial, the value is 0. The previous outcome refers to the original (source) trial index, not the retained trial index after filtering.

ii.
```python
previous_outcome = int(outcomes[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. Using source trial index ensures that filtering a corrupt-lick trial does not change the history seen by subsequent trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone identity for the current trial. The reward zone identity is determined by parsing the NWB file's `identifier` field to extract the scene/environment schedule, then applying a rule that the zone switches after the first 30 trials on switch days.

ii.
```python
def scene_from_identifier(identifier: str) -> str:
    return identifier.rstrip("/").split("/")[-1]

def scene_zone_labels(scene: str) -> list[str]:
    labels: list[str] = []
    for token in scene.split("_"):
        if token in ZONE_BOUNDS:
            labels.append(token)
        elif token.startswith("Location") and token[-1:] in ZONE_BOUNDS:
            labels.append(token[-1])
    ...

def zones_by_trial(scene: str, n_trials: int) -> list[str]:
    labels = scene_zone_labels(scene)
    if len(labels) == 1:
        return [labels[0]] * n_trials
    return [labels[0]] * 30 + [labels[1]] * (n_trials - 30)
```

iii. The AI parses the NWB identifier to derive zone labels (A/B/C), with the same zone coordinates from the paper (A=80-130, B=200-250, C=320-370 cm). This is a different approach from the reference (which uses a Viterbi algorithm on reward_zone positions), but derives from the same experimental schedule information.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the active reward zone. Distance is negative before the zone, zero inside the zone, and positive after the zone.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def reward_distance(position: np.ndarray, zone: str) -> np.ndarray:
    start, stop = ZONE_BOUNDS[zone]
    return np.where(position < start, position - start,
                    np.where(position > stop, position - stop, 0.0))
```

iii. Same signed-distance computation as the reference: `position - zone_start` when before, `position - zone_end` when after, 0 when inside.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boundary conditions:

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

iii. The AI uses explicit boolean indexing rather than `np.digitize`. The boundary conditions differ slightly from the reference (e.g., -10 goes to bin 1 in the AI but bin 2 in the reference; 50 goes to bin 5 in the AI but bin 6 in the reference).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial, using the same `[start:stop]` slice.

ii.
```python
position = b["position"][start:stop]
distance = reward_distance(position, zone)
```

iii. Same frame-level alignment as all other variables.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = b["position"][start:stop]
```

iii. The `position` variable records the animal's position in the virtual corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
discretize_position(position)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using explicit boundary conditions:

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

iii. Five 90-cm bins spanning the 450-cm track. The AI's boundary handling uses left-closed, right-open intervals for most bins, with `[270, 360]` closed on both sides. The reference uses `np.digitize` with slightly different boundary behavior.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as the neural data within each trial.

ii. Same `[start:stop]` indexing.

iii. Verified by shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = b["lick"][start:stop]
```

iii. The `lick` variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
(lick > 0).astype(np.int16)
```

iii. Matches the instructions for binary lick output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as the neural data within each trial.

ii. Same `[start:stop]` indexing.

iii. Frame-level alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file's `identifier` field, which encodes the scene/environment schedule including which reward zone is active. The zone coordinates are A=80-130, B=200-250, C=320-370 cm.

ii.
```python
scene = scene_from_identifier(nwb.identifier)
zones = zones_by_trial(scene, len(starts))
...
np.full(stop - start, "ABC".index(zone), dtype=np.int16)
```

iii. The AI parses the NWB identifier to determine the zone schedule: single-zone sessions get one label for all trials, switch sessions get the first zone for trials 1-30 and the second zone for trials 31+.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Zone labels (A/B/C) are mapped to integers (0/1/2) and broadcast as per-trial constants.

ii.
```python
def zones_by_trial(scene: str, n_trials: int) -> list[str]:
    labels = scene_zone_labels(scene)
    if len(labels) == 1:
        return [labels[0]] * n_trials
    return [labels[0]] * 30 + [labels[1]] * (n_trials - 30)
```

iii. Uses the reference paper's documented switch-after-30-trials rule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:])
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
```

iii. Reward delivery events with their own timestamps, separate from behavior sampling.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time interval `[start_time, end_time)`. Per-trial binary output broadcast across all timepoints.

ii.
```python
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
...
np.full(stop - start, outcomes[trial_idx], dtype=np.int16)
```

iii. Three reward events outside all trial intervals are intentionally excluded. The reward rate is ~84.6%, matching the paper's ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is cropped to the minimum common length across all neural and behavior arrays.
- **Short trials**: Trials with fewer than 2 samples are excluded.
- **Non-scanning periods**: Trials where scanning is not 1 are excluded.
- **Non-finite behavior**: Trials with NaN/Inf in required behavior variables are excluded.
- **Corrupt lick sensor**: Trials where >30% of frames have lick count > 2 are excluded (81 trials total).
- **Reward events outside trials**: Three inter-trial reward deliveries are not assigned to any trial.
- **Subject ID verification**: Cross-checked between filename and NWB metadata.

ii.
```python
common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                    [x.data.shape[0] for x in neuropil_series])
...
if stop - start < 2:
    reason = "fewer than two samples"
elif not np.all(b["scanning"][start:stop] == 1):
    reason = "outside valid scanning period"
...
lick_bad_fraction = float(np.mean(b["lick"][start:stop] > 2))
if lick_bad_fraction > 0.30:
    reason = "corrupt lick sensor (>30% frames with count >2)"
```

iii. The AI's error handling is comprehensive and well-documented.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** and reading large fluorescence/neuropil arrays — I/O bound.
2. **Computing dF/F** per session — involves Gaussian smoothing, min/max filtering, and per-trial operations.
3. **OASIS deconvolution** — per trial for all retained cells.
4. **Saving the pickle file** — the final dataset is ~9.5 GB.

ii. N/A

iii. The AI reported total conversion time of ~7.4 minutes for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially for input/output construction and deconvolution. The OASIS deconvolution is called per trial (cannot easily be vectorized across variable-length trials). The trial outcome computation uses a list comprehension over all trials.

ii.
```python
outcomes = np.array([
    np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
    for start, stop in zip(starts, stops)
], dtype=np.int16)
```

iii. The per-trial loop is natural given variable-length trials. The speed correlations are already vectorized.

## 13-c. What processing does the code repeat multiple times?

i. The AI's code processes each session in a single pass (no separate survey step), so there is minimal repeated processing. However, the dF/F is computed for all cells before the interneuron filter, meaning some cells are processed but then discarded.

ii. N/A

iii. The AI's approach is more efficient than a two-pass approach (survey then convert) in that each NWB file is loaded once.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dF/F computation includes all manually curated cells before the interneuron speed-correlation filter removes some. The plot payload construction occurs for every session even if `show_processing` is False (though it only captures data from the first trial when not None, and the check prevents overwriting). Negative OASIS events are clipped to 0, which is extra processing not in the reference.

ii.
```python
events[events < 0] = 0
```

iii. Clipping negative events is a minor extra step. Processing all manual cells before filtering is necessary since the interneuron filter requires dF/F to be computed first.
