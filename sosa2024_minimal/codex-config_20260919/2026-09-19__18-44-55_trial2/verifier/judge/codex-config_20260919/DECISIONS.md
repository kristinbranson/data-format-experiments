# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively selects every matching NWB file under `sub-*`, naturally sorts them by mouse and session number, and reads each directly with `h5py`. Each file becomes one output session. An optional `--max-sessions` is only a smoke-test override.

ii.
```python
files = sorted(data_root.glob("sub-*/*_behavior+ophys.nwb"), key=_natural_key)
...
for index, path in enumerate(files, start=1):
    arrays, info = convert_session(path)
...
with h5py.File(path, "r") as nwb:
```

iii. The trajectory says the release was surveyed first and found to contain 152 sessions from 11 task mice. The agent intended to preserve the source session/trial organization and include all complete data except documented quality exclusions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB's `general/subject/subject_id`; unique IDs are numerically sorted, and each session receives an index into that list.

ii.
```python
subject = _decode_scalar(nwb["general/subject/subject_id"][()])
...
subjects = sorted(set(session_subjects), key=lambda value: int(re.search(r"\d+", value).group()))
subject_lookup = {subject: index for index, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[s] for s in session_subjects], dtype=np.int64)
```

iii. The agent inspected the NWB subject fields and reported 11 mice in the completed conversion.

## 1-c. How are the data split into sessions?

i. Each `*_behavior+ophys.nwb` file is one session, ordered by parsed subject and `ses-` number.

ii.
```python
session = re.search(r"ses-(\d+)", path.name)
return (int(subject.group(1)), int(session.group(1)))
...
arrays, info = convert_session(path)
neural.append(arrays["neural"])
```

iii. The trajectory describes the release as daily NWB sessions and validates the final count of 152.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples in `trial_start`; ends are positive samples in `teleport`. Data are sliced `[start:stop)`, excluding teleport onset. Counts and ordering are validated.

ii.
```python
starts = np.flatnonzero(_read_behavior(behavior, "trial_start") > 0)
stops = np.flatnonzero(_read_behavior(behavior, "teleport") > 0)
...
for trial, (start, stop) in enumerate(zip(starts, stops)):
```

iii. The agent explicitly stated that it retained the on-track interval from trial start up to, but not including, teleport and inspected boundary values in representative files.

## 1-e. How are trials filtered based on quality controls?

i. Complete trials are retained except trials meeting the paper's faulty-lick criterion: more than 30% of samples have lick count greater than 2. No minimum-duration filter is applied.

ii.
```python
return np.asarray([
    np.mean(lick[start:stop] > 2) > 0.30
    for start, stop in zip(starts, stops)
], dtype=bool)
...
if bad_lick[trial]:
    continue
```

iii. The agent found exactly 81 trials matching the criterion in the paper/code and reasoned that they could not provide a valid lick target. It retained all other complete trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from manually curated ROI columns in the NWB `Fluorescence` and `Neuropil` series, resolved separately for every imaging plane. The stored `Deconvolved` series is not used.

ii.
```python
fluorescence_group = nwb[f"{OPHYS}/Fluorescence"]
neuropil_group = nwb[f"{OPHYS}/Neuropil"]
table_rows = np.asarray(fluorescence_group[f"{plane_name}/rois"], dtype=np.int64)
local_keep = np.flatnonzero(is_cell[table_rows])
```

iii. The agent concluded from the paper repository that the published analysis recomputed dF/F and events from F/Fneu, rather than using the NWB's Suite2p-deconvolved field. It also corrected its initial indexing after discovering the two-plane NWB layout.

## 2-b. How is the `neural` data processed?

i. Within each trial and plane, the code subtracts `0.7*neuropil`, restores `0.7` times the trial neuropil mean, applies Gaussian smoothing (sigma 15), 300-sample minimum then maximum filters for the maximin baseline, computes dF/F, smooths it with sigma 2, and runs OASIS with tau 0.7 and the aligned sampling rate. Plane results are concatenated. The baseline never spans teleport intervals.

ii.
```python
corrected = fluorescence - 0.7 * neuropil
corrected += 0.7 * neuropil.mean(axis=1, keepdims=True)
baseline = gaussian_filter1d(corrected, 15, axis=1)
baseline = minimum_filter1d(baseline, 300, axis=1)
baseline = maximum_filter1d(baseline, 300, axis=1)
dff = (corrected - baseline) / np.abs(baseline)
dff = gaussian_filter1d(dff, 2, axis=1).astype(np.float32, copy=False)
events = dcnv.oasis(dff, 2000, 0.7, rate_hz)
```

iii. The agent said this reproduced the paper's trial-wise neuropil correction, maximin dF/F, two-sample smoothing, and OASIS procedure. It chose timestamps to obtain the effective per-plane rate. It did not mention or implement the reference's mouse/day-specific cases where imaging continued and the baseline could span teleports.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must pass Suite2p/manual `iscell` curation. Then cells with a session-wide Pearson correlation between dF/F and speed greater than 0.5 are removed. Correlations include all source trials, even faulty-lick trials.

ii.
```python
is_cell = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation/iscell"][:, 0].astype(bool)
...
speed_correlation = np.divide(
    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
)
putative_interneuron = speed_correlation > 0.5
keep_neuron = ~putative_interneuron
```

iii. The trajectory identifies both filters as paper-matched and intentionally computes the screen across the session before trial exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays start exactly at the detected trial-start index and end before teleport; no further shifting is applied.

ii.
```python
np.asarray(f_data[start:stop, :], dtype=np.float32)
...
"temporal_alignment_event": "start of trial (entry into the 450 cm corridor)",
"off_start": 0.0,
```

iii. The agent treated NWB behavior and imaging rows as already aligned and defined trial start as time zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code retains the native aligned sampling of 15.5078125 Hz, or about 64.48 ms per bin. It performs no rebinning or interpolation (the Gaussian filters smooth but do not change sampling).

ii.
```python
NOMINAL_RATE_HZ = 15.5078125
rate = float(1.0 / np.median(np.diff(timestamps)))
...
"time_bin_size": 1000.0 / NOMINAL_RATE_HZ,
```

iii. The agent checked timestamps and noted that two-plane files have a misleading 31 Hz imaging attribute while their aligned effective bins remain 64.48 ms.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the number of samples since `trial_start` and the sampling rate inferred from `position/timestamps`.

ii.
```python
timestamps = np.asarray(behavior["position/timestamps"])
rate_hz = float(1.0 / np.median(np.diff(timestamps)))
...
np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz)
```

iii. The agent preferred aligned timestamps because they correctly represent two-plane sessions.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based sample counter for each trial is divided by the inferred rate.

ii.
```python
np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz)
```

iii. This makes the first retained frame time zero without temporal resampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Its length is `stop-start`, exactly matching the neural slice, with element zero corresponding to the same start frame.

ii.
```python
timepoints = stop - start
input_trial = np.vstack([
    np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
```

iii. The agent relied on row-wise NWB alignment and verified consistent rates/shape through the supplied validator.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment/data` stream within each trial.

ii.
```python
environment = _read_behavior(behavior, "environment")
trial_env = int(np.rint(np.median(environment[start:stop])))
```

iii. Inspection showed environment is a per-trial condition.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The within-trial median is rounded to an integer, then repeated at every trial timepoint.

ii.
```python
trial_env = int(np.rint(np.median(environment[start:stop])))
np.full(timepoints, trial_env, dtype=np.float32)
```

iii. This robustly enforces the specified binary per-trial representation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the NWB behavior stream `trial number/data`, not from the loop index.

ii.
```python
trial_number = _read_behavior(behavior, "trial number")
source_trial_number = float(np.median(trial_number[start:stop]))
```

iii. The trajectory inspected the stream but gives no explicit rationale for preferring it; the variable name directly corresponds to the requested input.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median value within the trial is computed and repeated over all timepoints.

ii.
```python
source_trial_number = float(np.median(trial_number[start:stop]))
np.full(timepoints, source_trial_number, dtype=np.float32)
```

iii. The median treats trial number as constant and guards against isolated inconsistent samples.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, behavior `position/timestamps`, and the preceding source trial's start/stop boundaries.

ii.
```python
outcomes = _trial_outcomes(
    np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
)
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
```

iii. Reward events have separate timestamps, so the agent searches those timestamps inside each trial interval.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if at least one reward timestamp falls between its start (inclusive) and stop (exclusive). The current trial receives the preceding source trial's binary outcome, or zero for trial zero, repeated over time.

ii.
```python
left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
outcomes[trial] = right > left
...
np.full(timepoints, previous_outcome, dtype=np.float32)
```

iii. This implements omitted=0/rewarded=1 and retains the actual previous source trial even when that previous trial is later excluded for lick quality.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses `position/data` and an inferred per-trial zone. Zone inference uses the nonzero `reward_zone/data` samples, their median position, fixed zone centers, session day, and the protocol's trial-30 switch structure.

ii.
```python
mask = rzone[start:stop] > 0
event_position = float(np.median(position[start:stop][mask]))
observed[trial] = int(np.argmin(np.abs(ZONE_CENTERS - event_position)))
boundaries = [0, min(30, len(starts)), len(starts)] if experiment_day in SWITCH_DAYS else [0, len(starts)]
```

iii. The agent determined that `reward_zone` is an event/count stream rather than a label. It cross-checked inferred zones against every NWB identifier and reported exact agreement with A/B/C and switch conditions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each sample, signed distance is position minus the near edge before a zone, zero inside its inclusive 50 cm interval, and position minus the far edge after it.

ii.
```python
start, end = ZONE_STARTS[zone], ZONE_ENDS[zone]
distance = np.where(position < start, position - start,
                    np.where(position > end, position - end, 0.0))
```

iii. This is the requested distance to any location in the active reward zone and follows the paper's A/B/C intervals.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code applies ordered comparisons implementing the seven requested bins; zero is its own class and positive boundary values 10 and 50 fall into classes 4 and 5.

ii.
```python
return np.select(
    [distance < -50, distance < -10, distance < 0, distance == 0,
     distance <= 10, distance <= 50],
    [0, 1, 2, 3, 4, 5], default=6,
).astype(np.int8)
```

iii. The comparisons directly encode the instruction thresholds.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the identical `[start:stop)` interval, so each output column corresponds to the same neural column.

ii.
```python
output_trial = _output_matrix(position[start:stop], ...)
neural_trials.append(np.ascontiguousarray(events_by_trial[trial][keep_neuron]))
```

iii. No interpolation was considered necessary because the NWB arrays are aligned sample-wise.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is directly derived from behavior `position/data`.

ii.
```python
position = _read_behavior(behavior, "position")
...
position[start:stop]
```

iii. The stream is the animal's absolute virtual-corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is discretized directly; there is no smoothing, normalization, or resampling.

ii.
```python
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. Five 90 cm bins span the stated 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with thresholds 90, 180, 270, and 360 returns categories 0–4.

ii.
```python
np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. These are the five equal-width bins required by the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both use the same trial start and stop row indices.

ii.
```python
output_trial = _output_matrix(position[start:stop], ...)
```

iii. The agent treated behavior and neural rows as pre-aligned and validated equal trial shapes.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from behavior `lick/data`.

ii.
```python
lick = _read_behavior(behavior, "lick")
```

iii. This is the NWB's aligned lick stream; trials known to have faulty lick values are removed.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every positive value is mapped to one and all other values to zero.

ii.
```python
lick_binary = (lick > 0).astype(np.int8)
```

iii. The requested output is binary, whereas the source may contain positive counts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick stream is sliced using the same `[start:stop)` indices as neural activity.

ii.
```python
output_trial = _output_matrix(
    position[start:stop], speed[start:stop], lick[start:stop], ...
)
```

iii. The agent relied on sample-wise NWB alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone/data`, `position/data`, `general/session_id`, and the known switch-day protocol.

ii.
```python
experiment_day = int(_decode_scalar(nwb["general/session_id"][()]))
zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
```

iii. The event stream can be missing on omission trials, so the agent inferred condition-level labels rather than treating raw event magnitudes as labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Observed trials are assigned to the nearest fixed zone center based on median event position. Within each protocol block (whole session, or trials before/after 30 on switch days), the modal observed label fills every trial, including omissions. The label is repeated across time.

ii.
```python
valid = observed[left:right]
valid = valid[valid >= 0]
labels[left:right] = np.bincount(valid, minlength=3).argmax()
...
np.full(timepoints, zone, dtype=np.int8)
```

iii. The agent reasoned that the protocol has one condition per session or two blocks on switch days and verified the results against session identifiers.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from `Reward/timestamps`, aligned using `position/timestamps` and trial boundaries.

ii.
```python
outcomes = _trial_outcomes(
    np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
)
```

iii. The reward event series has its own timestamps rather than a frame-aligned binary data stream.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Searchsorted identifies whether any reward timestamp lies in each trial; the resulting 0/1 value is repeated at all timepoints.

ii.
```python
left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
outcomes[trial] = right > left
...
np.full(timepoints, rewarded, dtype=np.int8)
```

iii. This directly represents omitted versus rewarded trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code fails loudly for unmatched/misordered boundaries, unexpected rates, absent curated cells, blocks with no observable reward zone, nonfinite processed neural values, or removal of every neuron. Missing zone events on individual omission trials are filled by the protocol-block mode. Faulty-lick trials are excluded. It does not crop neural/behavior length mismatches or remove short trials.

ii.
```python
if len(starts) != len(stops):
    raise ValueError(...)
...
if len(valid) == 0:
    raise ValueError(...)
...
if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(events)):
    raise ValueError(...)
```

iii. The trajectory emphasizes integrity checks and reports that the full output passed validation without warnings. The modal zone fill was explicitly designed for reward-omission trials.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large per-trial fluorescence/neuropil slices, repeated Gaussian/min/max filtering and OASIS deconvolution for every trial, assembling the 8.9 GB in-memory result, and pickle writing are the dominant costs.

ii.
```python
for start, stop in zip(starts, stops):
    fluorescence = np.concatenate([...], axis=0)
    ...
    dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. During the run the agent said trial-wise processing bounded memory and that final assembly/write would be heavier; the full conversion processed 152 large NWBs and produced 8.9 GB.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. `_trial_outcomes`, `_reward_zone_labels`, `_bad_lick_trials`, the neural-processing loop, and the retained-trial construction loop all iterate over trials. Outcome searches, bad-lick fractions, and some reward-zone summaries could be vectorized; the variable-length per-trial baseline/deconvolution is less naturally vectorized.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
return np.asarray([
    np.mean(lick[start:stop] > 2) > 0.30
    for start, stop in zip(starts, stops)
], dtype=bool)
```

iii. The trajectory does not explicitly discuss vectorization; it instead justifies trial-wise work as a way to keep memory bounded.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries are traversed four times: reward outcomes, zone observations, bad-lick detection, and neural processing/output assembly. Each retained trial is also revisited after events were already stored. Constant per-session behavioral slices and arrays are repeatedly indexed.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):  # outcomes
...
for trial, (start, stop) in enumerate(zip(starts, stops)):  # zones
...
for start, stop in zip(starts, stops):                      # neural
...
for trial, (start, stop) in enumerate(zip(starts, stops)):  # output
```

iii. No explicit trajectory justification was given beyond bounded memory and the need to compute a session-wide neural quality screen before final trial assembly.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes neural dF/F and OASIS events for faulty-lick trials, then discards those trials from the final dataset; only their dF/F contribution to the session-wide speed-correlation screen remains useful, while their event arrays are unused. It also computes/stores extensive session metadata not required by decoder training.

ii.
```python
events_by_trial.append(events)
...
if bad_lick[trial]:
    continue
...
"reward_zone_by_source_trial": zones.astype(int).tolist(),
"reward_outcome_by_source_trial": outcomes.astype(int).tolist(),
```

iii. The agent explicitly chose to process all trials for the paper's session-wide interneuron screen. It did not separately discuss the wasted event deconvolution or optional metadata overhead.
