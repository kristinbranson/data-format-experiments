# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It recursively selects every `sub-*/*.nwb` file, naturally sorts by mouse/session, and opens each file directly with `h5py`. Each file becomes a converted session.

ii.
```python
files = sorted(data_root.glob("sub-*/*.nwb"), key=natural_key)
...
with h5py.File(path, "r") as h5:
```

iii. The module docstring says every available session is used. The trajectory shows the agent surveyed the directory/NWB layout, chose direct HDF5 access for speed, and ultimately verified 152 sessions from 11 mice.

## 1-b. How are the data split into subjects (mice)?

i. Mouse IDs are parsed from NWB paths for sorting/list construction, while each session’s subject is read from `general/subject/subject_id`; a lookup maps each session to `subject_idx`.

ii.
```python
subjects = sorted({f"m{natural_key(path)[0]}" for path in files}, key=lambda value: int(value[1:]))
subject = h5["general/subject/subject_id"][()].decode()
subject_idx.append(subject_lookup[info["subject"]])
```

iii. The agent treated BIDS-style `sub-m...` directories and NWB subject metadata as mouse identity and cross-checked the resulting 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; `convert_session(path)` returns one list of trials, which is appended once to each top-level session list.

ii.
```python
for number, path in enumerate(files, start=1):
    session_neural, session_input, session_output, info = convert_session(path)
    neural.append(session_neural)
```

iii. The docstring explicitly identifies one recording (mouse/day) as a session. The trajectory reports validating 152 such sessions.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples of `trial_start`; stops are positive samples of `teleport`. Each trial is the half-open slice `[start:stop)`, with equal-count/order validation.

ii.
```python
starts = np.flatnonzero(behavior["trial_start/data"][:] > 0)
stops = np.flatnonzero(behavior["teleport/data"][:] > 0)
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(...)
```

iii. The agent interpreted the paper’s lap as trial-start through teleport and intentionally excluded teleport/ITI periods. It inspected sample counts and documented this interval in metadata.

## 1-e. How are trials filtered based on quality controls?

i. A trial is excluded when more than 30% of its lick samples exceed 2, interpreted as a failed cumulative lick detector. It also rejects a whole session if fewer than two trials remain; there is no short-trial filter.

ii.
```python
bad_lick_sensor = np.mean(trial_lick > 2) > LICK_ERROR_FRACTION
...
if bad_lick_sensor:
    excluded_lick_trials += 1
    continue
```

iii. The agent found the Methods’ 81 lick-detector failures and argued that invalid lick labels cannot be decoder targets. The final audit reported exactly 81 exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural output is recomputed from raw Suite2p ROI `Fluorescence` and `Neuropil`, after selecting ROIs via `iscell` and ordering via `planeIdx`.

ii.
```python
f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
```

iii. The agent concluded the paper derives its own events from F/Fneu rather than using the NWB `Deconvolved` field.

## 2-b. How is the `neural` data processed?

i. For every trial independently: subtract `0.7*Fneu`, add back the trial mean neuropil, Gaussian smooth (σ=15 samples), apply 300-sample min then max baseline filters, compute dF/F, Gaussian smooth (σ=2), then OASIS-deconvolve with τ=0.7 s at 15.5078125 Hz.

ii.
```python
corrected = f - NEUROPIL_COEF * f_neu
corrected += NEUROPIL_COEF * np.mean(f_neu, axis=1, keepdims=True)
baseline = gaussian_filter1d(corrected, 15, axis=1)
baseline = minimum_filter1d(baseline, 300, axis=1)
baseline = maximum_filter1d(baseline, 300, axis=1)
dff = gaussian_filter1d((corrected-baseline)/np.abs(baseline), 2, axis=1)
events = dcnv.oasis(dff, 2000, 0.7, FRAME_RATE_HZ)
```

iii. The agent traced the parameters to the Methods/repository and chose trial-local processing to reproduce the paper. It used float32 to keep the full conversion tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It first retains only manually curated `iscell[:,0]==1` ROIs. It then removes neurons whose session-wide Pearson correlation between trial dF/F and speed is greater than 0.5, using streaming sufficient statistics over all trials (including bad-lick trials).

ii.
```python
iscell = np.asarray(iscell_table == 1)
...
speed_correlation = correlation_from_sums(...)
keep_neuron = speed_correlation <= 0.5
neural_trials = [events[keep_neuron] for events in trial_events]
```

iii. Both filters were taken from the Methods. The trajectory shows the agent audited discrepancies in neuron masks and reran the final conversion after correcting the streaming correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays begin exactly at the `trial_start` index, so time zero is the first sample of each `[start:stop)` slice; no interpolation or shifting is applied.

ii.
```python
f = read_curated_trial(h5, "Fluorescence", start, stop, ...)
...
"temporal_alignment_event": "start of trial ..."
```

iii. The requested alignment is trial start, already shared by neural and behavior sample indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native samples are retained without rebinning at 15.5078125 Hz, giving `64.4838709677 ms` bins.

ii.
```python
FRAME_RATE_HZ = 15.5078125
...
"time_bin_size": 1000.0 / FRAME_RATE_HZ
```

iii. The agent reconciled misleading dual-plane metadata with behavioral timestamp spacing and the paper’s stated ~15.5 Hz effective per-plane sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
timestamps = behavior["position/timestamps"][:]
```

iii. The position timestamps are the aligned behavioral clock inspected by the agent.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each retained trial, the timestamp at trial start is subtracted from every timestamp in the trial and the result is cast to float32.

ii.
```python
time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. Subtraction makes the first retained sample exactly zero while preserving native, potentially variable trial duration.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[start:stop)` sample slice and therefore has one timestamp per neural column.

ii.
```python
time_from_start = timestamps[start:stop] - timestamps[start]
inputs = np.vstack((time_from_start, ...))
```

iii. The agent observed that behavior arrays and fluorescence arrays share aligned sample rows, so no resampling was needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The primary value is parsed from the NWB identifier’s scene name (`Env1`/`Env2`, including switches); the raw `environment/data` stream is read and used as a per-trial consistency check.

ii.
```python
scene = identifier.rsplit("/", 1)[-1]
environments, zones = scene_conditions(scene, len(starts))
env_stream = behavior["environment/data"][:]
```

iii. The agent found scene names encode experimental conditions and used the raw stream to ensure the parsing agrees.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Env1 maps to 0 and Env2 to 1. Static scenes fill all trials; switch scenes change at original trial index 30. The value is repeated over all timepoints in a trial.

ii.
```python
environments = np.where(np.arange(n_trials) < 30, int(env_switch.group(1))-1, int(env_switch.group(3))-1)
...
np.full(len(pos), environment, dtype=np.float32)
```

iii. The scene convention and 30-trial blocks were inferred from the repository/data; median raw-stream agreement is enforced.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavioral `trial number/data` stream.

ii.
```python
trial_number_stream = behavior["trial number/data"][:]
```

iii. The agent used the explicit NWB trial-number variable rather than the Python loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. It takes the median value across the trial, converts it to float, and repeats it for every trial timepoint.

ii.
```python
trial_number = float(np.median(trial_number_stream[start:stop]))
np.full(len(pos), trial_number, dtype=np.float32)
```

iii. Median robustly extracts the per-trial constant from a time series; time expansion satisfies the decoder’s 2-D input convention.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from the computed reward outcomes of the immediately preceding original trial; those outcomes use `Reward/timestamps` and `reward_zone/data`.

ii.
```python
reward_times = behavior["Reward/timestamps"][:]
rzone_stream = behavior["reward_zone/data"][:]
previous_outcome = float(outcomes[trial - 1]) if trial > 0 else 0.0
```

iii. The agent intentionally used the preceding original trial, not the preceding retained trial, so lick-QC removal does not change experimental history.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. All original outcomes are computed first. Trial 0 receives 0; later trials receive `outcomes[trial-1]`, repeated across the current trial.

ii.
```python
outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
...
previous_outcome = float(outcomes[trial - 1]) if trial > 0 else 0.0
```

iii. The first trial has no predecessor and is conventionally encoded omitted=0. Precomputing preserves literal previous-trial semantics.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position/data` plus reward-zone identity parsed from the scene identifier; fixed bounds are A=80–130, B=200–250, C=320–370 cm.

ii.
```python
position = behavior["position/data"][:]
ZONE_BOUNDS = {"A": (80.0,130.0), "B": (200.0,250.0), "C": (320.0,370.0)}
```

iii. The agent used reward-zone bounds from the paper/repository and scene metadata to avoid noisy instantaneous reward-zone samples.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the nearest boundary before/after the zone and exactly zero anywhere inside the interval.

ii.
```python
distance = np.where(position < start, position - start,
                    np.where(position > stop, position - stop, 0.0))
```

iii. This implements reward-relative position as described in the paper and the requested zero-valued zone interior.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It explicitly assigns seven classes: `<-50`, `[-50,-10)`, `[-10,0)`, exactly 0, `(0,10]`, `(10,50]`, and `>50`.

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

iii. The explicit comparisons were chosen to implement the instruction’s boundary semantics, especially the separate exact-zero category.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same trial start/stop indices and converted sample-for-sample, producing one class per neural column.

ii.
```python
pos = np.asarray(position[start:stop], dtype=np.float32)
outputs = np.vstack((discretize_distance(pos, ...), ...))
```

iii. All streams are stored on the same aligned NWB sample grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavioral `position/data`.

ii.
```python
position = behavior["position/data"][:]
pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. This is the NWB’s corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No transformation precedes categorical binning beyond slicing/casting the raw position.

ii.
```python
np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)
```

iii. The requested output is simply absolute track position categorized into equal track segments.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with internal edges 90, 180, 270, and 360 cm produces classes 0–4.

ii.
```python
np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)
```

iii. A 450 cm corridor divided into five equal bins has 90 cm boundaries.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The identical `[start:stop)` indices are used, with no interpolation.

ii.
```python
pos = position[start:stop]
```

iii. Shared sample indexing gives one position category per neural sample.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from behavioral `lick/data`.

ii.
```python
lick = behavior["lick/data"][:]
trial_lick = lick[start:stop]
```

iii. The NWB lick series is the direct behavioral source.

## 9-b. What processing is involved in computing `output` *Lick*?

i. For retained trials, any positive sample becomes 1 and all others 0. Trials with the documented failed-sensor pattern are removed first.

ii.
```python
bad_lick_sensor = np.mean(trial_lick > 2) > 0.30
...
(lick[start:stop] > 0).astype(np.int8)
```

iii. Binary thresholding meets the task definition; removing the 81 detector failures avoids training on corrupted lick labels.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same `[start:stop)` sample slice as neural activity.

ii.
```python
(lick[start:stop] > 0).astype(np.int8)
```

iii. The aligned behavioral time-series rows correspond directly to neural columns.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene string stored in the NWB identifier, not directly from the raw `reward_zone` stream. Scene parsing assigns A/B/C per original trial.

ii.
```python
identifier = h5["identifier"][()].decode()
scene = identifier.rsplit("/", 1)[-1]
environments, zones = scene_conditions(scene, len(starts))
```

iii. The agent judged scene metadata to be a clean representation of the designed condition, whereas the reward-zone stream marks active delivery periods rather than reliably naming the zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Static scenes fill one label; switch scenes use the first zone before trial 30 and the second thereafter. A/B/C map to 0/1/2 and are repeated through the trial.

ii.
```python
zones = np.where(np.arange(n_trials) < 30, ..., ...)
zone_to_class = {"A": 0, "B": 1, "C": 2}
np.full(len(pos), zone_to_class[zone], dtype=np.int8)
```

iii. The agent inferred the fixed 30-trial switch protocol from identifiers/repository and checked related environment values against the raw stream.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses reward-event timestamps from `Reward/timestamps`, behavioral position timestamps for interval lookup, and `reward_zone/data` as an active-zone confirmation.

ii.
```python
reward_times = behavior["Reward/timestamps"][:]
timestamps = behavior["position/timestamps"][:]
rzone_stream = behavior["reward_zone/data"][:]
```

iii. The agent aimed to match the repository’s `get_trial_types`, interpreting a rewarded trial as both a delivered reward and an active reward-zone sample.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each original trial, `searchsorted` counts reward timestamps in `[timestamps[start], timestamps[stop])`; outcome is 1 only if at least one exists and some reward-zone sample is positive. The result is repeated over all samples.

ii.
```python
left = np.searchsorted(reward_times, timestamps[start], side="left")
right = np.searchsorted(reward_times, timestamps[stop], side="left")
outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. The conjunction was chosen as a repository-faithful guard against spurious event timing; outcomes are computed before trial filtering for correct history.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code fails on invalid start/stop pairing, scene/environment mismatch, unknown scene names, absent files, or fewer than two retained trials. Nonfinite dF/F from degenerate baselines is replaced with finite zeros. Bad lick trials are excluded. Output is written atomically through a temporary file.

ii.
```python
if not np.isfinite(dff).all():
    dff = np.nan_to_num(dff, copy=False)
...
if stream_environment != int(environments[trial]): raise ValueError(...)
...
os.replace(temporary, args.output)
```

iii. The agent favored explicit invariants for structural errors, a narrow numerical fallback for unexpected zero baselines, and recoverable atomic saving. It identified and removed exactly the Methods’ 81 lick failures.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large fluorescence/neuropil slabs, repeated per-trial filtering and OASIS deconvolution, and serializing the roughly 9.5 GB pickle dominate runtime.

ii.
```python
slab = np.asarray(series[plane_name]["data"][start:stop, :], dtype=np.float32)
...
events = dcnv.oasis(dff, 2000, CALCIUM_TAU_S, FRAME_RATE_HZ)
...
pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Trajectory timing showed conversion and full decoder validation were long-running. Comments specifically optimize HDF5 slab reads and float precision for tractability.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over trials for outcomes, neural processing, and input/output assembly could partly be session-vectorized; per-plane reads could also be consolidated. Session/file loops remain natural because shapes differ.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
for trial in retained_trial_indices:
    ...
```

iii. The agent retained trial loops because trials are variable length and neural preprocessing is trial-local, while vectorizing the neuron-speed correlation itself via sufficient-statistic array operations.

## 13-c. What processing does the code repeat multiple times?

i. Each trial is traversed up to three times (outcomes, neural/QC, then labels), and fluorescence and neuropil are separately read plane-by-plane. Position/speed/lick slices are also revisited.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):  # outcomes
...
for trial, (start, stop) in enumerate(zip(starts, stops)):  # neural
...
for trial in retained_trial_indices:  # labels
```

iii. The passes separate original-trial outcome history, all-trial neural QC, and retained-trial output construction; this simplifies correctness but repeats indexing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes deconvolved events even for bad-lick trials, then discards those event matrices (though their dF/F contributes to interneuron QC). It also reads/checks the environment stream but saves scene-derived values, and computes dF/F only to retain events plus correlation statistics.

ii.
```python
dff, events = compute_dff_and_events(f, f_neu)
...
if bad_lick_sensor:
    continue
...
stream_environment = int(round(float(np.median(env_stream[start:stop]))))
```

iii. The agent deliberately includes all valid neural trials in interneuron correlation, but OASIS output for excluded trials is not needed. The environment work is a defensive cross-check rather than a saved signal.


