# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter recursively discovers every NWB under `/app/data/sub-*`, sorts the paths deterministically, and reads each with `pynwb.NWBHDF5IO`. Full mode processes all 152 files; sample mode deliberately selects two.

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes say the 152 files represent all 11 mice and explicitly emphasize that all inspection and conversion used `pynwb`, as required.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from file names, checked against `nwb.subject.subject_id`, then unique IDs are numerically sorted and mapped to `subject_idx`.

ii.
```python
subject_from_path, day = session_key(path)
subject = nwb.subject.subject_id
if subject != subject_from_path: raise ValueError(...)
subjects = sorted({x["subject"] for x in converted_sessions}, key=lambda x: int(x[1:]))
```

iii. The consistency check reports 11 subjects, matching the paper and source data.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session, identified by subject and `ses-<day>` in its filename. Sessions remain in sorted file order.

ii.
```python
match = re.search(r"sub-(m\d+)_ses-(\d+)", path.name)
for idx, path in enumerate(files):
    converted, metadata = convert_session(path, ...)
```

iii. The agent found 152 files/sessions and documented that each NWB contains one aligned behavior/ophys session.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and stops are positive `teleport` samples; slices include the start and exclude the teleport.

ii.
```python
starts = np.flatnonzero(b["trial_start"] > 0)
stops = np.flatnonzero(b["teleport"] > 0)
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
    trial_dff = dff[:, start:stop]
```

iii. The notes identify start-to-teleport as the interval used throughout the reference repository and validate matched ordered event counts in all files.

## 1-e. How are trials filtered based on quality controls?

i. A trial is excluded if it has fewer than two samples, contains non-scanning frames, has non-finite required behavior/timestamps, or has lick counts above 2 in more than 30% of frames. Every session must retain at least two trials.

ii.
```python
if stop - start < 2: reason = "fewer than two samples"
elif not np.all(b["scanning"][start:stop] == 1): reason = "outside valid scanning period"
if not np.all(np.isfinite(required)): reason = "non-finite required behavior"
if lick_bad_fraction > 0.30: reason = "corrupt lick sensor (>30% frames with count >2)"
```

iii. The agent cites the manuscript's lick-sensor corruption rule and reports exactly 81 excluded trials; it retains stationary frames because speed class 0 is required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is reconstructed from NWB `Fluorescence` (F) and `Neuropil` (Fneu), restricted to manually curated `iscell` ROIs; stored `Deconvolved` is not used.

ii.
```python
fluorescence_container = ophys["Fluorescence"]
neuropil_container = ophys["Neuropil"]
manual_ids = np.flatnonzero(iscell[:, 0] == 1)
```

iii. The notes say the stored deconvolution is not the paper's analyzed signal and therefore recreate the paper's pipeline.

## 2-b. How is the `neural` data processed?

i. For each trial, it subtracts `0.7*Fneu`, adds back the trial mean neuropil, computes a sigma-15/300-sample min-max baseline, calculates and sigma-2 smooths dF/F, then applies OASIS (`tau=.7`, 15.5078125 Hz) separately to each retained trial. Unlike the reference, it never allows baseline/deconvolution processing to span imaged teleport periods.

ii.
```python
corrected = f - 0.7 * fn + 0.7 * np.nanmean(fn, axis=1, keepdims=True)
baseline = gaussian_smooth(corrected, 15)
baseline = ndimage.minimum_filter1d(baseline, 300, axis=-1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
trial_dff = gaussian_smooth((corrected - baseline) / np.abs(baseline), 2)
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ)
```

iii. The agent says this recreates the reference dF/F and OASIS pipeline and avoids a full-session event matrix, but its notes do not address the reference `teleport_sessions` exception.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps manual Suite2p cells (`iscell[:,0]==1`) and removes cells whose dF/F–speed Pearson correlation is greater than 0.5. It does not restrict to place cells.

ii.
```python
manual_ids = np.flatnonzero(iscell[:, 0] == 1)
correlations = speed_correlations(dff, b["speed"])
keep = ~(correlations > 0.5)
dff = dff[keep]
```

iii. The paper's manual curation and putative-interneuron rule are followed; place-cell filtering was rejected because the requested targets include non-spatial behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows already share the behavior frame axis. Each trial begins at the `trial_start` index, so no resampling or shift is performed.

ii.
```python
trial_dff = dff[:, start:stop]
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. The agent validated equal neural/input/output lengths and describes the alignment event as entry into the 0-cm corridor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 15.5078125 Hz samples are retained, giving 64.483627 ms bins; no temporal rebinning is applied. Extra neural tail rows are truncated through `common_length`.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] + ...)
```

iii. The notes report uniform behavior timestamp spacing and explain that multi-plane NWB advertised rates do not change the already aligned row timebase.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It comes from the `position` time-series timestamps and `trial_start` boundaries.

ii.
```python
arrays["timestamps"] = np.asarray(behavior["position"].timestamps[:common_length])
starts = np.flatnonzero(b["trial_start"] > 0)
```

iii. Position timestamps were treated as the authoritative synchronized behavior timebase.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial's first frame is subtracted from every timestamp in that trial.

ii.
```python
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. This makes every retained trial start at exactly zero seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start:stop]` frame slice is used for timestamps and neural data, and matching lengths are asserted.

ii.
```python
if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError(...)
```

iii. Whole-dataset checks found no shape or nonmonotonic-time failures.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` series.

ii.
```python
arrays = {name: np.asarray(behavior[name].data[:common_length]) for name in names}
environment = mode_value(b["environment"][start:stop])
```

iii. The agent observed ENV1/ENV2 values 0/1 and verified day-8 transitions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The finite within-trial mode is calculated and repeated across all trial frames.

ii.
```python
environment = mode_value(b["environment"][start:stop])
np.full(stop - start, environment)
```

iii. Repetition makes per-trial context explicit in the required two-dimensional input matrix.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw behavior `trial number` time series, not from the loop index.

ii.
```python
trial_number = mode_value(b["trial number"][start:stop])
```

iii. The notes describe this as the source trial number and report a 0–99 range.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-trial mode is computed and repeated over time.

ii.
```python
np.full(stop - start, trial_number)
```

iii. This preserves original numbering even when corrupt trials are excluded.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is based on irregular `Reward.timestamps`, behavior timestamps, and the preceding source trial's start/stop interval.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:])
outcomes = np.array([np.any((reward_times >= timestamps[start]) &
                            (reward_times < timestamps[stop])) for start, stop in zip(starts, stops)])
```

iii. The agent treats reward delivery within the explicit interval as rewarded and leaves ITI rewards unmapped.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. It shifts the binary outcome array by one original trial; the first trial is zero, and the value is repeated over frames. Excluded trials still count as the immediately previous source trial.

ii.
```python
previous_outcome = int(outcomes[trial_idx - 1]) if trial_idx > 0 else 0
np.full(stop - start, previous_outcome)
```

iii. The notes report zero preceding-outcome mismatches and intentionally preserve source chronology.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus the active zone inferred from the NWB identifier's scene and original trial index (switch after trial 30), with canonical A/B/C bounds.

ii.
```python
scene = scene_from_identifier(nwb.identifier)
zones = zones_by_trial(scene, len(starts))
distance = reward_distance(position, zone)
```

iii. The agent cites the reference `get_reward_zones` logic and validated trials 29/30 to rule out switch off-by-one errors.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before a zone is measured relative to its start, position after it relative to its end, and positions inside are exactly zero; the result is then discretized.

ii.
```python
return np.where(position < start, position - start,
                np.where(position > stop, position - stop, 0.0))
```

iii. This implements signed distance to the nearest point in the active reward-zone interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create seven requested classes, with −50 and −10 in class 1, exactly zero in class 3, +10 in class 4, and +50 in class 5.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance <= -10)] = 1
out[(distance > -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. Built-in synthetic boundary tests check all edge assignments.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the identical `[start:stop]` indices; shape equality is asserted.

ii.
```python
position = b["position"][start:stop]
trial_dff = dff[:, start:stop]
```

iii. Independent raw-output checks passed exactly on selected trials.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavior `position`.

ii.
```python
position = b["position"][start:stop]
```

iii. The source records corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No continuous transform is applied before categorical discretization.

ii.
```python
discretize_position(position)
```

iii. Native synchronized values, including small endpoint excursions, are retained in extreme classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five masks implement <90, [90,180), [180,270), [270,360], and >360 cm.

ii.
```python
out[(position >= 90) & (position < 180)] = 1
out[(position >= 180) & (position < 270)] = 2
out[(position >= 270) & (position <= 360)] = 3
out[position > 360] = 4
```

iii. The thresholds are five equal 90-cm divisions of the 450-cm track and are boundary-tested.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial frame slice and is included in the time-dimension assertion.

ii.
```python
position = b["position"][start:stop]
if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]: ...
```

iii. The agent reports exact agreement in independent spot checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from behavior `lick` values.

ii.
```python
lick = b["lick"][start:stop]
```

iii. The notes describe this as a cumulative/event-count stream synchronized to imaging frames.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive value becomes 1 and all others 0; trials with severe sensor corruption are excluded first.

ii.
```python
lick_bad_fraction = float(np.mean(b["lick"][start:stop] > 2))
(lick > 0).astype(np.int16)
```

iii. This matches the requested binary output and the reference's clipping of positive counts to lick presence.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks and neural activity use the same trial-frame indices.

ii.
```python
lick = b["lick"][start:stop]
trial_dff = dff[:, start:stop]
```

iii. Processing plots and whole-dataset dimension checks were used to verify alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `nwb.identifier` scene tokens and the original trial index, rather than directly from `reward_zone` samples.

ii.
```python
scene = scene_from_identifier(nwb.identifier)
zones = zones_by_trial(scene, len(starts))
```

iii. The agent considered identifiers authoritative schedules and checked all switches after original trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. One/two zone labels are parsed; fixed sessions repeat one label, switch sessions use the first for 30 trials then the second. A/B/C map to 0/1/2 and repeat across frames.

ii.
```python
return [labels[0]] * 30 + [labels[1]] * (n_trials - 30)
np.full(stop - start, "ABC".index(zone), dtype=np.int16)
```

iii. This follows the scene/switch schedule in the reference behavior code and preserves original trial numbering after exclusions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses irregular `Reward` event timestamps and behavior timestamps defining each trial interval.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:])
```

iii. Reward delivery, rather than `autoreward` or zone entry, defines the requested binary outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded if any event time lies in `[trial_start_time, teleport_time)`; the binary value is repeated across the trial.

ii.
```python
outcomes = np.array([np.any((reward_times >= timestamps[start]) &
                            (reward_times < timestamps[stop])) ...], dtype=np.int16)
np.full(stop - start, outcomes[trial_idx], dtype=np.int16)
```

iii. The agent documents three ITI rewards as intentionally unmapped and reports an 84.64% trial reward rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Streams are trimmed to their common length; malformed trial counts/intervals, missing series/ROIs, non-finite values, and alignment failures raise errors. Invalid individual trials are logged and skipped according to the QC rules.

ii.
```python
common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] + ...)
if len(starts) != len(stops) or len(starts) < 2: raise ValueError(...)
if not np.all(np.isfinite(required)): reason = "non-finite required behavior"
```

iii. Ten two-plane files had one extra neural row, motivating common-length truncation; metadata records every excluded trial and source/retained count.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large NWB fluorescence/neuropil arrays, trial-wise baseline filtering across all curated cells, OASIS deconvolution, and writing the 9.53-GB pickle dominate runtime.

ii.
```python
fluorescence = load_selected_roi_series(...)
dff, plot_example = calculate_reference_dff(...)
events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ)
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes call loading 92 GB and cell-by-frame filtering intrinsically expensive; full conversion took 442.9 seconds.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loops over trials in dF/F computation, outcomes, and final trial assembly could partly be reduced, though variable trial lengths and per-trial baselines limit useful vectorization. File/session loops are necessary for bounded memory.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
outcomes = np.array([np.any(...) for start, stop in zip(starts, stops)])
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
```

iii. The agent instead vectorized correlations and class transforms and processed one session at a time to control memory.

## 13-c. What processing does the code repeat multiple times?

i. Trial slicing and construction occur in both dF/F and conversion loops; the first trial's arrays are copied for optional plots. Each source is otherwise loaded once per run, unlike the human reference's separate survey/conversion passes.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
    ...
for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
    ...
```

iii. The notes emphasize avoiding a full-session event matrix and processing sessions only once; repeated trial traversal is required by the chosen per-trial pipeline.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/checks several behavior fields (`reward_zone`, `autoreward`, `trial_start`, `teleport`, `scanning`) and segmentation metadata not stored as decoder variables. Optional plotting copies raw/intermediate traces; provenance metadata is also not used by training.

ii.
```python
names = ["position", "speed", "lick", "environment", "trial number", "scanning",
         "reward_zone", "autoreward", "trial_start", "teleport"]
example = {"raw_f": f[:3].copy(), ..., "dff": trial_dff[:3].copy()}
```

iii. The agent says these fields support validity/provenance checks; raw pixel masks and images are deliberately not read.
