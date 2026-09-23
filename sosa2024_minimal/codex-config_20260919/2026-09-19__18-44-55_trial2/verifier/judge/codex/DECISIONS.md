# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globbed every `*_behavior+ophys.nwb` file under `sub-*` directories below the data root, sorted them by subject/session parsed from the path, and processed each NWB file with `h5py`. Each file became one session.

ii.
```python
files = sorted(data_root.glob("sub-*/*_behavior+ophys.nwb"), key=_natural_key)
...
with h5py.File(path, "r") as nwb:
    behavior = nwb[BEHAVIOR]
```

iii. In the trajectory, the agent first enumerated the repository and data tree, then reported that the release contained 152 sessions from 11 switch-task mice and that it would preserve the session/trial organization while implementing the converter (steps 9, 26, 59).

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB subject id, then deduplicated and sorted numerically at the dataset level.

ii.
```python
subject = _decode_scalar(nwb["general/subject/subject_id"][()])
...
subjects = sorted(set(session_subjects), key=lambda value: int(re.search(r"\d+", value).group()))
subject_lookup = {subject: index for index, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[s] for s in session_subjects], dtype=np.int64)
```

iii. The trajectory shows the agent audited the available subjects across files before implementation and later summarized the finished dataset as 11 mice (steps 14, 59, 67).

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order is the sorted file order defined by `_natural_key`.

ii.
```python
def _natural_key(path: Path):
    subject = re.search(r"sub-m(\d+)", str(path))
    session = re.search(r"ses-(\d+)", path.name)
    return (int(subject.group(1)), int(session.group(1)))

for index, path in enumerate(files, start=1):
    arrays, info = convert_session(path)
```

iii. The agent explicitly described the source as a session-organized NWB release and later reported full-dataset counts in units of sessions (steps 9, 26, 59).

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start > 0` to `teleport > 0`. The converter slices every stream with those `[start:stop)` frame indices.

ii.
```python
def _trial_bounds(behavior) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(_read_behavior(behavior, "trial_start") > 0)
    stops = np.flatnonzero(_read_behavior(behavior, "teleport") > 0)
    ...
    return starts, stops
...
for trial, (start, stop) in enumerate(zip(starts, stops)):
```

iii. The agent inspected sample sessions around trial starts and teleports before coding, then stated that only the on-track interval from each trial-start event up to, but not including, its teleport would be retained (steps 15, 26).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not use the reference solution’s `<50` frame filter. Instead, it excludes trials whose lick trace fails the paper’s faulty-sensor criterion: more than 30% of samples have cumulative lick count above 2. All other complete trials are retained.

ii.
```python
def _bad_lick_trials(lick: np.ndarray, starts: np.ndarray,
                     stops: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.mean(lick[start:stop] > 2) > 0.30
        for start, stop in zip(starts, stops)
    ], dtype=bool)
...
if bad_lick[trial]:
    continue
```

iii. The agent explored the paper notebook parameter `correction_thr = 0.3`, counted the exact 81 matching trials across the dataset, and then justified excluding them because they cannot provide a valid lick target for the decoder (steps 24, 26).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` and `Neuropil` ROI response series, restricted to manually curated `iscell` ROIs. The NWB `Deconvolved` signal is not used.

ii.
```python
fluorescence_group = nwb[f"{OPHYS}/Fluorescence"]
neuropil_group = nwb[f"{OPHYS}/Neuropil"]
...
is_cell = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation/iscell"][:, 0].astype(bool)
```

iii. The agent inspected raw NWB contents, compared `Fluorescence`, `Neuropil`, and `Deconvolved`, and decided to recompute the paper’s signal rather than trust the stored deconvolution (steps 9, 10, 15, 23, 26).

## 2-b. How is the `neural` data processed?

i. For each trial, the AI concatenates the kept ROIs across planes, subtracts `0.7 * neuropil`, adds back each ROI’s within-trial neuropil mean, computes a maximin baseline with Gaussian smoothing then 300-sample min/max filters, converts to dF/F, smooths with a 2-sample Gaussian, and runs OASIS deconvolution with `tau=0.7`. Unlike the human reference, it always processes isolated `[start:stop)` trial windows and does not implement the session/day-specific `keep_teleports` behavior.

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

iii. The trajectory shows the agent read the paper methods and preprocessing code, then justified using trial-wise neuropil correction, maximin dF/F, two-sample smoothing, and OASIS as “paper-matched neural preprocessing” (steps 9, 11-18, 26, 67). The trace does not show any justification for omitting the reference solution’s `keep_teleports` branch.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC has two stages: keep only manually curated `iscell` ROIs, then remove putative interneurons whose dF/F has Pearson correlation greater than `0.5` with running speed across the session’s trial samples.

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

iii. The agent inspected the paper code for the speed-correlation interneuron screen and later summarized that it would keep manually curated ROIs and apply the `>0.5` exclusion (steps 17, 18, 26).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by constructing one trial matrix per `[trial_start:teleport)` interval. No additional temporal shift is applied after splitting.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    neural_trials.append(np.ascontiguousarray(events_by_trial[trial][keep_neuron]))
```

iii. The agent stated that the decoder should be time-aligned and that it would retain only the on-track interval from trial start to teleport, which is exactly how it defined per-trial neural matrices (steps 9, 26).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native aligned sampling resolution of `1 / 15.5078125` s, about `64.48 ms`, with no temporal rebinning.

ii.
```python
NOMINAL_RATE_HZ = 15.5078125
...
rate = float(1.0 / np.median(np.diff(timestamps)))
...
"time_bin_size": 1000.0 / NOMINAL_RATE_HZ,
```

iii. The agent checked the timestamp spacing and explicitly justified that two-plane sessions still have an effective `64.48 ms` bin despite a misleading 31 Hz imaging attribute (steps 21, 26, 67).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived indirectly from the behavior timestamp spacing: `_sampling_rate()` estimates one global sampling rate from `position/timestamps`, and then per-trial time is reconstructed from frame index.

ii.
```python
timestamps = np.asarray(behavior["position/timestamps"])
rate = float(1.0 / np.median(np.diff(timestamps)))
...
np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz)
```

iii. In the trajectory the agent emphasized checking the true sampling interval before choosing any resampling or alignment strategy, then used the verified constant rate instead of subtracting stored timestamps within each trial (steps 9, 21, 26).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI creates a regularly sampled vector `0, 1/rate, 2/rate, ...` for each retained trial, with length equal to the trial’s number of frames.

ii.
```python
input_trial = np.vstack([
    np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
    ...
])
```

iii. The agent’s justification was that the aligned frame rate was constant across sessions once the two-plane issue was handled, so trial-relative time could be reconstructed directly from frame number (steps 21, 26).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector is built with the same `timepoints = stop - start` used for the neural trial matrix.

ii.
```python
timepoints = stop - start
...
neural_trials.append(np.ascontiguousarray(events_by_trial[trial][keep_neuron]))
input_trial = np.vstack([
    np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
```

iii. The trajectory shows the agent treated synchronized behavior timestamps as the source of truth for the sampling interval and then used shared trial slices for all modalities (steps 9, 21, 26).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii.
```python
environment = _read_behavior(behavior, "environment")
...
trial_env = int(np.rint(np.median(environment[start:stop])))
```

iii. The agent inspected the per-session behavior channels and saw that `environment` encoded the binary condition, then carried that into the decoder inputs (steps 14, 15, 19).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI reduces the within-trial `environment` samples to one constant per trial by taking the median value over the trial and broadcasting it across all timepoints.

ii.
```python
trial_env = int(np.rint(np.median(environment[start:stop])))
...
np.full(timepoints, trial_env, dtype=np.float32),
```

iii. The trajectory indicates the agent treated environment as a trial-level condition and cross-checked session switch structure before conversion (steps 19, 35).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavior `trial number` time series, not from the trial loop counter. The AI takes the median trial-number value over each retained trial.

ii.
```python
trial_number = _read_behavior(behavior, "trial number")
...
source_trial_number = float(np.median(trial_number[start:stop]))
```

iii. During exploration the agent inspected the `trial number` channel around starts and teleports and then used it directly in the converter rather than recreating trial indices from the loop counter (step 15).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI computes the median `trial number` value inside each `[start:stop)` interval and broadcasts that constant over all trial timepoints.

ii.
```python
source_trial_number = float(np.median(trial_number[start:stop]))
...
np.full(timepoints, source_trial_number, dtype=np.float32),
```

iii. The trajectory provides only indirect justification: the agent had already verified that trial segmentation and the `trial number` stream were consistent enough within the retained on-track interval to use for trial labels (step 15).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` together with the aligned behavior timestamps and the trial start/stop frame indices.

ii.
```python
outcomes = _trial_outcomes(
    np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
)
```

iii. The agent explored reward events and their relation to trial structure before conversion, then used those reward timestamps to build per-trial reward outcomes (steps 19, 26).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first marks each trial as rewarded if any reward timestamp falls within that trial’s time window. It then assigns each retained trial the previous source trial’s reward outcome, with the first trial set to `0`.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
    right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
    outcomes[trial] = right > left
...
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
```

iii. The trajectory shows the agent was explicitly building a decoder target/input representation around complete trials and reward events rather than using raw reward amplitudes (steps 19, 26).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavior `position` time series and the inferred reward-zone label for that trial. The reward-zone label itself is inferred from `reward_zone` activity plus position and experiment day.

ii.
```python
position = _read_behavior(behavior, "position")
rzone = _read_behavior(behavior, "reward_zone")
...
zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
...
_distance_class(position, zone)
```

iii. The agent explored how `reward_zone` behaved in the NWB files, noted that nonzero samples localized the active zone and that switch days split around trial 30, then used that structure for zone inference (steps 19, 35).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes the signed distance from current position to the nearest edge of the active reward zone: negative before the zone, zero inside, positive after.

ii.
```python
start, end = ZONE_STARTS[zone], ZONE_ENDS[zone]
distance = np.where(position < start, position - start,
                    np.where(position > end, position - end, 0.0))
```

iii. The trajectory shows the agent validated reward-zone inference against session switch structure, then used the known A/B/C intervals to derive reward-relative position targets (steps 19, 35).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is thresholded into 7 categories using boundaries equivalent to `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
return np.select(
    [distance < -50, distance < -10, distance < 0, distance == 0,
     distance <= 10, distance <= 50],
    [0, 1, 2, 3, 4, 5], default=6,
).astype(np.int8)
```

iii. The trajectory does not contain a separate discussion of the bins; the AI appears to have taken them directly from the decoder instructions while using its inferred reward-zone labels (steps 26, 35).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same `[start:stop)` trial slice used for the neural activity and by generating one categorical distance value per retained frame.

ii.
```python
output_trial = _output_matrix(
    position[start:stop], speed[start:stop], lick[start:stop],
    int(zones[trial]), int(outcomes[trial]),
)
```

iii. The agent’s alignment strategy throughout the trajectory was to use shared trial slices for every modality once trial boundaries were fixed (steps 9, 26).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii.
```python
position = _read_behavior(behavior, "position")
...
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. The agent inspected the raw position channel early in the trajectory and used it directly as the source for corridor-position decoding (steps 15, 26).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the position trace per trial and discretizes each frame’s position into corridor bins.

ii.
```python
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. No extra justification appears in the trajectory beyond using the decoder’s required absolute-position target and the already explored VR position channel (steps 15, 26).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It uses 5 bins defined by thresholds `90, 180, 270, 360`, yielding the categories `<90`, `90-180`, `180-270`, `270-360`, `>360`.

ii.
```python
absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. The binning follows the decoder specification; the trajectory contains no separate alternative considered by the agent.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned frame-by-frame using the same retained trial slice as the neural data.

ii.
```python
output_trial = _output_matrix(
    position[start:stop], speed[start:stop], lick[start:stop],
```

iii. The agent consistently relied on common trial slices for all time-varying streams (steps 9, 26).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii.
```python
lick = _read_behavior(behavior, "lick")
...
lick_binary = (lick > 0).astype(np.int8)
```

iii. The trajectory shows the agent inspected the raw lick channel, found values greater than 1, and separately identified faulty lick trials before final conversion (steps 15, 24, 26).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Within retained trials, any positive lick value is binarized to `1`; zero stays `0`.

ii.
```python
lick_binary = (lick > 0).astype(np.int8)
```

iii. The agent justified additional trial exclusion because some trials had faulty cumulative lick traces, but for retained trials the output itself is a simple binary threshold (steps 24, 26).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned frame-by-frame using the same `[start:stop)` slice used for neural, position, and speed.

ii.
```python
output_trial = _output_matrix(
    position[start:stop], speed[start:stop], lick[start:stop],
```

iii. The agent’s overall alignment decision was to keep native synchronized frame slices after trial segmentation (steps 9, 26).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone`, `position`, and `general/session_id` (experiment day).

ii.
```python
experiment_day = int(_decode_scalar(nwb["general/session_id"][()]))
rzone = _read_behavior(behavior, "reward_zone")
zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
```

iii. The trajectory shows the agent inspected `reward_zone` event patterns across regular and switch sessions and used the known switch-day structure to infer the active zone label (steps 19, 35).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the AI looks for frames with `reward_zone > 0`, uses the median position of those frames to assign the closest zone center, then fills all trials in each session phase with the modal observed zone. On switch days, phases are split into trials `0:30` and `30:end`; otherwise the whole session is one phase.

ii.
```python
mask = rzone[start:stop] > 0
if np.any(mask):
    event_position = float(np.median(position[start:stop][mask]))
    observed[trial] = int(np.argmin(np.abs(ZONE_CENTERS - event_position)))
...
boundaries = [0, min(30, len(starts)), len(starts)] if experiment_day in SWITCH_DAYS else [0, len(starts)]
...
labels[left:right] = np.bincount(valid, minlength=3).argmax()
```

iii. The agent justified this by cross-checking every session’s inferred zones against the NWB session identifiers and switch conditions, and reported that the inference matched all A/B/C and switch patterns it expected (steps 19, 35).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` plus the aligned behavior timestamps and trial boundaries.

ii.
```python
outcomes = _trial_outcomes(
    np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
)
```

iii. The agent explored reward events in the sample sessions and then used them to build binary trial outcomes (steps 19, 26).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each source trial is labeled rewarded if any reward timestamp falls between that trial’s start and stop times. The resulting binary outcome is then broadcast across all frames of the retained trial.

ii.
```python
left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
outcomes[trial] = right > left
...
np.full(timepoints, rewarded, dtype=np.int8),
```

iii. The trajectory shows no more complex handling than “reward occurred in this trial or not”; that matches the decoder target the agent was building (steps 19, 26).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly uses fail-fast checks rather than repair. It raises errors if trial starts/stops are inconsistent, if the inferred sampling rate is unexpected, if a phase has no inferable reward-zone trials, if neural processing creates non-finite values, or if neuron filtering removes all cells. Missing reward-zone observations on some trials are implicitly filled by the modal phase label if at least one trial in that phase has a usable `reward_zone` event.

ii.
```python
if len(starts) != len(stops):
    raise ValueError(...)
if not np.isclose(rate, NOMINAL_RATE_HZ, rtol=0, atol=1e-4):
    raise ValueError(...)
if len(valid) == 0:
    raise ValueError(...)
if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(events)):
    raise ValueError(...)
if not np.any(keep_neuron):
    raise ValueError(...)
```

iii. The trajectory shows the agent debugging one real NWB layout issue, the two-plane ROI mapping in m17/m18, by inspecting the file structure and patching the code rather than silently tolerating the mismatch (steps 43-45, 53). Apart from that, its strategy was validation and hard failure, not the reference solution’s defensive cropping and short-trial skipping.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading large fluorescence/neuropil arrays from every NWB file, trial-wise dF/F and OASIS computation for every retained source trial, the session-wide speed-correlation accumulation, and final assembly/writing of the large pickle.

ii.
```python
for start, stop in zip(starts, stops):
    fluorescence = np.concatenate([...], axis=0)
    neuropil = np.concatenate([...], axis=0)
    dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent explicitly reported that trial-wise processing was the main runtime path and that final in-memory assembly and pickle writing would be the heaviest phase of the full run (steps 39, 59).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar/trial-wise: `_trial_outcomes`, `_reward_zone_labels`, `_bad_lick_trials`, the main trial loop that runs dF/F and OASIS one trial at a time, and the later retained-trial loop that builds input/output matrices. The reward-zone and bad-lick computations are the easiest candidates for vectorization; the neural loop is harder because trials have variable lengths and OASIS is run per trial.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
for start, stop in zip(starts, stops):
    ...
for trial, (start, stop) in enumerate(zip(starts, stops)):
    if bad_lick[trial]:
        continue
```

iii. The trajectory never claims that these loops were optimized; instead, the agent justified them pragmatically as the memory-bounded way to process the full dataset (step 39).

## 13-c. What processing does the code repeat multiple times?

i. The code loops over all source trials twice: first to compute dF/F/events and accumulate speed-correlation statistics, then again to build retained-trial neural/input/output arrays. It also rereads per-trial slices of `position`, `speed`, and `lick` in both trial passes.

ii.
```python
for start, stop in zip(starts, stops):
    ...
    dff, events = _dff_and_events(...)
...
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    output_trial = _output_matrix(
        position[start:stop], speed[start:stop], lick[start:stop],
```

iii. The trajectory frames this as a deliberate tradeoff: trial-wise processing kept memory bounded during conversion, even though it meant reusing the same trial boundaries and behavior slices across multiple passes (step 39).

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes dF/F for every source trial even though only deconvolved events are saved; dF/F is kept only long enough to support OASIS and the interneuron screen. It also computes reward-zone and reward-outcome labels for all source trials, including excluded bad-lick trials, and stores extensive `session_info` metadata that the decoder itself does not need.

ii.
```python
dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
...
events_by_trial.append(events)
...
"session_info": session_info,
```

iii. The trajectory shows the agent was optimizing for a validated, inspectable dataset rather than a minimal one; it repeatedly reported audit statistics, excluded-trial lists, and per-session metadata as part of its integrity checks (steps 35, 59, 67).
