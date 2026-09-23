# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data/sub-*/*.nwb`, sorts them numerically by mouse and session, and opens each session with `pynwb.NWBHDF5IO`. It processes sessions one file at a time rather than building a separate survey pass.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_session_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    ...
    return files

with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. `CONVERSION_NOTES.md` says each NWB file is one session in a one-level subject/session directory layout and explicitly states that all NWB inspection used PyNWB rather than direct HDF5 parsing.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the NWB session metadata (`nwb.subject.subject_id`) and then deduplicated/sorted at the end to form `data['subjects']`.

ii.
```python
subject_id = str(nwb.subject.subject_id)
...
subject_ids.append(converted["subject"])
...
subjects = sorted(set(subject_ids), key=lambda value: int(re.search(r"\d+", value).group()))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[subject] for subject in subject_ids], dtype=np.int64)
```

iii. The notes describe the dataset as 11 subject directories (`sub-m3`, `sub-m4`, ...), and the AI treated the NWB subject metadata as authoritative for the exported subject list and per-session subject index.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session day is parsed from the filename with `natural_session_key`, and the converted output stores one top-level session entry per file.

ii.
```python
def natural_session_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    ...
    return int(match.group(1)), int(match.group(2))

for session_idx, path in enumerate(files):
    converted, info = process_session(path, show_processing=plot_this)
    neural.append(converted["neural"])
    inputs.append(converted["input"])
    outputs.append(converted["output"])
```

iii. `CONVERSION_NOTES.md` states that each `sub-<mouse>_ses-<day>_behavior+ophys.nwb` file is one session and that session ordering is natural numeric mouse/day order.

## 1-d. How are the data split into trials?

i. Trials are split with `trial_start` as the start index and any positive `teleport` sample as the stop index, then each trial is emitted as the slice `[start:stop)`.

ii.
```python
starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial bounds in {path.name}")
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
    neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The notes say the `trials` table is absent in NWB, so boundaries must come from the frame-aligned `trial_start` and `teleport` behavior streams. The AI also records that it wanted the emitted trial to be the track interval `[trial_start, teleport)`.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes any trial whose lick trace looks corrupted, defined as more than 30% of samples having cumulative lick counts greater than 2. It also raises an error if a retained trial has fewer than 2 frames, but it does not use the human reference’s `<50`-frame threshold.

ii.
```python
corrupt_lick = np.asarray([
    np.mean(lick[start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, stops)
])
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    if corrupt_lick[raw_idx]:
        continue
    if stop - start < 2:
        raise ValueError(f"Trial {raw_idx} in {path.name} has fewer than two frames")
```

iii. The notes justify this by citing the paper’s corrupt-lick-trial rule and arguing that categorical lick outputs cannot represent unknown/invalid lick labels, so the AI chose to drop the 81 corrupt trials rather than keep questionable lick targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB `Fluorescence` and `Neuropil` ROI response series, restricted to manually curated `iscell` ROIs from `ImageSegmentation`. The AI explicitly does not use the NWB `Deconvolved` series as final neural input.

ii.
```python
segmentation = ophys["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
fluorescence_series = ophys["Fluorescence"].roi_response_series
neuropil_series = ophys["Neuropil"].roi_response_series
...
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
neuropil = np.asarray(n_series.data[:len(speed), local_keep], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the NWB `Deconvolved` array is the Suite2p export rather than the paper’s analyzed signal, so the AI decided to recompute the paper’s own dF/F and OASIS events from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. For each trial and plane, the AI subtracts `0.7 * neuropil`, adds back the trial neuropil mean, estimates a maximin baseline with Gaussian smoothing and 300-sample min/max filters, computes dF/F, smooths with a 2-sample Gaussian, and then applies OASIS deconvolution. It does this only within emitted trial windows.

ii.
```python
corrected = f_trial - NEUROPIL_COEF * n_trial
corrected += NEUROPIL_COEF * np.mean(n_trial, axis=0, keepdims=True)

baseline = gaussian_filter1d(corrected, 15, axis=0)
baseline = minimum_filter1d(baseline, 300, axis=0)
baseline = maximum_filter1d(baseline, 300, axis=0)
trial_dff = gaussian_filter1d((corrected - baseline) / np.abs(baseline), 2, axis=0)
...
events[start:stop] = dcnv.oasis(
    dff[start:stop].T, 2000, OASIS_TAU_S, EFFECTIVE_FS_HZ
).T.astype(np.float32, copy=False)
```

iii. The notes say this was intended to reproduce the paper’s “trial-wise maximin dF/F and OASIS” pipeline. The AI also argues in the notes that explicit behavior timestamps show a common effective 15.5 Hz per-plane sampling interval, so it hard-codes `EFFECTIVE_FS_HZ = 15.5078125`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps only manually curated `iscell` ROIs, then drops putative interneurons whose dF/F has Pearson correlation greater than 0.5 with running speed.

ii.
```python
iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
...
local_keep = np.flatnonzero(iscell[roi_ids])
...
correlations = speed_correlations(dff, speed, starts, stops)
keep = np.isfinite(correlations) & (correlations <= 0.5)
plane_events.append(events[:, keep])
```

iii. The notes explicitly cite the paper’s two curation rules: manual ROI curation via `iscell` and a speed-correlation cutoff to exclude putative interneurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of the trial by slicing each trial as `[trial_start, teleport)`. Time zero is therefore the `trial_start`-flagged sample.

ii.
```python
starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
...
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The notes say the alignment event is “trial_start flag / entry to 0-cm track,” so no additional temporal shifting is applied after trial splitting.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data keeps the native synchronized frame rate at about 15.5078 Hz, i.e. `64.4836 ms` bins. No rebinning or interpolation is applied.

ii.
```python
EFFECTIVE_FS_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / EFFECTIVE_FS_HZ
...
"time_bin_size": float(TIME_BIN_MS),
```

iii. The notes say inspected behavior timestamps had a median spacing of `0.0644836 s`, including sessions whose metadata nominally reported 31 Hz because the two planes were already interleaved into one per-plane-aligned stream.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the timestamps attached to the `position` behavior time series.

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The notes say behavior streams are frame-synchronous and share the same timestamps, so using the `position` timestamps was sufficient for the per-frame trial clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI subtracts the timestamp at trial start from each timestamp in the trial, producing a continuous per-frame elapsed-time series.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
inputs = np.vstack([
    time_from_start,
    ...
])
```

iii. The justification is implicit in the task requirement and restated in the notes: align all streams to trial start and preserve the small timestamp jitter instead of imposing a synthetic uniform clock.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start:stop)` indices are used for behavior and neural arrays, so the time input and neural activity have identical frame counts and are aligned sample-by-sample within each trial.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError("Neural/input/output time axes differ")
```

iii. The notes repeatedly state that imaging and behavior are already synchronized at one row per imaging frame, so trial slicing is the only required alignment operation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavior time series.

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float32)
...
env_values = np.unique(environment[start:stop])
```

iii. The notes say the frame-level environment values are `0` for ENV1 and `1` for ENV2, with a single constant environment per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI checks that exactly one valid environment label is present and then repeats that value across all timepoints in the trial.

ii.
```python
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(f"Trial {raw_idx} has invalid environment values {env_values}")
...
np.full(stop - start, env_values[0], dtype=np.float32)
```

iii. The notes justify this by saying environment never changes within a trial, so validating constancy is a useful consistency check before filling the per-frame decoder input array.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the `trial number` behavior time series, specifically the value at the first frame of each trial.

ii.
```python
trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
...
source_trial_number = trial_number[start]
if not np.isclose(source_trial_number, raw_idx):
    raise ValueError(
        f"Trial-number mismatch in {path.name}: row {raw_idx}, value {source_trial_number}"
    )
```

iii. The AI’s justification is that the native trial-number field can be used as long as it agrees with the chronological trial index implied by `trial_start`/`teleport`; it enforces that agreement with an assertion.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. After validating that the field matches the chronological raw trial index, the AI repeats that scalar trial number across every timepoint in the trial.

ii.
```python
inputs = np.vstack([
    time_from_start,
    np.full(stop - start, env_values[0], dtype=np.float32),
    np.full(stop - start, source_trial_number, dtype=np.float32),
    ...
])
```

iii. The notes describe per-trial variables as repeated over time because the decoder format mixes time-varying and per-trial variables in one rectangular array.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward` event timestamps together with the current session’s trial start/stop timestamps.

ii.
```python
def reward_outcomes(reward_times: np.ndarray, timestamps: np.ndarray,
                    starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    reward_times = np.sort(np.asarray(reward_times, dtype=np.float64))
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)
```

iii. The notes say sparse reward timestamps are authoritative for reward delivery and that reward omission trials must be preserved because current-trial and previous-trial reward outcomes are decoder targets/inputs.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary rewarded/omitted outcome for every raw trial. Then, for each emitted trial, it uses the immediately preceding raw trial’s outcome; the first trial gets `0`.

ii.
```python
outcomes = reward_outcomes(
    np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops
)
...
previous_outcome = outcomes[raw_idx - 1] if raw_idx > 0 else 0
...
np.full(stop - start, previous_outcome, dtype=np.float32)
```

iii. The notes explicitly say that if a corrupt-lick trial is excluded, the next retained trial should still reference the actual previous raw trial rather than the previous retained trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus the reward-zone identity parsed from `NWBFile.identifier` for the current session/trial schedule.

ii.
```python
scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
...
trial_position = position[start:stop]
zone_label = str(zone_labels_raw[raw_idx])
zone_start, zone_stop = ZONE_BOUNDS[zone_label]
```

iii. The notes say the identifier strings encode fixed and switch schedules directly (for example, `Env1_LocationB_to_A`), and the AI treated that schedule as cleaner and more authoritative than inferring zones from noisy `reward_zone` traces.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes signed distance to the nearest point of the active reward-zone interval: negative before the zone, zero inside it, positive after it. It then discretizes those distances into seven classes.

ii.
```python
signed_distance = np.where(
    trial_position < zone_start,
    trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
outputs = np.vstack([
    distance_classes(signed_distance),
    ...
])
```

iii. `CONVERSION_NOTES.md` says the instruction “distance to any location in the reward zone” was interpreted as distance to the closed interval `[zone_start, zone_stop]`, which makes the entire zone map to exact distance 0.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Thresholding is implemented manually with boolean comparisons that realize the seven requested bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
def distance_classes(distance: np.ndarray) -> np.ndarray:
    out = np.full(distance.shape, 6, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    return out
```

iii. The notes and code comments say the boundaries were chosen to match the instruction exactly, and `check_discretization_boundaries()` asserts the intended class assignments on synthetic edge values.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance from the per-trial `position[start:stop]` slice that shares the same frame indices as the emitted neural trial.

ii.
```python
trial_position = position[start:stop]
...
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The notes say behavior and imaging are already frame-synchronous, so reusing the same trial indices is the intended alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
...
trial_position = position[start:stop]
```

iii. The notes describe `position` as the synchronized VR corridor position in centimeters over the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the position trace trial-by-trial and discretizes it into five 90 cm bins covering the track.

ii.
```python
outputs = np.vstack([
    distance_classes(signed_distance),
    position_classes(trial_position),
    ...
])
```

iii. The notes say this follows the decoder specification rather than the paper’s 10 cm spatial binning used for other analyses.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholding is implemented manually as five bins: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360`.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    out = np.full(position.shape, 4, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    return out
```

iii. The notes say the boundary behavior was chosen to match the instruction exactly, especially the special case that exactly `360 cm` belongs to the fourth bin and only values `>360` belong to the last bin.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial frame slice as the neural and other behavioral arrays, so no extra alignment is performed.

ii.
```python
trial_position = position[start:stop]
...
if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError("Neural/input/output time axes differ")
```

iii. The notes justify this by saying all streams are synchronized row-by-row at the imaging frame rate.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
...
trial_lick = lick[start:stop]
```

iii. The notes describe this signal as frame-synchronous cumulative lick counts, often greater than 1 within a frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the trial lick trace so that any positive value becomes class `1` and zero becomes class `0`.

ii.
```python
outputs = np.vstack([
    ...
    (trial_lick > 0).astype(np.int8),
    ...
])
```

iii. The notes say binary `lick>0` matches the paper’s lick quantification and the decoder’s required `no/yes` label format.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same `[start:stop)` slice as the neural trial.

ii.
```python
trial_lick = lick[start:stop]
...
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The notes say all behavior streams are already synchronized to imaging frames, so trial slicing preserves alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session identifier string (`nwb.identifier`), which is parsed into an A/B/C reward-zone schedule.

ii.
```python
scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
...
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
```

iii. The notes say the scene identifiers explicitly encode all fixed and switch schedules, so parsing them is simpler and more reliable than reconstructing zone identity indirectly from noisy `reward_zone` activity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed sessions assign the same A/B/C zone to all trials. Switch sessions assign one zone for the first 30 trials and a second zone thereafter, matching the parsed identifier string. The resulting class is then repeated over all frames in the trial.

ii.
```python
def parse_zone_schedule(scene: str, n_trials: int) -> np.ndarray:
    if "_to_" in scene:
        ...
        labels = [left_match.group(1)] * min(30, n_trials)
        labels.extend([right_match.group(1)] * max(0, n_trials - 30))
        return np.asarray(labels)
    ...

np.full(stop - start, ZONE_TO_CLASS[zone_label], dtype=np.int8)
```

iii. The notes explicitly tie this to the paper’s rule that switch sessions change reward zone after trial 30.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse `Reward` event timestamps together with the session’s trial boundaries.

ii.
```python
outcomes = reward_outcomes(
    np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops
)
```

iii. The notes say sparse delivery events are the correct source for reward outcome and that outcomes should be preserved because rewarded versus omitted is itself one of the decoder outputs.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A raw trial is labeled rewarded if any reward timestamp falls between that trial’s start timestamp and stop timestamp; otherwise it is labeled omitted. The resulting binary label is repeated across the trial’s frames.

ii.
```python
def reward_outcomes(...):
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)

...
np.full(stop - start, outcomes[raw_idx], dtype=np.int8)
```

iii. The notes report that this yields the expected approximately 85% reward rate and ignores the few reward events that occur outside valid track epochs.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly uses validation plus targeted trimming rather than broad repair. It trims at most one unmatched terminal ophys frame if the ophys stream is one sample longer than behavior, filters corrupt lick trials, raises errors on invalid environment/trial-number/trial-boundary conditions, and raises if nonfinite converted data remain.

ii.
```python
extra_tail = n_ophys_frames - len(speed)
if extra_tail > 1:
    raise ValueError(...)
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
...
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(...)
...
if not (np.all(np.isfinite(neural)) and np.all(np.isfinite(inputs))):
    raise ValueError(f"Nonfinite converted data in {path.name}, trial {raw_idx}")
```

iii. The notes justify this as a deliberate preference for strict consistency checks while making only one small data repair that was observed in ten sessions: a single extra terminal ophys frame after the last behavior sample.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading large NWB arrays, recomputing dF/F plus OASIS events, and writing the final pickle. The AI explicitly avoided a separate survey pass to reduce duplicate I/O.

ii.
```python
dff, events, trace = trial_dff_and_events(
    fluorescence, neuropil, starts, stops,
    capture_trace=capture_trace and plot_trace is None,
)
...
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 and Step 7 of the notes say the 92 GB dataset makes NWB reads and neural preprocessing the dominant costs, and the runtime estimate is explicitly based on neural workload.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the speed-correlation calculation across neurons, but trial-wise loops remain in `trial_dff_and_events` and `process_session`. Those per-trial loops could only be removed with padding/masking or a substantially different representation.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
for start, stop in zip(starts, stops):
    events[start:stop] = dcnv.oasis(...)
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes explicitly mention vectorized filtering/correlation as an optimization already added and keep the per-trial/session structure because the data are variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. Within one session, the code iterates over trials multiple times: once during dF/F computation, again during OASIS event generation, again while computing/using trial-level outputs, and optionally again while preparing diagnostic plot inputs. It does not repeat the whole-dataset survey/conversion pass used by the human reference.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
for start, stop in zip(starts, stops):
    events[start:stop] = dcnv.oasis(...)
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes say the AI intentionally avoided a second “survey” sweep over every NWB file and instead tried to keep repeated work local to one session and one plane at a time.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In the default conversion path, there is little obviously unnecessary work. The main discardable work is optional diagnostics: `capture_trace`, the per-session processing plot inputs, and stored continuous `positions`/`speeds`/`zone_labels` used only when `--show-processing` is enabled.

ii.
```python
if capture_trace and trial_idx == trace_trial and fluorescence.shape[1] > 0:
    trace = {...}
...
kept_positions: list[np.ndarray] = []
kept_speeds: list[np.ndarray] = []
kept_zone_labels: list[str] = []
...
if show_processing:
    plot_path = plot_processing(...)
```

iii. The notes describe these as sanity-check machinery rather than part of the exported dataset. In the normal path, the main intermediate arrays (`dff`, `events`, raw fluorescence/neuropil) are produced because they are needed for curation or final neural output, then deleted.
