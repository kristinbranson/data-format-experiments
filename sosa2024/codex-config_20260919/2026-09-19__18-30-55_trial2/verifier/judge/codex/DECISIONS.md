# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every `sub-*/*_behavior+ophys.nwb`, numerically sorts the files by mouse and session, and opens each with `h5py`. Full mode uses all 152 files; sample mode deliberately uses only the first two.

ii.
```python
files = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*_behavior+ophys.nwb")), key=_session_sort_key)
return files[:2] if sample else files
...
with h5py.File(path, "r") as nwb:
```

iii. The notes report 152 NWBs, 11 subjects, one file per subject/day, and explain that processing one session at a time limits memory. The full conversion reports all 152 sessions.

## 1-b. How are the data split into subjects?

i. Subject ID is read from each NWB's `general/subject/subject_id`; unique IDs are numerically sorted and each session receives an index into that list.

ii.
```python
subject = _decode(nwb["general/subject/subject_id"])
subjects = sorted({x["subject"] for x in converted}, key=lambda s: int(s[1:]))
subject_idx = np.asarray([subject_lookup[x["subject"]] for x in converted])
```

iii. The notes say this yielded all 11 imaged mice and preserved subject/session provenance.

## 1-c. How are the data split into sessions?

i. Each NWB is one session; sessions are ordered by numeric mouse ID and numeric `ses-*` day.

ii.
```python
def _session_sort_key(path):
    match = re.search(r"sub-m(\d+)_ses-(\d+)", os.path.basename(path))
    return int(match.group(1)), int(match.group(2))
...
converted.append(convert_session(path, show_processing and i < 2))
```

iii. The agent observed that the 152 files represent available imaging days and treated session identity in filenames and NWB metadata as authoritative.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples of dense `trial_start`; ends are positive samples of `teleport`. Each paired interval is sliced `[start:stop)`, excluding teleport/ITI.

ii.
```python
starts = np.flatnonzero(dense["trial_start"] > 0)
teleports = np.flatnonzero(dense["teleport"] > 0)
...
fluorescence_ds[start:stop, :]
```

iii. The notes state this is physically equivalent to the reference's 1-based `[start-1:teleport-1]` convention and matches an on-track lap.

## 1-e. How are trials filtered based on quality controls?

i. A trial is removed when more than 30% of its frames have raw cumulative lick count greater than 2. No minimum-duration filter is applied. At least two retained trials per session are required.

ii.
```python
lick_bad = np.array([np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
                     for start, stop in zip(starts, teleports)])
...
if lick_bad[i]:
    continue
```

iii. The agent describes this as the published lick-artifact criterion and reports dropping 81 of 12,216 paired trials, arguing that corrupt lick targets should not be fabricated.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from plane-0 `Fluorescence` and `Neuropil`, plus `ImageSegmentation`'s `planeIdx` and `iscell`; the stored NWB `Deconvolved` signal is not used.

ii.
```python
fluorescence_ds = neural_group["Fluorescence/plane0/data"]
neuropil_ds = neural_group["Neuropil/plane0/data"]
plane0_iscell = iscell[plane_index == 0]
```

iii. The agent correctly notes that the paper recomputes dF/F and OASIS events from F/Fneu and that NWB `Deconvolved` is a different Suite2p-stage product.

## 2-b. How is the `neural` data processed?

i. For each trial independently, it subtracts `0.7*neuropil`, restores `0.7` times that trial's mean neuropil, applies a 15-sample Gaussian, 300-sample minimum then maximum filters, computes `(F-baseline)/abs(baseline)`, smooths with sigma 2, and runs OASIS with tau 0.7 at 15.5078125 Hz. Nonfinite events become zero.

ii.
```python
corrected = fluorescence - NEUROPIL_COEF * neuropil
corrected += NEUROPIL_COEF * np.mean(neuropil, axis=1, keepdims=True)
baseline_seed = gaussian_filter(corrected, sigma=(0.0, 15.0), mode="reflect")
baseline = maximum_filter1d(minimum_filter1d(baseline_seed, 300, axis=1), 300, axis=1)
dff = gaussian_filter1d((corrected - baseline) / np.abs(baseline), 2.0, axis=1)
events = dcnv.oasis(np.asarray(dff, dtype=np.float32), 2000, 0.7, FRAME_RATE_HZ)
```

iii. The notes cite the Methods and repository parameters. They intentionally process each trial, but do not implement the reference's session-dependent `keep_teleports` baseline windows or available multi-plane pooling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps manually curated `iscell` plane-0 ROIs and removes cells whose session-wide dF/F–speed Pearson correlation exceeds 0.5. It does not restrict to place cells.

ii.
```python
roi_columns = np.flatnonzero(plane0_iscell)
...
is_interneuron = np.isfinite(speed_corr) & (speed_corr > INTERNEURON_SPEED_R)
kept_events = [x[~is_interneuron] for x in kept_events]
```

iii. The agent identifies both filters as paper curation and explains that place-cell selection is inappropriate for a general population decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows are sliced beginning at the `trial_start` sample, so time zero is track entry; variable-length data end immediately before teleport.

ii.
```python
fluorescence = np.asarray(fluorescence_ds[start:stop, :])[:, roi_columns].T
...
"temporal_alignment_event": "entry into the 450 cm track (trial_start)"
```

iii. The notes state that dense neural and behavior rows are already synchronized, making the common slice the alignment operation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It retains native 15.5078125 Hz samples (64.483627 ms) and performs no temporal rebinning.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The agent says timestamps establish a common per-plane 15.5 Hz rate and that rebinning is unnecessary.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses the dense `position/timestamps` array after verifying all dense behavioral timestamps match it.

ii.
```python
timestamps = np.asarray(behavior["position/timestamps"][:common_length])
```

iii. The notes report all dense behavioral and plane-0 neural samples are synchronized at 64.4836 ms.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first trial frame is subtracted from every timestamp in that trial.

ii.
```python
timestamps[start:stop] - timestamps[start]
```

iii. This makes every retained trial start at exactly zero while preserving native sampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `[start:stop)` row slice is used, and equal time dimensions are asserted.

ii.
```python
if events.shape[1] != T or input_data.shape != (4, T):
    raise AssertionError(...)
```

iii. The agent verified timestamp identity and validates the time axis against `arange(T)/FRAME_RATE_HZ`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is read from the dense `environment` behavior series.

ii.
```python
env_values = np.unique(dense["environment"][start:stop])
```

iii. The agent found it is constant per trial and encoded 0/1 for ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code validates a single value in `{0,1}` and repeats it across all frames.

ii.
```python
if len(env_values) != 1 or env_values[0] not in (0, 1): raise ValueError(...)
np.full(T, env_values[0])
```

iii. Repetition makes the static variable compatible with the `(4,T)` input layout.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It reads dense `trial number` at every detected trial start.

ii.
```python
raw_trial_numbers = np.rint(dense["trial number"][starts]).astype(int)
```

iii. The agent chose the original zero-based number so exclusions do not renumber experimental chronology.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. It rounds/casts the start value, checks uniqueness, and repeats it through the trial.

ii.
```python
raw_trial = int(raw_trial_numbers[i])
np.full(T, raw_trial)
```

iii. The notes emphasize preserving raw indices, including gaps after bad-lick exclusions.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward/timestamps`, dense timestamps, trial bounds, and dense `reward_zone` activity.

ii.
```python
outcomes = _reward_outcomes(timestamps, starts, teleports,
    np.asarray(behavior["Reward/timestamps"][:]), dense["reward_zone"])
```

iii. The agent follows its interpretation of `get_trial_types`: reward delivery within the trial plus a zone entry defines success.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are computed for all raw trials before filtering; a retained trial receives the preceding raw trial's binary outcome, or 0 for the first.

ii.
```python
np.full(T, outcomes[i - 1] if i > 0 else 0)
```

iii. The notes explicitly preserve chronology when an intervening bad-lick trial is removed.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses dense `position` plus reward-zone identity parsed from the NWB scene/identifier and raw trial number; fixed zone bounds are A 80–130, B 200–250, C 320–370 cm.

ii.
```python
labels = parse_scene_zones(scene)
zone_label = zone_for_trial(labels, raw_trial)
zone_start, zone_end = ZONE_BOUNDS[zone_label]
```

iii. The agent validated the scene/switch schedule against 10,394 zone-entry positions with zero nearest-zone mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. It computes signed point-to-interval distance: negative before the zone, zero anywhere inside, positive after.

ii.
```python
distance = position - np.clip(position, zone_start, zone_end)
```

iii. The agent argues this directly implements “distance to any location in the reward zone” and creates a meaningful exact-zero class.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create seven classes: `<-50`, `[-50,-10)`, `[-10,0)`, `==0`, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
out[distance < -50.0] = 0
out[(distance >= -50.0) & (distance < -10.0)] = 1
...
out[distance > 50.0] = 6
```

iii. Explicit inequalities were chosen to follow the wording at exact boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[start:stop)` frames and shape equality is asserted.

ii.
```python
position = np.asarray(dense["position"][start:stop])
if events.shape[1] != T or output_data.shape != (6, T): raise AssertionError(...)
```

iii. No interpolation is needed because the NWB streams are already synchronized.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from dense behavioral `position`.

ii.
```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
```

iii. The notes identify this as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. There is no smoothing or resampling; the trial slice is passed to the categorical mapping.

ii.
```python
position_classes(position)
```

iii. The synchronized raw values are already suitable for the requested bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five masks implement `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360` cm.

ii.
```python
out[position < 90.0] = 0
...
out[(position >= 270.0) & (position <= 360.0)] = 3
out[position > 360.0] = 4
```

iii. The agent used explicit boundaries to honor the task's strict `>360` wording.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced with the same trial bounds and checked to have the same `T`.

ii.
```python
position = dense["position"][start:stop]
```

iii. Dense stream timestamp equality is validated before conversion.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the dense `lick` series.

ii.
```python
lick = np.asarray(dense["lick"][start:stop])
```

iii. The agent recognizes this as a cumulative/event-count stream rather than an already binary label.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After excluding artifact trials, every positive sample becomes 1 and all others 0.

ii.
```python
(lick > 0).astype(np.int8)
```

iii. This follows the paper utility and requested no/yes representation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The same `[start:stop)` indices are used, with common timestamps and shape checks.

ii.
```python
lick = np.asarray(dense["lick"][start:stop])
```

iii. No additional temporal transform is applied.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It derives chronological zone labels from the NWB identifier/scene and uses the raw trial number to select the pre- or post-switch label.

ii.
```python
labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
zone_label = labels[0] if len(labels) == 1 or raw_trial_number < 30 else labels[1]
```

iii. The agent prefers experiment metadata over noisy zone-entry samples and reports perfect validation against available entries.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Single-zone sessions always use their sole label; switch sessions change at raw trial 30. A/B/C become 0/1/2 and are repeated across frames.

ii.
```python
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
np.full(T, ZONE_TO_CLASS[zone_label], dtype=np.int8)
```

iii. The notes state the default paper switch convention is trial 30 and raw numbering prevents filter-induced shifts.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse `Reward/timestamps`, dense behavior timestamps, trial boundaries, and dense `reward_zone`.

ii.
```python
reward_times = np.asarray(behavior["Reward/timestamps"][:])
outcomes = _reward_outcomes(timestamps, starts, teleports, reward_times, dense["reward_zone"])
```

iii. The agent says this matches the paper's rewarded-trial definition and ignores three rewards outside on-track intervals.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Searchsorted finds reward events in `[start_time, teleport_time)`; the trial is 1 only if such an event exists and some reward-zone flag is positive. The value is repeated across frames.

ii.
```python
lo = np.searchsorted(reward_times, timestamps[start], side="left")
hi = np.searchsorted(reward_times, timestamps[stop], side="left")
outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
```

iii. The notes report 10,342 rewarded paired trials (84.66%) and preserve these outcomes before trial filtering.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Dense streams are cropped to their common neural/behavior length; mismatched timestamps, nonuniform sampling, unpaired bounds, repeated trial numbers, invalid environments, plane/ROI inconsistencies, nonfinite values, negative events, shape errors, and sessions with fewer than two trials raise errors. OASIS nonfinites become zero. Missing m17/m18 plane-1 response data are disclosed and omitted; lick-artifact trials are dropped.

ii.
```python
common_length = min(fluorescence_ds.shape[0], *(behavior[name]["data"].shape[0] ...))
events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
if len(starts) != len(teleports) or not np.all(teleports > starts): raise ValueError(...)
```

iii. The notes frame these as defensive checks and explicitly document the unavailable plane rather than inventing data.

## 13-a. What are the most time-consuming steps of the code?

i. Per-trial HDF5 fluorescence/neuropil reads, dF/F filtering, OASIS deconvolution, and writing the 7.5-GiB pickle dominate. Full conversion took about 222 seconds plus 8 seconds to pickle.

ii.
```python
for i, (start, stop) in enumerate(zip(starts, teleports)):
    fluorescence = np.asarray(fluorescence_ds[start:stop, :])[:, roi_columns].T
    dff, events, processing_trace = compute_dff_and_events(...)
```

iii. The notes benchmarked sample and full runs and identify contiguous I/O and trial-wise OASIS as major costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial metadata loops (`lick_bad`, reward outcomes, and trial assembly) and final per-trial neuron masking could be partly vectorized. The agent already vectorized correlation updates across neurons; variable trial lengths and reference-style trial-wise baselines/OASIS limit useful vectorization.

ii.
```python
for i, (start, stop) in enumerate(zip(starts, teleports)):
    ...
kept_events = [np.asarray(x[neuron_keep], dtype=np.float32) for x in kept_events]
```

iii. The notes say OASIS must remain trial-wise to match the processing, while contiguous row reads and online sufficient statistics avoid worse loops/copies.

## 13-c. What processing does the code repeat multiple times?

i. Each trial repeats HDF5 reads, neuropil correction, baseline filters, smoothing, OASIS, and array construction. Validation later iterates through every trial again; plotting may also retain and render an example processing trace. It does not perform the reference solution's separate survey pass.

ii.
```python
for i, (start, stop) in enumerate(zip(starts, teleports)):
    dff, events, processing_trace = compute_dff_and_events(...)
...
for n, x, y in zip(data["neural"][s], data["input"][s], data["output"][s]):
```

iii. The agent justifies session-at-a-time, trial-wise work as necessary for memory and faithful processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `compute_dff_and_events` creates and returns `corrected`, `baseline`, unsmoothed dF/F, and smoothed dF/F traces for every trial although only events are saved; most are used transiently for correlation and only one optional plot. It also computes dF/F/correlation for bad-lick trials (while deliberately skipping their OASIS), and optional diagnostic plotting is not consumed by the decoder.

ii.
```python
trace = {"corrected": corrected, "baseline": baseline,
         "dff_unsmoothed": dff_unsmoothed, "dff": dff}
return dff, events, trace
```

iii. The notes describe plots and extensive validations as sanity checks; thus they are intentionally diagnostic even though downstream training discards them.
