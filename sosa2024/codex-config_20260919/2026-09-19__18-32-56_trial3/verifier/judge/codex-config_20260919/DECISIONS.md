# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB file under `/app/data/sub-*`, naturally sorts files by numeric mouse and session, and processes every file unless `--sample` is requested. It reads NWB/HDF5 datasets directly with `h5py`.

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_file_key)
with h5py.File(path, "r") as nwb:
    behavior = nwb[BEHAVIOR_ROOT]
```

iii. The notes report 152 NWB assets, 11 subjects, one mouse/day per file, and explain that the repository had no native NWB loader, so direct HDF5 loading was used while selecting synchronized behavioral and neural streams.

## 1-b. How are the data split into subjects?

i. Mouse identity is parsed from filenames for sorting/discovery and read from `general/subject/subject_id` for each converted session. Unique IDs are naturally sorted and sessions receive an index into that list.

ii.
```python
subject = decode_text(nwb["general/subject/subject_id"])
subjects = sorted({f"m{natural_file_key(path)[0]}" for path in files},
                  key=lambda value: int(value[1:]))
subject_idx = [subject_lookup[item["subject"]] for item in converted_sessions]
```

iii. The agent verified that the 11 discovered subject IDs match the paper and source metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Files are ordered by numeric subject and session ID, and each produces one entry in the top-level session lists.

ii.
```python
match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
for index, path in enumerate(files):
    converted, info = convert_session(path, show_processing and index < 2)
```

iii. The notes identify the layout as one mouse/day NWB per session and account for all 152 sessions.

## 1-d. How are the data split into trials?

i. Starts are frames where `trial_start > 0`; stops are frames where `teleport > 0`. A trial is sliced `[start:stop)`, including the explicit start frame and excluding the teleport frame.

ii.
```python
starts = np.flatnonzero(group["trial_start/data"][:] > 0)
stops = np.flatnonzero(group["teleport/data"][:] > 0)
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    neural = activity[start:stop].T.copy()
```

iii. The agent states that NWB uses explicit binary event samples and that direct source spot checks supported `[start:teleport)`. It avoided blindly applying the reference saved-session code’s legacy one-based offset.

## 1-e. How are trials filtered based on quality controls?

i. It excludes a trial when more than 30% of its frames have lick count greater than 2. It also requires scanning throughout retained trials and at least two retained trials per session; those latter checks raise errors rather than silently filter.

ii.
```python
bad_trials = np.array(
    [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)]
)
if bad_trials[raw_trial]:
    continue
```

iii. The agent attributes the 30% rule to the paper’s lick-sensor fault QC and reports exactly 81 excluded trials. It deliberately did not use place-cell or speed filters because the requested decoder needs all curated cells and a below-2-cm/s speed class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural arrays are loaded from each plane of `processing/ophys/Deconvolved`, then restricted using the segmentation `iscell` flag and an interneuron mask. Fluorescence and neuropil are read only to build the interneuron mask.

ii.
```python
deconvolved = nwb[f"{OPHYS_ROOT}/Deconvolved"]
chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
fluorescence = nwb[f"{OPHYS_ROOT}/Fluorescence"]
neuropil = nwb[f"{OPHYS_ROOT}/Neuropil"]
```

iii. The agent believed the NWB `Deconvolved` arrays were author-produced OASIS events corresponding to the paper’s processed `events`, so recomputation was unnecessary. This directly conflicts with the human reference’s conclusion that final events must be recomputed from F and Fneu.

## 2-b. How is the `neural` data processed?

i. Stored deconvolved values from all planes are subset to retained cells, concatenated, restored to global ROI order, transposed to neuron-by-time, and cast to float32. No dF/F or OASIS deconvolution is recomputed for the final signal. Separately, paper-like dF/F is recomputed in blocks solely for speed-correlation QC.

ii.
```python
activity = np.concatenate(chunks, axis=1)
order = np.argsort(ids)
return activity[:, order]
neural = activity[start:stop].T.copy()
```

iii. The notes justify reuse of the already processed NWB stream as a faithful format adaptation. They describe the QC-only dF/F parameters as 0.7 neuropil subtraction, Gaussian 15, 300-frame min/max baseline, and Gaussian 2 smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains manually curated Suite2p `iscell` ROIs, then removes cells whose reconstructed within-trial dF/F has Pearson correlation with speed greater than 0.5. Planes are pooled afterward.

ii.
```python
iscell = segmentation["iscell"][:, 0].astype(bool)
keep, speed_correlations = compute_interneuron_mask(nwb, iscell, speed, starts, stops)
keep[ids[corr > INTERNEURON_R_THRESHOLD]] = False
```

iii. The agent identifies both filters as the paper’s cell curation and reports 138,678 manual cells minus 409 putative interneurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the explicit trial-start frame simply by slicing the synchronized frame grid at `[start:stop)`; each trial’s first neural column is time zero.

ii.
```python
relative_time = timestamps[start:stop] - timestamps[start]
neural = activity[start:stop].T.copy()
```

iii. The agent verified neural and behavior on the same native grid and spot-checked endpoints against the NWBs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 15.5078125-Hz grid is preserved, giving 64.483627 ms per sample. There is no temporal rebinning or resampling.

ii.
```python
DT_SECONDS = 1.0 / 15.5078125
"time_bin_size": DT_SECONDS * 1000.0
```

iii. The notes say all synchronized streams use this common frame grid, so resampling would be unnecessary.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps` and the detected trial-start index.

ii.
```python
timestamps = behavior["position/timestamps"][:]
relative_time = timestamps[start:stop] - timestamps[start]
```

iii. The agent verified uniform timestamp differences at the expected native sample interval.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial’s first included frame is subtracted from every timestamp in that trial, then values are cast to float32.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. This makes every retained trial start exactly at zero and preserves native elapsed seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time and neural activity use the identical `[start:stop)` indices and length `T`.

ii.
```python
T = stop - start
neural = activity[start:stop].T.copy()
if neural.shape[1] != T or decoder_input.shape != (4, T):
    raise AssertionError(...)
```

iii. Shape and monotonic-time validations plus raw-data spot checks were used to confirm alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes directly from the synchronized `environment/data` behavioral series.

ii.
```python
environment = behavior["environment/data"][:]
env_values = np.unique(environment[start:stop])
```

iii. The notes identify source coding as ENV1=0 and ENV2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code verifies exactly one valid binary value in the trial and broadcasts it across the trial’s timepoints.

ii.
```python
if env_values.size != 1 or env_values[0] not in (0, 1):
    raise ValueError(...)
np.full(T, env_values[0], dtype=np.float32)
```

iii. Broadcasting satisfies the required time-aligned input matrix while preserving the source’s trial-constant value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes directly from the synchronized `trial number/data` series, not from the Python loop counter.

ii.
```python
trial_number = behavior["trial number/data"][:]
trial_values = np.unique(trial_number[start:stop])
```

iii. The agent describes it as the native zero-based lap/trial number and independently checked its range, 0–99.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code verifies the trial contains one ID and broadcasts that source value over all trial frames.

ii.
```python
if trial_values.size != 1:
    raise ValueError(...)
np.full(T, trial_values[0], dtype=np.float32)
```

iii. The notes favor native source numbering and preserve raw-trial identity even when a faulty trial is removed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward/timestamps`, dense `reward_zone/data`, behavior timestamps, and the previous raw trial’s bounds.

ii.
```python
reward_timestamps = behavior["Reward/timestamps"][:]
outcomes = reward_outcomes(timestamps, reward_timestamps,
                           reward_zone_signal, starts, stops)
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. The agent says this follows the paper helper’s trial-type semantics and deliberately keeps history based on the actual preceding raw trial, even if that trial is later excluded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A current trial is rewarded only if a sparse reward timestamp lies inside its timestamp interval and its dense reward-zone signal is positive somewhere. The outcomes vector is shifted by one; the first trial defaults to 0; the result is broadcast.

ii.
```python
out[i] = int(delivered and zone_evidence)
previous_outcomes = np.r_[0, outcomes[:-1]]
np.full(T, previous_outcomes[raw_trial], dtype=np.float32)
```

iii. The justification is that reward delivery plus zone evidence reproduces `get_trial_types`, with omitted/unavailable encoded 0 and rewarded encoded 1.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses `position/data` and a trial reward-zone label inferred from the NWB identifier/scene protocol. Fixed scenes yield one label; switch scenes use the first label for trials 0–29 and the second thereafter. Fixed A/B/C bounds are 80–130, 200–250, and 320–370 cm.

ii.
```python
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
labels = scene_zone_labels(scene, starts.size)
zone_start, zone_stop = ZONE_BOUNDS[label]
```

iii. The agent cites the paper’s scene protocol and `get_reward_zones`, including the normal switch at raw trial 30, as more direct evidence than inferring zones from noisy reward-zone samples.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the near edge before a zone, zero inside it, and position minus the far edge after it; the continuous result is then classified.

ii.
```python
signed_distance = np.where(
    trial_position < zone_start, trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
```

iii. This implements the requested distance to any location in the zone, with sign indicating before versus after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement seven classes with exact handling at −50, −10, 0, 10, and 50 cm.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. The agent wrote boundary unit tests and states that these inequalities directly follow the task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same trial bounds; the classified distance has exactly `T` elements.

ii.
```python
trial_position = position[start:stop]
neural = activity[start:stop].T.copy()
```

iii. Native synchronization and independent array reconstructions were used to validate the alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from synchronized `position/data`.

ii.
```python
position = behavior["position/data"][:]
trial_position = position[start:stop]
```

iii. The source is documented as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No transformation is applied before categorical thresholding; the per-trial source slice is passed to `discretize_position`.

ii.
```python
discretize_position(trial_position)
```

iii. The requested five equal track bins require only thresholding the synchronized centimeter values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is assigned to five explicit classes: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360` cm.

ii.
```python
out[position < 90] = 0
out[(position >= 90) & (position < 180)] = 1
out[(position >= 180) & (position < 270)] = 2
out[(position >= 270) & (position <= 360)] = 3
out[position > 360] = 4
```

iii. The agent used explicit masks and edge tests to implement the wording in the decoder instructions exactly.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity use identical `[start:stop)` frame slices.

ii.
```python
trial_position = position[start:stop]
neural = activity[start:stop].T.copy()
```

iii. The common synchronized NWB grid and direct spot checks support the alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from synchronized `lick/data`.

ii.
```python
lick = behavior["lick/data"][:]
trial_lick = lick[start:stop]
```

iii. The notes describe this source as the imaging-frame-aligned cumulative lick count.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive source count is mapped to 1 and all other values to 0.

ii.
```python
(trial_lick > 0).astype(np.int8)
```

iii. The agent states that this converts the source count into the requested yes/no lick output; faulty sensor trials are separately excluded.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural streams use the same `[start:stop)` frame slice.

ii.
```python
trial_lick = lick[start:stop]
neural = activity[start:stop].T.copy()
```

iii. The NWB dense behavior streams were already synchronized to imaging frames and direct checks found no shift.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB identifier’s scene/protocol name and raw trial index, not from the dense `reward_zone` and position values.

ii.
```python
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
labels = scene_zone_labels(scene, starts.size)
```

iii. The agent chose the explicit experimental protocol as authoritative and verified balanced A/B/C counts and switch behavior.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. A regex extracts one label for fixed sessions or two for switches; switch trials 0–29 use the first and later trials the second. A/B/C are mapped to 0/1/2 and broadcast over time.

ii.
```python
labels = re.findall(r"(?:Location)?([ABC])", scene)
out[:30] = labels[0]
np.full(T, ZONE_TO_CLASS[label], dtype=np.int8)
```

iii. The notes link this to the paper protocol and say the raw index is preserved across QC exclusions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse `Reward/timestamps`, dense `reward_zone/data`, position timestamps, and trial boundaries.

ii.
```python
reward_timestamps = behavior["Reward/timestamps"][:]
reward_zone_signal = behavior["reward_zone/data"][:]
outcomes = reward_outcomes(timestamps, reward_timestamps,
                           reward_zone_signal, starts, stops)
```

iii. The agent attributes this compound definition to the reference repository’s `get_trial_types` behavior.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. It tests whether any reward timestamp lies from the trial-start timestamp up to but excluding the stop timestamp and whether any reward-zone sample is positive. Their conjunction becomes a binary trial value broadcast across time.

ii.
```python
delivered = np.any((reward_timestamps >= timestamps[start]) &
                   (reward_timestamps < timestamps[stop]))
zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
out[i] = int(delivered and zone_evidence)
```

iii. The notes report an 84.64% rewarded rate, consistent with the paper’s approximately 85%, and independent raw-label checks.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mostly fails explicitly rather than imputing or cropping: it checks dense lengths, uniform timestamps, ordered trial bounds, scanning, constant trial metadata, finite values, ROI references, shapes, classes, and minimum trial count. Ten extra terminal neural rows are harmlessly omitted by limiting neural loading to behavior length. Faulty lick trials are excluded. First-trial history is set to zero.

ii.
```python
if set(dense_lengths.values()) != {n_behavior}:
    raise ValueError(...)
group["data"][:n_behavior, local]
if not np.all(np.isfinite(neural)):
    raise ValueError(...)
```

iii. The notes emphasize explicit validation and independent reconstruction rather than silently repairing source inconsistencies. They document the terminal-row mismatch and all exclusion/accounting identities.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large NWB arrays and computing the blockwise per-cell dF/F–speed QC across all trials/cells dominate conversion; serializing the roughly 8.9-GiB pickle is another measured cost. The full run took about 383 s for conversion and 8 s to write.

ii.
```python
for block_start in range(0, local_cells.size, block_size):
    for start, stop in qc_segments:
        f = f_data[start:stop, local].T.astype(np.float64, copy=False)
```

iii. The notes provide measured full-run timings and a cell-by-trial-frame work estimate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested plane/cell-block/trial loops in `compute_interneuron_mask`, the per-trial reward-outcome loop, faulty-trial list comprehension, and conversion loop are candidates. Some trial work could be performed over whole-session arrays, although variable lengths and per-trial baselines constrain vectorization.

ii.
```python
for plane_name in sorted(fluorescence):
    for block_start in range(0, local_cells.size, block_size):
        for start, stop in qc_segments:
            ...
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
```

iii. The agent intentionally block-processes cells to control memory and reports that measured runtime was already below the budget.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries and raw behavior are revisited for outcome labeling, lick QC, interneuron QC, conversion, and optional plotting. Per-trial slices are also used separately for dF/F QC and final assembly. Unlike the human reference’s survey workflow, the final script does not load every session in a separate survey pass.

ii.
```python
for i, (start, stop) in enumerate(zip(starts, stops)):  # outcomes
qc_segments = [...]                                     # neural QC
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):  # conversion
```

iii. The notes favor bounded-memory session processing. Optional plots repeat some reconstruction only when explicitly requested.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores full per-cell speed correlations although downstream output only needs the keep mask and a summary maximum. Optional plotting reconstructs continuous quantities and normalized traces that are discarded. It also validates `scanning` and numerous invariants that are not saved, though these are useful safeguards rather than scientific outputs.

ii.
```python
correlations = np.full(iscell.size, np.nan, dtype=np.float32)
"max_dff_speed_correlation": float(np.nanmax(speed_correlations))
if show_processing:
    make_processing_plot(...)
```

iii. The agent presents these operations as curation, auditing, and visualization checks; it does not identify substantive required processing as unnecessary.
