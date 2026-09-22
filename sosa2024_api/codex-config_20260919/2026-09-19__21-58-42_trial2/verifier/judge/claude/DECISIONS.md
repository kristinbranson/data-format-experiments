# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All `*.nwb` files one level below `/app/data` are globbed (`sub-*/*.nwb`) and sorted numerically by mouse then session day with a regex key, giving 152 files = 152 sessions across 11 subject directories. Each file is opened once with `pynwb.NWBHDF5IO(..., "r", load_namespaces=True)` inside a context manager and every array is read through the PyNWB object model (`nwb.processing["behavior"]["BehavioralTimeSeries"].time_series`, `nwb.processing["ophys"]["Fluorescence"/"Neuropil"/"ImageSegmentation"]`, `nwb.subject.subject_id`, `nwb.identifier`). No `h5py` access is used. `--full` (default) processes every file; `--sample` processes two hand-picked day-8 switch sessions (m3, m12).

ii.
```python
def natural_session_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse mouse/session from {path}")
    return int(match.group(1)), int(match.group(2))

def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_session_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    if not sample:
        return files
    ...
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    behavior = nwb.processing["behavior"]["BehavioralTimeSeries"].time_series
    ...
```

iii. From CONVERSION_NOTES Step 2: "DANDI 001361, version 0.251124.0550; local manifest reports 92,448,350,544 bytes, 152 NWB assets, and 11 mice. Each `sub-<mouse>/sub-<mouse>_ses-<day>_behavior+ophys.nwb` is one session." The manifest count (152 assets, 11 mice) is used as the completeness check, and the notes state "All NWB inspection used `pynwb.NWBHDF5IO(...).read()`; no direct HDF5 parsing was used," which is the instruction's hard constraint. The conversion log confirms all 152 files were processed.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the NWB metadata field `nwb.subject.subject_id` of each converted session (not from the directory name). The unique ids are sorted by their embedded integer (`m3, m4, m7, m11 … m19`) to build `subjects`, and `subject_idx` maps each session to its subject.

ii.
```python
subject_id = str(nwb.subject.subject_id)
...
subjects = sorted(set(subject_ids), key=lambda value: int(re.search(r"\d+", value).group()))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[subject] for subject in subject_ids], dtype=np.int64)
```

iii. Step 2 of CONVERSION_NOTES: 11 subject directories `sub-m3, m4, m7, m11–m15, m17–m19`, which matches the paper's "counterbalanced across mice (n = 11 mice)". Using the in-file `subject_id` rather than the path makes the subject label authoritative to the NWB metadata; the verification log shows the expected 11 subjects with 14 sessions each except m11 with 12.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The mouse number and session (experiment day) are parsed from the filename for ordering and for `session_info`; the day is also recorded as `session_day`, and the scene string from `nwb.identifier` is stored per session. Sessions are emitted in mouse-then-day order; no cross-session neuron alignment (the paper's multi-day ROI registration) is attempted.

ii.
```python
subject_id = str(nwb.subject.subject_id)
_, session_day = natural_session_key(path)
session_id = f"{subject_id}_ses-{session_day:02d}"
...
session_metadata = {
    "session_id": session_id, "subject": subject_id, "session_day": int(session_day),
    "scene": scene, "source_file": str(path), ...
}
```

iii. Step 2: "Mice generally have days 1-14; m11 begins at day 3 and has 12 sessions", consistent with the methods statement that imaging for m11 began on day 3. The session day is needed because the reward-zone schedule and switch days are day-dependent.

## 1-d. Are the data correctly split into trials?

i. A trial is the track lap `[trial_start, teleport)`: starts are the indices where the frame-synchronous `trial_start` flag is positive, stops are the indices where the `teleport` flag is positive. The teleport sample itself is excluded from the trial. The code asserts the starts and stops are paired and ordered, and that every trial has at least two frames.

ii.
```python
starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial bounds in {path.name}")
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
    if stop - start < 2:
        raise ValueError(f"Trial {raw_idx} in {path.name} has fewer than two frames")
    trial_position = position[start:stop]
```

iii. Step 4 discrepancy table: "NWB event flag at `start` is the first track crossing (median position 1.75 cm); `teleport` row is an interpolated/artifact transition, while `teleport-1` is median 448.94 cm … Slice NWB rows `[trial_start, teleport)` and align time zero to the timestamp of the flagged start. This is semantically equivalent to reference bounds after accounting for legacy one-based indices." The reference repo's `preprocessing.dff`/`glmUtils.get_timeseries_data` use `start-1:stop-1` on one-based legacy indices; the AI checked position endpoints directly instead of copying the offset ("every audited trial begins below 15 cm, ends above 425 cm … and excludes the teleport row").

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level exclusion is the paper's lick-sensor-failure rule: a trial is dropped if more than 30% of its samples have a cumulative lick count > 2. 81 of 12,216 raw trials are removed (12,135 retained). No minimum-duration, speed, engagement or reward-outcome filter is applied; trials are dropped in their entirety (neural, input and output) rather than just NaN-ing the lick channel, because the requested lick output is categorical and cannot carry NaN. The code also asserts each session retains ≥ 2 trials.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
corrupt_lick = np.asarray([
    np.mean(lick[start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, stops)
])
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    if corrupt_lick[raw_idx]:
        continue
...
if len(neural_trials) < 2:
    raise ValueError(f"Only {len(neural_trials)} retained trials in {path.name}")
```

iii. Methods (quoted in Step 3/4): "~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice. These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". Step 4: "Exactly 81 trials at the paper's >30% threshold (0.663%) … Exact match. Since categorical lick labels cannot be NaN, exclude these 81 complete trials from the decoder dataset." Step 3 also records the decision not to impose the paper's other analysis-specific filters (speed >2 cm/s, place-cell significance) because speed class 0 and the reward-outcome output are required.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `neural` is recomputed from the raw Suite2p traces `processing/ophys/Fluorescence/<plane>` (F) and `processing/ophys/Neuropil/<plane>` (Fneu), restricted to ROIs curated by `ImageSegmentation/PlaneSegmentation.iscell[:,0] > 0`. The NWB `Deconvolved` series is deliberately **not** used.

ii.
```python
ophys = nwb.processing["ophys"]
segmentation = ophys["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
fluorescence_series = ophys["Fluorescence"].roi_response_series
neuropil_series = ophys["Neuropil"].roi_response_series
...
roi_ids = np.asarray(f_series.rois.data[:], dtype=np.int64)
local_keep = np.flatnonzero(iscell[roi_ids])
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
neuropil = np.asarray(n_series.data[:len(speed), local_keep], dtype=np.float32)
```

iii. Step 4 discrepancy table: "NWB `Deconvolved` is finite/nonzero during ITIs, is 1000s-fold larger than reference-recomputed events, and correlates only moderately with recomputed events (median r=0.483 across 32 sample neurons) … Recompute dF/F and OASIS events from NWB `Fluorescence` and `Neuropil`; do not use NWB `Deconvolved` as final neural input. This is the largest processing correction found." The module docstring states the same: "Neural preprocessing reproduces the paper's trial-wise neuropil correction, maximin dF/F, smoothing, and OASIS event extraction rather than using the distinct Suite2p `Deconvolved` NWB export."

## 2-b. How is the `neural` data processed?

i. Per plane and per trial, exactly the paper's `preprocessing.dff` pipeline: subtract 0.7 × neuropil, add back 0.7 × the trial-mean neuropil, take a maximin baseline (Gaussian σ = 15 frames along time, then a 300-frame minimum filter followed by a 300-frame maximum filter ≈ the Methods' 20 s window), form `(F − baseline)/|baseline|`, smooth with a 2-frame Gaussian, then deconvolve each trial with Suite2p's OASIS at `tau = 0.7` and the per-plane rate 15.5078125 Hz. Baseline windows are always restricted to the lap (the teleport/ITI is never included). Planes are pooled by concatenating neurons. Output is float32, neurons × time, unnormalized event amplitudes.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
    f_trial = fluorescence[start:stop]
    n_trial = neuropil[start:stop]
    corrected = f_trial - NEUROPIL_COEF * n_trial
    corrected += NEUROPIL_COEF * np.mean(n_trial, axis=0, keepdims=True)

    baseline = gaussian_filter1d(corrected, 15, axis=0)
    baseline = minimum_filter1d(baseline, 300, axis=0)
    baseline = maximum_filter1d(baseline, 300, axis=0)
    denom = np.abs(baseline)
    ...
    trial_dff = gaussian_filter1d((corrected - baseline) / denom, 2, axis=0)
    dff[start:stop] = trial_dff.astype(np.float32, copy=False)

events = np.full(dff.shape, np.nan, dtype=np.float32)
for start, stop in zip(starts, stops):
    # Reference call: dcnv.oasis(cells x time, batch_size=2000, tau, fs).
    events[start:stop] = dcnv.oasis(dff[start:stop].T, 2000, OASIS_TAU_S, EFFECTIVE_FS_HZ).T
...
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. Step 10 check 6: "Conversion matches `preprocessing.dff`: 0.7 neuropil subtraction plus 0.7 trial mean, Gaussian sigma 15, 300-frame minimum then maximum baseline, division by absolute baseline, Gaussian sigma 2, and Suite2p OASIS with tau 0.7 at 15.5078125 Hz." Step 5 key decision 7: "Preserve the reference OASIS event amplitudes as float32 without z-scoring or per-cell normalization; the supplied decoder learns session projections, and altering amplitudes would depart from paper processing." The Methods state dF/F is computed "within each trial independently using a maximin procedure with a 20 s sliding window", which is what the always-restricted-to-lap baseline implements.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's: (1) keep only manually curated ROIs (`iscell[:,0] > 0`); (2) drop putative interneurons, defined as cells whose trial-restricted dF/F correlates with running speed at Pearson r > 0.5. Non-finite correlations are also dropped. Place-cell significance is deliberately not applied. Result: 138,678 manual cells → 402 speed-correlated exclusions (0.29%) → 138,276 retained.

ii.
```python
iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
...
local_keep = np.flatnonzero(iscell[roi_ids])
...
correlations = speed_correlations(dff, speed, starts, stops)
keep = np.isfinite(correlations) & (correlations <= 0.5)
interneuron_count += int(np.sum(~keep))
plane_events.append(events[:, keep])
```

iii. Step 3 curation rules: "Manual Suite2p curation removed multi-soma/dendritic ROIs …"; "Exclude additional putative interneurons when dF/F-speed Pearson r >0.5 (mean exclusion 0.42%)"; "Place-cell significance is an analysis-specific filter … and should not be imposed on the general neural decoder, which should retain all curated pyramidal neurons." Step 9 consistency table compares the release (155–2,341 cells/session) to the paper's "155–2172 putative pyramidal neurons per session" and attributes the maximum difference to the DANDI release version rather than imposing an arbitrary cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the trial start, which requires no extra work: neural frames and behavior samples are one-to-one rows of the same synchronized stream, so slicing `[trial_start, teleport)` from both streams with the same indices aligns them. `off_start = 0.0` and `off_end = None` (variable lap duration); `temporal_alignment_event` is the `trial_start` flag. The only alignment safeguard is on stream length: ophys must have at least as many rows as behavior, and at most one extra trailing row.

ii.
```python
if n_ophys_frames < len(speed):
    raise ValueError(f"Ophys ends before behavior for {key}: {n_ophys_frames} < {len(speed)}")
extra_tail = n_ophys_frames - len(speed)
if extra_tail > 1:
    raise ValueError(f"Unexpected ophys/behavior mismatch for {key}: +{extra_tail} frames")
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
...
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError("Neural/input/output time axes differ")
```
```python
"temporal_alignment_event": "trial_start flag: entry into the 0-cm start of the virtual track",
"off_start": 0.0,
"off_end": None,
```

iii. Step 1/Step 2 notes: "Imaging/behavior are already synchronized at one row per imaging frame"; "Behavior timestamp length, deconvolved row count, and fluorescence row count agree within each inspected session." Step 9 documents the only exception found: ten dual-plane sessions (m17 ses 4, 6; m18 ses 1, 5, 7, 10–14) carry exactly one extra terminal ophys row after the last teleport, "outside every retained trial", so only that row is discarded.

## 2-e. How is the `neural` data temporally binned/resampled?

i. No rebinning or resampling. One converted time bin = one synchronized imaging/behavior sample. The bin size is the per-plane sampling interval, 1000/15.5078125 = 64.483627 ms, stored in `metadata['time_bin_size']`. For the dual-plane m17/m18 sessions the stored series `rate` is 31.015625 Hz (scanner rate), but each plane contributes one row per behavior sample, so the per-plane rate is the same 15.5078125 Hz; the constant is used both as the bin size and as the OASIS sampling rate.

ii.
```python
EFFECTIVE_FS_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / EFFECTIVE_FS_HZ
...
events[start:stop] = dcnv.oasis(dff[start:stop].T, 2000, OASIS_TAU_S, EFFECTIVE_FS_HZ).T
...
"time_bin_size": float(TIME_BIN_MS),
...
median_dt = float(np.median(np.diff(timestamps)))   # recorded per session as behavior_median_dt_s
```

iii. Step 4: "m17/m18 ROI series metadata says 31.015625 Hz, but each plane independently has one row per behavior timestamp and those timestamps are 0.0644836 s apart … Trust explicit behavior timestamps and aligned row counts … No temporal downsampling is needed." Step 5 key decision 1: "Keep one row per synchronized behavior/imaging sample (median 64.483627 ms) … This meets the common-bin requirement without interpolation artifacts."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the explicit behavior timestamps, read off the `position` time series (all frame-synchronous behavior series share the same timestamps array).

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Step 5 mapping table: "common behavior timestamps → `input[0]`; Timestamp minus timestamp at trial-start flag, seconds, float32 time series … Starts exactly at 0; preserves slight timestamp jitter." Step 10 check 7: "Time uses explicit timestamps rather than assuming perfectly uniform samples."

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp of the trial's first sample, cast to float32. Nothing else; the sample-to-sample jitter of the VR/imaging clock is preserved. The resulting range over the dataset is [0, 216.5] s.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
inputs = np.vstack([
    time_from_start,
    np.full(stop - start, env_values[0], dtype=np.float32),
    np.full(stop - start, source_trial_number, dtype=np.float32),
    np.full(stop - start, previous_outcome, dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. Straightforward implementation of "time from start of trial in seconds" with the trial start being the alignment event; the notes emphasise it "starts exactly at 0" and that this was verified in the processing plots and sanity checks.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with the same `[start, stop)` indices as the neural data, so no interpolation or shift is needed. Consistency is enforced by (a) requiring every frame-synchronous behavior series to have the same length as the timestamps, (b) the ophys length check described in 2-d, and (c) an assertion that neural, input and output time axes are equal for every trial.

ii.
```python
if not all(len(x) == len(timestamps) for x in (position, speed, lick, environment, trial_number)):
    raise ValueError(f"Behavior series length mismatch in {path.name}")
...
if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError("Neural/input/output time axes differ")
```

iii. Step 2: behavior and ophys rows are one-to-one at the imaging frame rate, so shared indexing is the alignment. Step 12 item 2: "Neural events and outputs share exactly the same trial axis … No one-bin offset or teleport artifact is visible."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The frame-synchronous `environment` behavior time series (0 = ENV1, 1 = ENV2; −1 marks samples outside valid acquisition).

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float32)
...
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
```

iii. Step 2: "Frame environment values are 0 (ENV1) and 1 (ENV2), with -1 only outside synchronized/valid acquisition." Step 5 maps this to `input[1]` via `behavior.get_trial_types` (`morph`), noting "Environment never changes within a trial."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique valid (≥ 0) environment value within the trial is extracted, validated to be a single value in {0, 1}, and broadcast as a constant across all timepoints of the trial (per-trial variable stored as a constant time series).

ii.
```python
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(f"Trial {raw_idx} has invalid environment values {env_values}")
...
np.full(stop - start, env_values[0], dtype=np.float32),
```

iii. Step 5 mapping: "Unique valid per-trial value, 0=ENV1, 1=ENV2, repeated across time … Validate exactly one environment per trial." Key decision 5: per-trial values are repeated over T "because time-varying and per-trial variables coexist in a single rectangular trial array."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The NWB `trial number` behavior time series, sampled at the trial's start index. The code additionally asserts that this stored value equals the chronological index of the trial-start flag, so the input is the zero-based raw trial index within the session (0 … 99).

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

iii. Step 5 mapping: "`trial number` → `input[2]`: Native zero-based continuous trial index, repeated across time, float32 … Validate integer and equals chronological trial index." Step 10 edge-case audit: "source trial numbers equal raw indices". Using the raw index (rather than a counter over retained trials) keeps the numbering chronologically faithful when a corrupt-lick trial is removed.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond reading the value at the trial-start sample and broadcasting it as a constant float32 time series over the trial.

ii.
```python
np.full(stop - start, source_trial_number, dtype=np.float32),
```

iii. As above: the number is already the zero-based trial index in the NWB stream; the assertion against `raw_idx` is the only processing, and it guarantees the input equals the trial's chronological position in the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The sparse `Reward` behavior time series — specifically its event **timestamps** (data values are all 0.004 mL). Per-trial outcomes are computed once for all raw trials by counting reward events falling in `[timestamps[start], timestamps[stop])`, and the previous trial's entry is used as the input.

ii.
```python
def reward_outcomes(reward_times, timestamps, starts, stops):
    """Map sparse delivery events into binary track-trial outcomes."""
    reward_times = np.sort(np.asarray(reward_times, dtype=np.float64))
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)
...
outcomes = reward_outcomes(np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops)
```

iii. Step 2: "It also has sparse `Reward`: 0.004-mL delivery values with event timestamps." Step 4: "Map a reward timestamp in `[start,teleport)` to outcome 1; otherwise 0. Keep omission trials", giving 84.659% rewarded, matching the paper's ~15% omission rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For raw trial *i*, the input is `outcomes[i−1]`; for the first trial of a session it is 0. The value is broadcast across the trial's timepoints. Crucially the *raw* predecessor is used, so a trial that follows an excluded corrupt-lick trial still reports that excluded trial's true outcome.

ii.
```python
previous_outcome = outcomes[raw_idx - 1] if raw_idx > 0 else 0
inputs = np.vstack([
    time_from_start,
    ...,
    np.full(stop - start, previous_outcome, dtype=np.float32),
])
```

iii. Step 5 mapping: "Previous raw chronological trial outcome; omitted=0, rewarded=1; repeat across time. First trial=0 because no previous observation … If a corrupt-lick trial is excluded, the following trial still references that actual raw preceding trial." Step 10 check 7 reports an explicit test of this edge case: "m4 session 14 converted trial 12/raw trial 13 explicitly passed this edge case."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavior time series plus the active reward-zone interval for that trial. The zone identity is **not** inferred from the `reward_zone` data stream; it is taken from the session's scene string in `nwb.identifier` (e.g. `Env1_LocationB`, `Env1_B_to_Env2_A`) with the paper's fixed switch point of 30 trials, and zone coordinates A = [80, 130], B = [200, 250], C = [320, 370] cm.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def parse_zone_schedule(scene: str, n_trials: int) -> np.ndarray:
    """Return A/B/C label per raw trial from the scene identifier."""
    if "_to_" in scene:
        left, right = scene.split("_to_", 1)
        left_match = re.search(r"([ABC])$", left)
        right_match = re.search(r"([ABC])$", right)
        ...
        labels = [left_match.group(1)] * min(30, n_trials)
        labels.extend([right_match.group(1)] * max(0, n_trials - 30))
        return np.asarray(labels)
    match = re.search(r"Location([ABC])$", scene)
    ...
    return np.full(n_trials, match.group(1))
...
scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
```

iii. Step 4: "Code maps scene A/B/C to X/Y/Z coordinates [80,130], [200,250], [320,370], with switch at trial 30 … Parse `NWBFile.identifier`; use first zone for trials 0-29 and second thereafter." This mirrors the reference repo function `behavior.get_reward_zones(sess, rz_dict, change_trial=30)` and the Methods statement "Each switch occurred after 30 trials." Step 10 check 8: "Zones and the trial-30 switch follow `behavior.get_reward_zones`."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest point of the closed 50-cm zone interval: negative before the zone (`position − zone_start`), exactly 0 anywhere inside the zone, positive after (`position − zone_stop`). Computed per timepoint with a vectorized `np.where`, then discretized (7-c).

ii.
```python
zone_label = str(zone_labels_raw[raw_idx])
zone_start, zone_stop = ZONE_BOUNDS[zone_label]
signed_distance = np.where(
    trial_position < zone_start,
    trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
```

iii. Step 3: "Reward-relative position in the paper is circular and anchored to reward-zone start. The requested output instead explicitly asks for signed 'distance to any location in the reward zone', so Step 5 will use signed distance to the nearest point of the closed zone: negative before the zone, zero inside, positive after." Step 5 adds: "'Any location' is interpreted as distance to the closed 50-cm interval, hence an exact 0 class throughout the zone."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit comparisons: 0: d < −50; 1: −50 ≤ d < −10; 2: −10 ≤ d < 0; 3: d = 0 (inside the zone); 4: 0 < d ≤ 10; 5: 10 < d ≤ 50; 6: d > 50. Boundary behaviour is unit-tested at start-up against hand-written expected classes.

ii.
```python
def distance_classes(distance: np.ndarray) -> np.ndarray:
    """Discretize signed distance to the nearest point in the reward zone."""
    out = np.full(distance.shape, 6, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    return out

def check_discretization_boundaries() -> None:
    d = np.asarray([-51, -50, -10, -0.1, 0, 0.1, 10, 10.1, 50, 50.1])
    assert np.array_equal(distance_classes(d), [0, 1, 2, 2, 3, 4, 4, 5, 5, 6])
```

iii. Step 5 mapping: "Classes: d<-50:0; [-50,-10):1; [-10,0):2; 0:3; (0,10]:4; (10,50]:5; >50:6" — a direct reading of the Decoder Task bin list, with class 3 reserved for being inside the zone. Step 10 check 8: "Position/speed/distance bins implement the requested inequalities; exact equality-side tests passed."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[start, stop)` index range as the neural events, so it is aligned by construction; the per-trial assertion on equal time axes covers it.

ii.
```python
trial_position = position[start:stop]
...
outputs = np.vstack([distance_classes(signed_distance), position_classes(trial_position), ...])
...
if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError("Neural/input/output time axes differ")
```

iii. Behavior and imaging are one row per synchronized frame (Step 2), so identical indexing is the alignment; the `--show-processing` plots overlay position/zone/time to confirm no offset, and Step 12 reports "position/distance increase coherently … No one-bin offset or teleport artifact is visible."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the 450-cm virtual corridor), sliced to the trial.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
...
trial_position = position[start:stop]
```

iii. Step 2: "Track positions nominally span 0-450 cm; invalid/pre-acquisition is -500 cm and teleport intervals are near -50 cm" — excluded because trials stop at the teleport flag.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the trial slice and discretization; raw centimetres are used directly.

ii.
```python
outputs = np.vstack([
    distance_classes(signed_distance),
    position_classes(trial_position),
    ...
])
```

iii. Step 5 mapping: "Tiny boundary overshoots remain in endpoint classes; trial slicing excludes teleport artifacts."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90-cm classes spanning the 450-cm track, with open end bins: 0: p < 90; 1: 90 ≤ p < 180; 2: 180 ≤ p < 270; 3: 270 ≤ p ≤ 360; 4: p > 360. Boundaries are unit-tested.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    out = np.full(position.shape, 4, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    return out
...
p = np.asarray([89.9, 90, 179.9, 180, 269.9, 270, 360, 360.1])
assert np.array_equal(position_classes(p), [0, 1, 1, 2, 2, 3, 3, 4])
```

iii. Direct implementation of the Decoder Task specification ("5 equal-sized bins spanning the 450 cm track"), with open first/last bins so the few samples marginally outside 0–450 cm fall in the end classes. Resulting distribution [0.212, 0.177, 0.231, 0.226, 0.154] is recorded in Step 9.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start, stop)` indices as the neural data; no extra alignment.

ii.
```python
trial_position = position[start:stop]
```

iii. Same justification as 7-d: one behavior row per imaging frame, verified by the per-trial shape assertion, the sanity-check script, and the processing plots.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series, which holds the cumulative lick count within each imaging frame.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
...
trial_lick = lick[start:stop]
```

iii. Step 2: "Lick and reward-zone signals are cumulative within an imaging frame (often >1), so binary events use `>0` per the repository documentation."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized as `lick > 0` → int8 {0, 1}. Trials whose sensor was stuck (>30% of samples with count > 2) are removed entirely beforehand (see 1-e), so no NaN/unknown lick labels enter the dataset.

ii.
```python
(trial_lick > 0).astype(np.int8),
```
```python
corrupt_lick = np.asarray([
    np.mean(lick[start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, stops)
])
```

iii. Step 3: "Licks are converted from cumulative counts to binary per-frame events. Sensor-error trials are NaN/excluded from licking analysis when >30% of samples have counts >2." Step 10 check 8: "Lick is reference-style `>0` binary." Final distribution [0.777, 0.223].

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start, stop)` indices as neural; no shift.

ii.
```python
trial_lick = lick[start:stop]
```

iii. Same as 7-d/8-d; Step 12 notes "licks occur as frame-level events" in the inspected sample-trial plots with no visible offset.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session scene string in `nwb.identifier` (with the trial-30 switch rule), not from the `reward_zone` data stream — see 7-a. Labels are mapped A→0, B→1, C→2.

ii.
```python
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
...
scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
...
np.full(stop - start, ZONE_TO_CLASS[zone_label], dtype=np.int8),
```

iii. Step 4: "Identifiers encode all schedules; parsed totals A=4,186, B=4,010, C=4,020 trials … Same coordinates; each switch after 30 trials." This reproduces the reference repo's `behavior.get_reward_zones` mapping (scene → zone X/Y/Z = A/B/C) and the Methods' 30-trial switch.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Parse the scene, assign the first label to raw trials 0–29 and the second label to trials ≥ 30 in `_to_` (switch) sessions, or a single label for `Location<X>` sessions; map to 0/1/2 and repeat the constant over the trial's timepoints. Zone identity is then also used to build the distance output (7-b).

ii.
```python
if "_to_" in scene:
    ...
    labels = [left_match.group(1)] * min(30, n_trials)
    labels.extend([right_match.group(1)] * max(0, n_trials - 30))
    return np.asarray(labels)
return np.full(n_trials, match.group(1))
```

iii. As 10-a. The retained-trial class balance is A/B/C = 4,172/3,974/3,989 trials (timepoint fractions [0.332, 0.336, 0.333]), which Step 9 uses as the consistency check on the parsed schedule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` time series event timestamps (same source as the previous-trial-outcome input, 6-a).

ii.
```python
outcomes = reward_outcomes(
    np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops
)
```

iii. Step 2/Step 4: `Reward` is a sparse series of 0.004-mL deliveries with their own timestamps, not a frame-synchronous channel, so its timestamps must be mapped onto trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward times are sorted, and for each raw trial the number of reward events in `[timestamps[start], timestamps[stop])` is obtained with two `searchsorted` calls; the trial is rewarded (1) if that count is positive, otherwise omitted (0). The per-trial value is repeated across the trial's timepoints. Omission trials are kept in the dataset.

ii.
```python
def reward_outcomes(reward_times, timestamps, starts, stops):
    reward_times = np.sort(np.asarray(reward_times, dtype=np.float64))
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)
...
np.full(stop - start, outcomes[raw_idx], dtype=np.int8),
```

iii. Step 4: "Sparse NWB reward timestamps give 10,342/12,216 rewarded = 84.659%; three reward events lie outside track trials and are ignored … Exact expected rate" versus the paper's "reward was randomly omitted on ~15% of trials". Step 5: "Retains omission trials" because reward outcome is itself a required output and previous outcome a required input.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script's stance is fail-loud except for one precisely characterized defect:
- **Extra terminal ophys frame**: ten dual-plane sessions have exactly one ophys row more than the behavior stream, after the final teleport; only that single row is truncated. Any other length mismatch, or ophys shorter than behavior, raises.
- **Corrupt lick trials**: 81 trials dropped entirely (1-e).
- **Invalid `environment` samples (−1)**: dropped from the per-trial unique check; the trial must have exactly one valid environment value in {0, 1}, else raise.
- **Structural checks that raise**: unequal trial-start/teleport counts or non-increasing bounds; behavior series with mismatched lengths; F/Fneu shape or ROI-mapping mismatch; zero fluorescence baseline; non-finite neural/input values; unequal neural/input/output time axes; inconsistent neuron counts across trials; fewer than two retained trials in a session.
- Reward events outside any trial window are simply not counted.

ii.
```python
extra_tail = n_ophys_frames - len(speed)
if extra_tail > 1:
    raise ValueError(f"Unexpected ophys/behavior mismatch for {key}: +{extra_tail} frames")
discarded_tail_frames = max(discarded_tail_frames, extra_tail)
```
```python
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial bounds in {path.name}")
if not all(len(x) == len(timestamps) for x in (position, speed, lick, environment, trial_number)):
    raise ValueError(f"Behavior series length mismatch in {path.name}")
...
if np.any(denom == 0):
    raise ValueError(f"Zero fluorescence baseline in trial {trial_idx}")
...
if not (np.all(np.isfinite(neural)) and np.all(np.isfinite(inputs))):
    raise ValueError(f"Nonfinite converted data in {path.name}, trial {raw_idx}")
if any(trial.shape[0] != retained_neurons for trial in neural_trials):
    raise AssertionError("Retained neuron count differs across trials")
```

iii. Step 9: "The first full run stopped at m17 session 4 because its ophys matrices contained one more terminal frame than the behavior streams. A PyNWB audit found this exact +1 condition in ten dual-plane sessions … In every case both streams began at time zero and the surplus ophys row was after the last teleport, hence outside every retained trial. The loader was tightened to allow and discard only this precisely characterized single terminal row, while raising on any other mismatch." Step 10 lists the full edge-case audit (paired trial bounds, first-trial previous outcome, omission trials retained, plane pooling, minimum 40 retained trials per session).

## 13-a. What are the most time-consuming steps of the code?

i. Per the timing instrumentation (per-session wall clock printed for all 152 sessions; 301.43 s total, of which 8.43 s is the pickle write), the dominant costs are, in order: (1) reading the curated F and Fneu columns out of the NWB/HDF5 files (I/O bound, the raw assets total 92 GB); (2) the per-trial dF/F computation — the σ=15 Gaussian plus 300-sample minimum/maximum filters over neurons × frames; (3) OASIS deconvolution per trial per plane; (4) assembling and writing the 8.86-GiB pickle. Per-session times range ≈1–7 s and scale with curated-neuron × frame count. There is no separate survey pass: each NWB file is opened exactly once.

ii.
```python
started = time.perf_counter()
...
session_metadata["conversion_seconds"] = elapsed
...
print(f"[{session_idx + 1:3d}/{len(files)}] {path.name}: trials ..., neurons ..., {info['conversion_seconds']:.2f} s", flush=True)
...
print(f"Pickle write: {write_seconds:.2f} s; total conversion: {total_seconds:.2f} s", flush=True)
```

iii. Step 6: "The raw dataset is 92.45 GB and recomputing dF/F can create several hundred MB of intermediate arrays in high-cell sessions." Step 7 runtime table extrapolates from the sample using a "curated neuron × track-frame workload" ratio (68.75×) to estimate ~6.3 min for the neural conversion, "below the 15-minute optimization threshold"; the realized full run was 301 s.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Everything expensive is already vectorized across neurons; the remaining Python loops are per-trial and per-plane, which the variable trial lengths make natural:
- the dF/F loop and the separate OASIS loop in `trial_dff_and_events` (one pass per trial each);
- the accumulation loop in `speed_correlations` (already vectorized over neurons; loops only over trials);
- the `corrupt_lick` list comprehension (one `np.mean` per trial over a short slice);
- the main per-trial assembly loop in `process_session`, including the per-trial `np.concatenate` across planes.
The discretizations, neuropil correction, baseline filters and Pearson statistics are all array operations. The dF/F baseline filters and the class discretizations could in principle be applied once per session with a lap mask instead of per trial, and the plane concatenation could be done once per session rather than once per trial; neither was done.

ii.
```python
def speed_correlations(dff, speed, starts, stops):
    """Vectorized Pearson correlation over all valid trial samples."""
    ...
    for start, stop in zip(starts, stops):
        x = dff[start:stop].astype(np.float64, copy=False)
        y = speed[start:stop].astype(np.float64, copy=False)
        n += len(y); sum_x += np.sum(x, axis=0); sum_xx += np.sum(x * x, axis=0)
        sum_xy += np.sum(x * y[:, None], axis=0); sum_y += float(np.sum(y)); sum_yy += float(np.sum(y * y))
    numerator = sum_xy - sum_x * sum_y / n
    denominator = np.sqrt((sum_xx - sum_x * sum_x / n) * (sum_yy - sum_y * sum_y / n))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
```

iii. Step 6: "Code speedups added: … vectorize filtering/baseline operations over neurons; compute speed correlations using vectorized sufficient statistics accumulated by trial." Loops that remain are per-trial because the baseline, smoothing and deconvolution are defined *within* a trial by the paper's method, so they cannot be run across trial boundaries.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly once — there is no survey/second pass. What is repeated is iteration over the trial windows inside a session: the dF/F loop, the OASIS loop, the speed-correlation loop, the corrupt-lick comprehension and the final assembly loop each walk the same `zip(starts, stops)` (and the first three are repeated once per imaging plane). The dF/F array is computed for all curated cells and then used twice (for OASIS and for the speed correlation). `np.concatenate` over planes is repeated per trial rather than once per session. The `--sample` and `--full` runs reprocess overlapping sessions, but that is by design of the required workflow.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(starts, stops)):   # dF/F
...
for start, stop in zip(starts, stops):                            # OASIS
...
for start, stop in zip(starts, stops):                            # speed correlation
...
corrupt_lick = np.asarray([... for start, stop in zip(starts, stops)])
...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):      # assembly
```

iii. Step 6: "Read only manually curated ROI columns; process one plane and one session at a time … delete large plane intermediates after filtered event extraction … Full conversion remains sequential to avoid competing reads from 92-GB NWB assets." The repeated trial loops are cheap index walks over already-loaded arrays; the expensive resource (file I/O) is touched once per session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, but not nothing:
- dF/F and OASIS events are computed for **all** manually curated cells, including the 402 (0.29%) later dropped as putative interneurons. The interneuron test needs only dF/F, so the deconvolution of those cells is wasted work (the ordering is forced only for the dF/F step, not for OASIS).
- `kept_positions`, `kept_speeds` and `kept_zone_labels` (copies of the raw position/speed for every retained trial) are accumulated in every session even when `--show-processing` is off, and are then discarded.
- Per-trial constants (environment, trial number, previous outcome, zone, outcome) are materialized as full-length time series; this is required by the target rectangular format rather than being waste, but it is what makes the pickle 8.9 GB.
- `check_discretization_boundaries()`, `median_dt`, `segmented_rois` and the other bookkeeping fields are computed for documentation only.
- The `dff` array itself is a full session-sized float32 buffer retained only until correlations are computed.

ii.
```python
dff, events, trace = trial_dff_and_events(fluorescence, neuropil, starts, stops, ...)
correlations = speed_correlations(dff, speed, starts, stops)
keep = np.isfinite(correlations) & (correlations <= 0.5)
plane_events.append(events[:, keep])       # events for ~0.3% of cells were computed and dropped
...
kept_positions.append(trial_position.copy())
kept_speeds.append(trial_speed.copy())
kept_zone_labels.append(zone_label)
```

iii. The notes do not flag any of these as waste; Step 6 instead documents the opposite (memory-oriented) choices: "Read only manually curated ROI columns … delete large plane intermediates after filtered event extraction; retain float32 neural/input and int8 outputs; use sparse reward timestamp search rather than per-frame event expansion." The residual waste is small (0.29% of deconvolutions, a few MB of behavior copies per session) relative to the 301-s total runtime.
