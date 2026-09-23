# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files under `/app/data/sub-*/` are discovered using `Path.glob("sub-*/*.nwb")`, sorted lexically. Each file is opened with `pynwb.NWBHDF5IO`. Subject ID and session ID are read from the NWB metadata (`nwb.subject.subject_id`, `nwb.session_id`). All 152 files are processed.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")
    if not sample:
        return files
    ...
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    subject = nwb.subject.subject_id
    session_id = str(nwb.session_id)
```

iii. The AI verified that 152 NWB files exist across 11 subject directories (CONVERSION_NOTES Step 2), consistent with the paper's 11 switch-task mice.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file's `nwb.subject.subject_id` field. Unique subject IDs are collected from all discovered files and sorted.

ii.
```python
subjects = sorted({p.parent.name.removeprefix("sub-") for p in files})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
```

iii. Reading subject ID from NWB metadata rather than parsing directory names. The AI verified 11 unique subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session identity comes from the NWB file's `session_id` field.

ii.
```python
for session_i, path in enumerate(files):
    result, diagnostics = process_session(path, ...)
    ...
    subject_idx.append(subject_lookup[result["info"]["subject"]])
```

iii. Each NWB file contains one imaging session; 152 total sessions across 11 subjects.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined from `trial_start` and `teleport` behavior time series. `trial_start` samples with value > 0 mark trial starts; `teleport` samples with value > 0 mark trial ends. The trial interval is `[start, teleport)` (teleport sample excluded). Start/stop counts are verified to match and events are verified to strictly alternate.

ii.
```python
def trial_events(behavior) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
    stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
    if starts.size != stops.size:
        raise ValueError(f"trial starts ({starts.size}) != teleports ({stops.size})")
    if not (np.all(starts < stops) and (starts.size < 2 or np.all(stops[:-1] < starts[1:]))):
        raise ValueError("Trial start/teleport events do not strictly alternate")
    return starts, stops
```

iii. The AI verified 12,216 paired trials across all 152 sessions (CONVERSION_NOTES Step 2). The alternation check ensures proper pairing.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on the paper's lick sensor QC: trials where >30% of frames have cumulative lick count >2 are excluded. This matches the paper's described 81 trials removed. No minimum trial length filter is applied.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])
...
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    if bad_lick[trial_i]:
        continue
```

iii. CONVERSION_NOTES Step 3: "81 (~0.65%), threshold >30% of frames with cumulative count >2" from the Methods. The AI reproduces exactly 81 flagged trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from raw `Fluorescence` (F) and `Neuropil` (Fneu) time series in the NWB ophys processing module. The NWB `Deconvolved` field is explicitly NOT used, as it contains Suite2p's raw deconvolution which does not match the paper's processing.

ii.
```python
f_series = ophys["Fluorescence"].roi_response_series[plane]
fn_series = ophys["Neuropil"].roi_response_series[plane]
...
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
neuropil = np.asarray(fn_series.data[:n_behavior, local_cells], dtype=np.float32).T
```

iii. CONVERSION_NOTES Step 4: "NWB Deconvolved is on raw Suite2p scale... recomputed reference events are ~0-1 and correlate only ~0.40 median across cells with NWB Deconvolved."

## 2-b. How is the `neural` data processed?

i. dF/F is computed per trial: subtract 0.7 * neuropil, add back within-trial mean neuropil, apply maximin baseline (Gaussian smooth sigma=15, 300-sample minimum filter, 300-sample maximum filter), compute (F - baseline) / |baseline|, smooth with sigma=2 Gaussian, then OASIS deconvolution with tau=0.7 at 15.5078125 Hz. The code does NOT implement the `keep_teleports` logic that the paper uses for certain sessions where the laser was not blanked between trials.

ii.
```python
def reference_dff(fluorescence, neuropil, starts, stops):
    out = np.full(fluorescence.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        fneu = neuropil[:, start:stop]
        corrected = fluorescence[:, start:stop] - NEUROPIL_COEF * fneu
        corrected += NEUROPIL_COEF * np.mean(fneu, axis=1, keepdims=True)
        smooth = ndimage.gaussian_filter(corrected, sigma=(0, 15))
        baseline = ndimage.minimum_filter1d(smooth, 300, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
        dff = (corrected - baseline) / np.abs(baseline)
        out[:, start:stop] = ndimage.gaussian_filter1d(dff, 2, axis=-1)
    return out
```
```python
def reference_events(dff, starts, stops):
    events = np.zeros(dff.shape, dtype=np.float32)
    for start, stop in zip(starts, stops):
        events[:, start:stop] = dcnv.oasis(
            dff[:, start:stop], batch_size=2000, tau=OASIS_TAU_S, fs=FRAME_RATE
        )
    return events
```

iii. CONVERSION_NOTES Step 4: "Recompute dF/F and OASIS events from NWB Fluorescence and Neuropil using the reference logic." Processing matches the Methods description of maximin baseline, neuropil subtraction, and OASIS deconvolution, but omits the `keep_teleports` session-specific metadata that controls whether the baseline window extends across teleport periods.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Suite2p manual curation via `iscell[:,0] > 0` from the PlaneSegmentation table, (2) putative interneuron exclusion where dF/F-speed Pearson correlation > 0.5. Cells from multiple planes are pooled after filtering.

ii.
```python
ps = ophys["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
iscell_table = np.asarray(ps["iscell"][:])
iscell = iscell_table[:, 0] > 0
...
speed_corr = correlations_with_speed(dff, speed, valid_mask)
keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)
```

iii. The paper Methods describe manual curation and dF/F-speed correlation > 0.5 exclusion (0.42 +/- 0.85% of cells). The AI reports 402 excluded interneurons (0.290%) across all sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start. Neural data is sliced using the same `[start:stop]` indices as behavior data, so no additional temporal realignment is needed. The trial starts at the `trial_start` event sample.

ii.
```python
neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```

iii. Neural and behavioral data share the same sampling clock and indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging-frame rate of ~64.5 ms (15.5078125 Hz). No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. The AI verified that all sessions have the same effective per-plane rate. For dual-plane sessions, the scanner rate is ~31 Hz but each plane is sampled at ~15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series timestamps.

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
timestamps[start:stop] - timestamps[start],
```

iii. Behavior timestamps are consistent across all behavior time series.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first sample in the trial is subtracted from all timestamps in the trial, giving time in seconds from trial start.

ii.
```python
timestamps[start:stop] - timestamps[start],
```

iii. Standard approach for computing relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices within each trial, so no additional alignment is needed.

ii. Same `[start:stop]` indexing used for both neural and behavioral data.

iii. Verified via timestamp interval checks (all intervals are ~64.5 ms).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float64)
...
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
...
env = int(round(float(env_values[0])))
```

iii. Environment is verified to be constant within each trial (exactly one valid value 0 or 1).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique environment value within the trial is extracted, negative values are excluded, and the result is rounded to an integer (0 or 1). The value is broadcast to all timepoints.

ii.
```python
np.full(T, env),
```

iii. Values are validated to be exactly 0 or 1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the zero-based ordinal of the trial within the session, from the enumeration of paired start/teleport events.

ii.
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    ...
    np.full(T, trial_i),
```

iii. Uses the raw trial ordinal rather than the stored `trial number` variable (which has known anomalies).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond broadcasting the ordinal to all timepoints. Gaps in trial numbering are preserved when trials are excluded by QC.

ii.
```python
np.full(T, trial_i),
```

iii. Simple ordinal assignment.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps. Reward outcomes are precomputed for all trials by checking whether any reward timestamp falls within the trial's time range.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
```

iii. Reward timestamps have their own timing independent of the behavior sampling clock.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's outcome is used. For the first trial, the value is 0. The value is broadcast to all timepoints.

ii.
```python
previous_outcome = int(outcomes[trial_i - 1]) if trial_i > 0 else 0
...
np.full(T, previous_outcome),
```

iii. Previous outcome uses the raw previous trial's outcome regardless of whether that trial was excluded by QC.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and reward zone labels. Reward zone labels (A/B/C) are parsed from the NWB file's `identifier` field (scene name), with a switch at trial 30 for sessions that have a zone switch.

ii.
```python
def parse_scene_zones(scene: str) -> tuple[str, str | None]:
    match = re.search(r"Location([ABC])(?:_to_([ABC]))?$", scene)
    ...
    return match.group(1), match.group(2)
...
zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
```

iii. The paper's reference code `get_reward_zones` similarly parses scene names and applies the switch at trial 30. The AI validates switches against the environment stream.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative before the zone, zero inside the zone (entire 50 cm), positive after the zone. Reward zone coordinates are A=[80,130], B=[200,250], C=[320,370] cm.

ii.
```python
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
def distance_classes(position, zone):
    start, stop = zone
    distance = np.where(position < start, position - start, np.where(position > stop, position - stop, 0.0))
```

iii. Matches the instruction's definition: "distance to any location in the reward zone."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit comparisons:
- 0: < -50 cm
- 1: [-50, -10)
- 2: [-10, 0)
- 3: == 0 (in reward zone)
- 4: (0, 10]
- 5: (10, 50]
- 6: > 50 cm

ii.
```python
classes = np.full(position.shape, -1, dtype=np.int8)
classes[distance < -50] = 0
classes[(distance >= -50) & (distance < -10)] = 1
classes[(distance >= -10) & (distance < 0)] = 2
classes[distance == 0] = 3
classes[(distance > 0) & (distance <= 10)] = 4
classes[(distance > 10) & (distance <= 50)] = 5
classes[distance > 50] = 6
```

iii. The bins match the instructions. Uses explicit comparisons rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `[start:stop]` indexing as neural data; no additional alignment needed.

ii.
```python
pos = position[start:stop]
...
signed_distance, dist_class = distance_classes(pos, REWARD_ZONES[zone_label])
```

iii. Neural and behavioral data share the same time indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float64)
...
pos = position[start:stop]
```

iii. Position is recorded in cm on the 450 cm VR track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
position_classes(pos),
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using explicit comparisons:
- 0: < 90 cm
- 1: [90, 180]
- 2: (180, 270]
- 3: (270, 360]
- 4: > 360 cm

ii.
```python
def position_classes(position):
    classes = np.full(position.shape, -1, dtype=np.int8)
    classes[position < 90] = 0
    classes[(position >= 90) & (position <= 180)] = 1
    classes[(position > 180) & (position <= 270)] = 2
    classes[(position > 270) & (position <= 360)] = 3
    classes[position > 360] = 4
    return classes
```

iii. Five equal-sized 90 cm bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop]` indexing as neural data.

ii. `pos = position[start:stop]`

iii. Neural and behavioral data share the same time indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float64)
```

iii. The lick variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0. Trials with bad lick sensor data (>30% frames with count >2) are excluded entirely.

ii.
```python
(lick[start:stop] > 0).astype(np.int8),
```

iii. The instructions specify binary output (no/yes). The lick QC ensures sensor failures are excluded.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop]` indexing as neural data.

ii. `lick[start:stop]`

iii. Neural and behavioral data share the same time indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB file identifier (scene name), which encodes the initial and optionally switched reward zone labels (A/B/C).

ii.
```python
scene = nwb.identifier.rstrip("/").split("/")[-1]
initial_zone, switched_zone = parse_scene_zones(scene)
```

iii. The reference code `behavior.get_reward_zones` similarly parses scene names to determine zone labels. The AI validates that switches occur at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The initial zone applies to trials 0-29; after trial 30 (if there's a switch), the switched zone applies. Zones are mapped to integers: A=0, B=1, C=2.

ii.
```python
zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
...
np.full(T, ZONE_CODES[zone_label], dtype=np.int8),
```

iii. Switch at trial 30 matches the paper Methods: "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from sparse `Reward` timestamps in the behavior time series.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
```

iii. Reward delivery events have their own timestamps separate from the behavior sampling clock.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check whether any reward timestamp falls within `[trial_start_time, teleport_time)`. Output is 1 if rewarded, 0 if omitted. The value is constant across all timepoints in the trial.

ii.
```python
np.full(T, outcomes[trial_i], dtype=np.int8),
```

iii. Observed reward rate is ~84.66%, matching the paper's ~85% (15% random omissions).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior length mismatch**: Ten dual-plane sessions have one extra neural row; behavior length is used as authoritative and the extra row is ignored.
- **Bad lick sensor**: 81 trials with >30% of frames having lick count >2 are excluded entirely.
- **Strict alternation check**: Trial starts and teleports are verified to strictly alternate, catching any pairing issues.
- **Environment validation**: Environment is verified to be constant within each trial and exactly 0 or 1.
- **Scene parsing**: Multiple scene name formats are handled with regex fallbacks.

ii.
```python
if f_series.data.shape[0] < n_behavior or fn_series.data.shape[0] < n_behavior:
    raise ValueError(f"Neural series in {plane} is shorter than behavior")
...
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
```

iii. CONVERSION_NOTES Step 4 documents: "Ten dual-plane sessions have one more neural row than behavior rows (documented one-frame scan termination artifact)."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** and reading large fluorescence/neuropil arrays
2. **dF/F computation** with per-trial maximin baseline (Gaussian smoothing, min/max filters)
3. **OASIS deconvolution** for each cell and trial
4. **Pickle serialization** of the ~9 GB output file

ii. N/A

iii. Full conversion completed in 350 seconds (5.83 min) including 8.4 s serialization, well under the 15-minute target.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial sequentially for input/output construction. The `reference_dff` function also loops over trials for baseline computation. The per-cell interneuron correlation could be vectorized (and was, using matrix operations in `correlations_with_speed`).

ii.
```python
def correlations_with_speed(dff, speed, valid_mask):
    x = dff[:, valid_mask].astype(np.float64, copy=False)
    y = speed[valid_mask].astype(np.float64, copy=False)
    x -= np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    return (x @ y) / denominator
```

iii. The per-trial baseline loop is inherent to the paper's processing (baselines are computed independently per trial). The speed correlation was vectorized.

## 13-c. What processing does the code repeat multiple times?

i. Unlike the reference solution which has a separate `survey` step that loads all NWB files before conversion, the AI's code loads each NWB file only once during conversion. No major processing is repeated.

ii. N/A

iii. The AI avoids the double-loading pattern by processing each file in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects optional diagnostics (raw fluorescence, neuropil, dF/F for plotting) which are discarded when not plotting. The `validate_session` function performs per-session checks that add some overhead. Otherwise, processing is focused on producing the required output.

ii.
```python
if collect_diagnostics:
    chosen = int(np.flatnonzero(keep)[0])
    diagnostics.update({"raw_f": fluorescence[chosen].copy(), ...})
```

iii. The diagnostic collection is gated by a flag and only active for up to 2 sessions when `--show-processing` is used.
