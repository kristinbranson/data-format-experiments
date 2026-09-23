# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files using a glob pattern `sub-*/\*_behavior+ophys.nwb` under the data root directory, sorted by subject and session number. Each file is opened with `h5py` (not `pynwb`) and all dense behavior time series and neural data are read per-session.

ii.
```python
def discover_files(sample: bool) -> list[str]:
    files = sorted(
        glob.glob(str(DATA_ROOT / "sub-*" / "*_behavior+ophys.nwb")),
        key=_session_sort_key,
    )
    ...
    return files[:2] if sample else files
```
```python
with h5py.File(path, "r") as nwb:
    subject = _decode(nwb["general/subject/subject_id"])
    ...
    behavior = nwb["processing/behavior/BehavioralTimeSeries"]
    neural_group = nwb["processing/ophys"]
```

iii. The AI systematically finds all NWB files and validates their contents. The glob pattern matches the expected naming convention and the sort key ensures deterministic ordering.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject_id` field stored inside each NWB file. Unique subjects are sorted numerically and indexed.

ii.
```python
subject = _decode(nwb["general/subject/subject_id"])
...
subjects = sorted({x["subject"] for x in converted}, key=lambda s: int(s[1:]))
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
```

iii. Subject identity comes from the authoritative NWB metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session day is read from the NWB `session_id` field.

ii.
```python
day = int(_decode(nwb["general/session_id"]))
session_id = f"{subject}_ses-{day:02d}"
```

iii. One file per session is consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified from the `trial_start` and `teleport` dense behavior time series. Trial start indices are frames where `trial_start > 0`, and teleport indices are frames where `teleport > 0`. Trials span `[start, teleport)`, excluding the teleport/ITI.

ii.
```python
starts = np.flatnonzero(dense["trial_start"] > 0)
teleports = np.flatnonzero(dense["teleport"] > 0)
if len(starts) != len(teleports) or not np.all(teleports > starts):
    raise ValueError(f"Unpaired or reversed trial bounds in {session_id}")
```

iii. The AI validates that starts and teleports are paired and ordered. This approach differs from the reference which uses a rising-edge detector for teleport; both produce the same result because the teleport signal is a single-frame pulse in all sessions (verified by the assertion passing on all 152 sessions).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using the paper's published lick-sensor criterion: a trial is excluded if more than 30% of its frames have a cumulative lick count > 2. This removes 81 trials across 152 sessions, matching the paper's reported count.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
lick_bad = np.array([
    np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, teleports)
])
...
if lick_bad[i]:
    continue
```

iii. The AI followed the paper's published Methods which state ">30% of frames with cumulative lick count >2" and report 81 excluded trials. The code's 30% threshold reproduces this count exactly over the 12,216 available paired imaging trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence/plane0` and `Neuropil/plane0` time series in the NWB ophys processing group. The NWB `Deconvolved` series is explicitly NOT used.

ii.
```python
fluorescence_ds = neural_group["Fluorescence/plane0/data"]
neuropil_ds = neural_group["Neuropil/plane0/data"]
```

iii. The AI correctly identified that the NWB `Deconvolved` array is Suite2p's own deconvolution of raw fluorescence, not the paper's trial-wise dF/F-based OASIS events.

## 2-b. How is the `neural` data processed?

i. Per-trial, the AI implements: (1) subtract 0.7 x neuropil from fluorescence, (2) add back 0.7 x trial mean neuropil, (3) Gaussian smooth with sigma=15 samples, (4) 300-sample minimum filter then maximum filter for maximin baseline, (5) dF/F = (corrected - baseline) / |baseline|, (6) Gaussian smooth dF/F with sigma=2 samples, (7) OASIS deconvolution with tau=0.7 at 15.5078125 Hz. Non-finite events are replaced with 0. Each trial is processed independently.

ii.
```python
def compute_dff_and_events(fluorescence, neuropil, compute_events=True):
    corrected = fluorescence - NEUROPIL_COEF * neuropil
    corrected += NEUROPIL_COEF * np.mean(neuropil, axis=1, keepdims=True)
    baseline_seed = gaussian_filter(corrected, sigma=(0.0, BASELINE_SMOOTH_SIGMA), mode="reflect")
    baseline = minimum_filter1d(baseline_seed, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect")
    baseline = maximum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        dff_unsmoothed = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff_unsmoothed, DFF_SMOOTH_SIGMA, axis=1)
    events = None
    if compute_events:
        events = dcnv.oasis(np.asarray(dff, dtype=np.float32), 2000, OASIS_TAU_SECONDS, FRAME_RATE_HZ)
        events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
    ...
```

iii. The AI's processing parameters match the paper's Methods. However, the AI processes each trial independently without implementing the `keep_teleports` logic from the paper's `teleport_metadata.py`. On sessions where imaging continued during the teleport/ITI, the reference code extends the baseline window to include ITI data, while the AI restricts it to within-trial data only. This affects ~60% of sessions. Additionally, the AI uses `gaussian_filter` (regular) rather than `nansmooth` (NaN-aware), but since per-trial data has no NaNs, this is functionally equivalent within the trial window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) manual Suite2p `iscell` curation restricts to labeled cells in plane0, and (2) putative interneurons are excluded based on dF/F-speed Pearson correlation > 0.5 computed across all trials using a numerically stable batch-Welford accumulator.

ii.
```python
segmentation = neural_group["ImageSegmentation/PlaneSegmentation"]
plane_index = np.asarray(segmentation["planeIdx"][:])
iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
plane0_iscell = iscell[plane_index == 0]
roi_columns = np.flatnonzero(plane0_iscell)
...
speed_corr = _finalize_corr(corr_sums)
is_interneuron = np.isfinite(speed_corr) & (speed_corr > INTERNEURON_SPEED_R)
neuron_keep = ~is_interneuron
kept_events = [np.asarray(x[neuron_keep], dtype=np.float32) for x in kept_events]
```

iii. The AI correctly identified both the manual curation and the speed-correlation filter from the paper's Methods (">0.5 Pearson correlation"). The plane0-only restriction for m17/m18 is a source limitation due to missing plane1 response data in the NWBs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since neural and behavioral data share the same time indices (both sampled at the same rate and synchronized), no additional alignment beyond slicing `[start:teleport)` is needed. Time t=0 corresponds to the trial_start frame.

ii.
```python
fluorescence = np.asarray(fluorescence_ds[start:stop, :], dtype=np.float32)[:, roi_columns].T
```

iii. The trial_start flag marks the entry into the 450 cm track, which is the alignment event. Slicing from start to teleport extracts the on-track trial period.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native imaging rate of 15.5078125 Hz (64.4836 ms per bin). No temporal rebinning is applied. The AI uses a single constant FRAME_RATE_HZ for all sessions.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The AI verified that all behavior timestamps have an invariant 64.4836 ms interval across all 152 sessions, including the two-plane m17/m18 sessions where the NWB `rate` attribute is 31 Hz (volume rate, not per-plane rate).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the dense behavior `timestamps` array (specifically the `position` timestamps, verified to be identical for all dense behavior series).

ii.
```python
timestamps = np.asarray(behavior["position/timestamps"][:common_length])
...
input_data = np.vstack((
    timestamps[start:stop] - timestamps[start],
    ...
))
```

iii. The AI validated that all dense behavior time series share identical timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's start timestamp is subtracted to produce time relative to trial onset, starting at exactly 0 seconds.

ii.
```python
timestamps[start:stop] - timestamps[start],
```

iii. Straightforward computation from timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (verified via timestamp assertions), so no additional alignment is needed.

ii.
```python
if not np.allclose(np.diff(timestamps), 1.0 / FRAME_RATE_HZ, atol=1e-9):
    raise ValueError(f"Nonuniform timestamps in {session_id}")
```

iii. Both neural and behavioral data are at the same sampling rate, and the AI verified uniform spacing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the dense `environment` behavior time series.

ii.
```python
env_values = np.unique(dense["environment"][start:stop])
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(...)
...
np.full(T, env_values[0]),
```

iii. The AI validates that environment is constant (0 or 1) within each trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. No transformation beyond extracting the single constant value per trial and repeating it across time.

ii.
```python
np.full(T, env_values[0]),
```

iii. The environment is already binary in the source data, matching ENV1=0 and ENV2=1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` dense behavior time series, read at the trial start frame and rounded to integer.

ii.
```python
raw_trial_numbers = np.rint(dense["trial number"][starts]).astype(int)
...
raw_trial = int(raw_trial_numbers[i])
...
np.full(T, raw_trial),
```

iii. The AI preserves the original experimental trial number from the NWB data rather than using a re-indexed loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The raw trial number is rounded to integer and repeated across all time frames within the trial. No re-indexing or renumbering is applied.

ii.
```python
np.full(T, raw_trial),
```

iii. Preserving the original numbering maintains experimental information (e.g., switch at trial 30 is preserved).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps and the dense `reward_zone` time series. Reward outcomes are computed for all raw trials before any filtering.

ii.
```python
def _reward_outcomes(timestamps, starts, teleports, reward_times, reward_zone):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
```

iii. The AI follows the paper's `get_trial_types` logic which requires both a reward event and zone entry to constitute a rewarded trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous RAW trial's outcome is used (0 = omitted/no predecessor, 1 = rewarded). The first trial gets 0. Outcomes are computed before lick filtering, so even dropped trials contribute their outcome as "previous" for the next trial.

ii.
```python
np.full(T, outcomes[i - 1] if i > 0 else 0),
```

iii. This correctly preserves experimental chronology: the animal experienced the previous trial regardless of our data quality filtering.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the dense `position` behavior time series and the reward zone identity (A/B/C) for each trial. Zone identity is parsed from the NWB `identifier` (scene name) using regex, with the second zone label applied from raw trial 30 onward on switch days.

ii.
```python
def parse_scene_zones(scene: str) -> list[str]:
    labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
    ...
    return labels

def zone_for_trial(labels: list[str], raw_trial_number: int) -> str:
    if len(labels) == 1 or raw_trial_number < 30:
        return labels[0]
    return labels[1]
```

iii. The AI's scene-based zone determination matches the reference code's `behavior.get_reward_zones` approach. Validated against raw reward-zone entry positions with zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed point-to-interval distance: `position - clip(position, zone_start, zone_end)`. This is 0 inside the zone, negative before the zone, and positive after. The continuous distance is then discretized into 7 classes.

ii.
```python
zone_start, zone_end = ZONE_BOUNDS[zone_label]
distance = position - np.clip(position, zone_start, zone_end)
```

iii. The vectorized clip approach computes the same signed distance as the reference's explicit conditional logic. Zone bounds are A=[80,130], B=[200,250], C=[320,370] from the paper.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using explicit inequalities into 7 classes: 0 (< -50), 1 ([-50, -10)), 2 ([-10, 0)), 3 (== 0), 4 ((0, 10]), 5 ((10, 50]), 6 (> 50).

ii.
```python
def distance_classes(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    out[distance > 50.0] = 6
    return out
```

iii. The explicit inequalities correctly match the task specification's bin definitions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices within each trial, so no additional alignment is needed beyond slicing the same `[start:stop)` range.

ii.
```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
```

iii. Same time indexing as neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the dense `position` behavior time series.

ii.
```python
position = np.asarray(dense["position"][start:stop], dtype=np.float32)
```

iii. The position variable directly records position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting per-trial slices and discretizing into 5 bins.

ii.
```python
position_classes(position),
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized using explicit inequalities: 0 (< 90), 1 ([90, 180)), 2 ([180, 270)), 3 ([270, 360]), 4 (> 360). Exactly 360 cm is included in class 3.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    out[position > 360.0] = 4
    return out
```

iii. The AI carefully implemented "270 to 360 cm" as [270, 360] (inclusive) and "> 360 cm" as strictly greater, matching the task specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data, no additional alignment needed.

ii. Same slicing: `dense["position"][start:stop]`

iii. Verified via timestamp assertions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the dense `lick` behavior time series (cumulative lick counts per frame).

ii.
```python
lick = np.asarray(dense["lick"][start:stop])
```

iii. The lick variable records cumulative lick counts at each frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick count maps to 1, otherwise 0.

ii.
```python
(lick > 0).astype(np.int8),
```

iii. The instructions specify binary output (no/yes). Trials with bad lick sensor data are already excluded before this step.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data, no additional alignment needed.

ii. Same slicing as neural data.

iii. Verified via timestamp assertions.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field (which contains the scene name encoding zone labels) and the raw trial number (to determine when the zone switches on switch days).

ii.
```python
scene = _decode(nwb["identifier"]).rstrip("/").split("/")[-1]
labels = parse_scene_zones(scene)
...
zone_label = zone_for_trial(labels, raw_trial)
```

iii. The scene name encodes one or two zone labels (e.g., "LocationA_to_LocationB"). The AI parses these and applies the second label from raw trial 30 onward, matching the paper's switch schedule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene names are parsed via regex to extract A/B/C labels. On switch days (two labels), the second label is used from raw trial index 30 onward. The zone label is mapped to classes 0=A, 1=B, 2=C.

ii.
```python
def parse_scene_zones(scene: str) -> list[str]:
    labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
    ...
    return labels

def zone_for_trial(labels: list[str], raw_trial_number: int) -> str:
    if len(labels) == 1 or raw_trial_number < 30:
        return labels[0]
    return labels[1]
...
np.full(T, ZONE_TO_CLASS[zone_label], dtype=np.int8),
```

iii. This follows the paper's `behavior.get_reward_zones` logic. The AI validated against raw zone-entry positions with zero mismatches.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps and the dense `reward_zone` behavior time series.

ii.
```python
outcomes = _reward_outcomes(timestamps, starts, teleports,
    np.asarray(behavior["Reward/timestamps"][:]), dense["reward_zone"])
```

iii. Uses both reward event timing and zone-entry confirmation.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded (1) if a sparse Reward timestamp falls within `[start_time, teleport_time)` AND the dense reward_zone signal is positive at some point during the trial. Otherwise 0. The outcome is constant across all timepoints in the trial.

ii.
```python
def _reward_outcomes(timestamps, starts, teleports, reward_times, reward_zone):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
...
np.full(T, outcomes[i], dtype=np.int8),
```

iii. The dual condition matches the paper's `get_trial_types` logic. The resulting ~84.66% reward rate matches the paper's ~85%/~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Neural/behavior length mismatch**: Data is cropped to the common minimum length across fluorescence and all dense behavior series (affects 10 two-plane sessions with 1 extra neural sample).
- **Bad lick sensor trials**: Excluded using the paper's >30% criterion (81 trials).
- **Non-finite OASIS events**: Replaced with 0 via `np.nan_to_num`.
- **Sparse reward events outside trials**: 3 events fall outside paired trial windows and are naturally ignored by the start/stop search.
- **Timestamp uniformity**: Verified and raises error if non-uniform.
- **Trial pairing**: Verified starts and teleports are paired and ordered.
- **Session minimum trials**: Requires at least 2 retained trials per session.

ii.
```python
common_length = min(
    fluorescence_ds.shape[0],
    *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
)
...
events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. These defensive checks were identified through systematic data exploration documented in the CONVERSION_NOTES.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **HDF5 I/O**: Reading fluorescence and neuropil arrays per trial from NWB files
2. **OASIS deconvolution**: Computing deconvolved events per trial
3. **Pickle write**: Serializing ~7.5 GiB of converted data

The full conversion took ~222 seconds plus ~8 seconds for pickle write across 152 sessions.

ii. N/A

iii. The AI optimized by reading contiguous trial row blocks rather than full session arrays, skipping OASIS for rejected trials, and using float32 throughout.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially to read HDF5 data, compute dF/F, and construct outputs. Some operations (like `distance_classes`, `position_classes`, `speed_classes`) could theoretically be applied to the full session arrays before splitting into trials, but variable trial lengths make this awkward. The Welford correlation accumulation already uses vectorized batch updates.

ii. N/A

iii. The per-trial structure is natural given variable-length trials and the need for per-trial dF/F baseline computation.

## 13-c. What processing does the code repeat multiple times?

i. The code reads each NWB file exactly once and processes all trials in a single pass. However, for trials that are rejected due to bad lick sensor data, dF/F is still computed (though OASIS is skipped) because the dF/F-speed correlation must include all trials to match the paper's interneuron filter. This means dF/F computation is done for 81 trials that are ultimately discarded.

ii.
```python
dff, events, processing_trace = compute_dff_and_events(
    fluorescence, neuropil, compute_events=not lick_bad[i]
)
```

iii. Computing dF/F for bad-lick trials is necessary to get correct speed-correlation estimates for the interneuron filter.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F for all trials (including bad-lick trials) to support the interneuron filter. The intermediate dF/F values, processing traces, and correlation statistics are not saved in the final output. Additionally, the detailed per-session `info` dictionaries stored in metadata contain diagnostic information not used by the decoder.

ii. N/A

iii. The dF/F computation for rejected trials is necessary for correct interneuron filtering. The metadata overhead is negligible.
