# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `sub-*/*.nwb` file under `/app/data`, sorted them, and treated that full list as the dataset. Each session file is opened with `pynwb.NWBHDF5IO`.

ii.
```python
DATA_ROOT = Path("/app/data")

def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")
    if not sample:
        return files

with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. `CONVERSION_NOTES.md` Step 2 says there are 11 subject directories and 152 NWB assets, and that every file was opened with `pynwb` and no `h5py` access was used. Step 5 also states that each NWB file is one session.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory names `sub-<id>` and tracked again from `nwb.subject.subject_id` inside each NWB file.

ii.
```python
subjects = sorted({p.parent.name.removeprefix("sub-") for p in files})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
subject = nwb.subject.subject_id
...
subject_idx.append(subject_lookup[result["info"]["subject"]])
```

iii. In Step 5 the agent explicitly chose “each NWB file is one session” and “subject list is unique sorted mouse IDs,” matching the subject-directory organization it had documented in Step 2.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a separate session.

ii.
```python
for session_i, path in enumerate(files):
    result, diagnostics = process_session(path, args.show_processing and session_i < 2)
...
session_id = str(nwb.session_id)
info = {
    "session_id": f"{subject}_ses-{session_id}",
    "source_file": str(path.relative_to(DATA_ROOT)),
}
```

iii. `CONVERSION_NOTES.md` Step 5 says “Session definition: each NWB file is one session,” and Step 2 says the 152 `*_behavior+ophys.nwb` files correspond to the full recording set.

## 1-d. How are the data split into trials?

i. Trials are defined from the `trial_start` and `teleport` behavioral time series, using `[start, teleport)` as the valid in-trial interval.

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
...
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    pos = position[start:stop]
```

iii. Step 2 says trials are “exactly recoverable from the 1-valued samples in `trial_start` and `teleport`” and Step 4 resolves the trial interval as `[trial_start sample, teleport sample)` because the teleport sample itself can contain interpolation artifacts.

## 1-e. How are trials filtered based on quality controls?

i. The agent excluded whole trials when the lick sensor looked stuck: if more than 30% of frames in a trial had `lick > 2`, that trial was dropped. It did not apply a minimum-trial-length filter.

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

iii. In Step 4 the agent states that the paper’s final licking QC is `>30%` of frames with cumulative lick count `>2`, reproducing exactly 81 bad trials, and that because lick is a required categorical output those trials should be excluded rather than mislabeled as “no lick.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw `ophys/Fluorescence` and `ophys/Neuropil` series, not from the NWB `Deconvolved` export.

ii.
```python
f_series = ophys["Fluorescence"].roi_response_series[plane]
fn_series = ophys["Neuropil"].roi_response_series[plane]
...
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
neuropil = np.asarray(fn_series.data[:n_behavior, local_cells], dtype=np.float32).T
```

iii. Step 4 says the NWB `Deconvolved` array is on an incompatible raw Suite2p scale, while the paper recomputes deconvolved events from `F` and `Fneu`, so the agent explicitly chose to recompute the paper-matched signal.

## 2-b. How is the `neural` data processed?

i. For each plane and trial interval, the agent subtracts `0.7 * neuropil`, adds back the within-trial mean neuropil, computes a per-trial maximin baseline (Gaussian sigma 15, then 300-sample min and max filters), forms dF/F, smooths with sigma 2, and then deconvolves with OASIS to produce events.

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

def reference_events(dff, starts, stops):
    events = np.zeros(dff.shape, dtype=np.float32)
    for start, stop in zip(starts, stops):
        events[:, start:stop] = dcnv.oasis(
            dff[:, start:stop], batch_size=2000, tau=OASIS_TAU_S, fs=FRAME_RATE
        )
    return events
```

iii. Step 5 says the script ports the paper’s relevant `preprocessing.dff` logic: neuropil coefficient `0.7`, maximin baseline, sigma-2 smoothing, OASIS `tau=0.7`, and native imaging cadence. The trajectory also mentions resolving the NWB `Deconvolved` mismatch by recomputing the paper’s custom dF/F/OASIS stream.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It first keeps only Suite2p-curated ROIs with `iscell[:, 0] > 0`, then removes putative interneurons whose dF/F correlates with running speed above `r = 0.5`.

ii.
```python
iscell_table = np.asarray(ps["iscell"][:])
iscell = iscell_table[:, 0] > 0
...
roi_indices = np.asarray(f_series.rois.data[:], dtype=np.int64)
local_cells = np.flatnonzero(iscell[roi_indices])
...
speed_corr = correlations_with_speed(dff, speed, valid_mask)
keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)
```

iii. Step 4 says the ROI table contains accepted and rejected ROIs, so `iscell[:,0]` must be applied first, and then the paper’s additional `r > 0.5` dF/F-speed interneuron exclusion is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial on the same `[start, stop)` indices derived from `trial_start` and `teleport`, so time zero is the trial start sample.

ii.
```python
starts, stops = trial_events(behavior)
...
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```

iii. Step 5 states the alignment event is the start of the trial and records `off_start = 0`, with variable trial durations through the sample before teleport.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data stay at the native imaging-plane cadence, `15.5078125 Hz`, i.e. `1000 / 15.5078125 = 64.4836 ms` per sample. No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
dt = np.diff(timestamps)
if not np.allclose(dt, 1.0 / FRAME_RATE, rtol=0, atol=1e-9):
    raise ValueError(f"Unexpected behavior sample interval in {path.name}")
...
"time_bin_size": TIME_BIN_MS,
```

iii. Step 2 says the authoritative synchronized sample interval is `0.064483627204 s` (15.5078125 Hz) even for dual-plane sessions, and Step 5 explicitly says to retain every native sample with no temporal rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamps attached to the `position` time series.

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
input_trial = np.vstack(
    (
        timestamps[start:stop] - timestamps[start],
        np.full(T, env),
        np.full(T, trial_i),
        np.full(T, previous_outcome),
    )
).astype(np.float32)
```

iii. Step 2 says the behavioral streams share a common frame-aligned clock, and the agent treated behavior timestamps as authoritative for the native 15.5078125-Hz time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial’s first timestamp is subtracted so that each trial begins at 0 seconds.

ii.
```python
timestamps[start:stop] - timestamps[start]
```

iii. Step 5 says “Time-varying; begins at exactly 0,” and the metadata records trial-start alignment with `off_start = 0.0`.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same frame indices and trial slices as the neural data, so behavioral time and neural events are aligned sample-by-sample.

ii.
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    input_trial = np.vstack((timestamps[start:stop] - timestamps[start], ...))
    neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```

iii. Step 2 and Step 4 both say the behavior/imaging streams share the same synchronized frame clock, and that extra trailing neural rows are ignored if they extend beyond the behavior length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float64)
...
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
```

iii. Step 2 identifies `environment` as a frame-aligned behavioral stream, and Step 5 maps it to `input[1]`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the agent requires a single valid environment value, rounds/casts it to `0` or `1`, and repeats that constant value over all frames in the trial.

ii.
```python
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
if env_values.size != 1:
    raise ValueError(f"Trial {trial_i} does not have one valid environment")
env = int(round(float(env_values[0])))
...
np.full(T, env)
```

iii. Step 5 says environment is a per-trial binary variable and explicitly notes not to infer it from session identity alone, because some sessions switch environments after trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial ordinal produced by iterating over the paired `[start, stop)` trial intervals.

ii.
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    ...
    np.full(T, trial_i)
```

iii. Step 5 says to use the zero-based paired-trial ordinal rather than the stored NWB `trial number`, because one session contains a trailing spurious trial-number fragment.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond using the zero-based trial index and repeating it across time within that trial.

ii.
```python
np.full(T, trial_i)
```

iii. The justification in Step 5 is that this is the unambiguous within-session chronological trial index once the valid trial intervals have been defined.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward.timestamps` events after those reward events are converted into a binary per-trial reward outcome array.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
```

iii. Step 4 says outcome is defined by whether a sparse reward timestamp falls inside `[start, teleport)`, which matched the paper’s expected reward rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The current trial gets the previous trial’s binary reward outcome; the first trial is assigned 0. The value is then repeated over all frames of the trial.

ii.
```python
previous_outcome = int(outcomes[trial_i - 1]) if trial_i > 0 else 0
...
np.full(T, previous_outcome)
```

iii. Step 5 says the previous outcome should be based on the true preceding source trial, even if that preceding trial itself would later be excluded for lick-sensor QC.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavioral `position` time series plus the active reward-zone identity for the trial, where zone identity is parsed from the session `scene`/identifier and trial ordinal (switch after trial 30 when applicable).

ii.
```python
scene = nwb.identifier.rstrip("/").split("/")[-1]
initial_zone, switched_zone = parse_scene_zones(scene)
...
zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
signed_distance, dist_class = distance_classes(pos, REWARD_ZONES[zone_label])
```

iii. Step 4 says the agent chose to parse reward zones from the NWB identifier and the paper’s switch rule instead of using the noisy `reward_zone` stream, and Step 5 says this matches the reference `get_reward_zones` logic more directly.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each position sample, the agent computes signed distance to the nearest point in the active reward-zone interval: negative before the zone, zero anywhere inside it, positive after it.

ii.
```python
def distance_classes(position: np.ndarray, zone: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    start, stop = zone
    distance = np.where(position < start, position - start, np.where(position > stop, position - stop, 0.0))
    ...
    return distance, classes
```

iii. Step 3 and Step 5 both state that the requested decoder target is not the paper’s circular reward-relative coordinate; instead it is a linear signed distance that stays at 0 throughout the full 50-cm reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is explicitly thresholded into seven categories: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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

iii. Step 5 says the agent preferred explicit comparisons over `np.digitize` so that edge behavior would be unambiguous and match the verbal bin definitions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[start, stop)` samples used for the neural slice, so the distance label at each frame aligns to the neural frame at the same index.

ii.
```python
pos = position[start:stop]
signed_distance, dist_class = distance_classes(pos, REWARD_ZONES[zone_label])
...
neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```

iii. Step 5 says the conversion retains native framewise samples instead of spatially rebinned matrices, precisely to preserve this time alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float64)
...
pos = position[start:stop]
```

iii. Step 2 identifies `position` as the synchronized VR position signal in centimeters, and Step 5 maps it directly to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The only processing is discretization of the raw position values into five 90-cm corridor bins.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    classes = np.full(position.shape, -1, dtype=np.int8)
    classes[position < 90] = 0
    classes[(position >= 90) & (position <= 180)] = 1
    classes[(position > 180) & (position <= 270)] = 2
    classes[(position > 270) & (position <= 360)] = 3
    classes[position > 360] = 4
    return classes
```

iii. Step 5 says these are “5 equal-sized bins spanning the 450-cm track,” with explicit boundary conventions rather than relying on implicit `np.digitize` defaults.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into `<90`, `90–180`, `>180–270`, `>270–360`, and `>360` cm categories.

ii.
```python
classes[position < 90] = 0
classes[(position >= 90) & (position <= 180)] = 1
classes[(position > 180) & (position <= 270)] = 2
classes[(position > 270) & (position <= 360)] = 3
classes[position > 360] = 4
```

iii. Step 5 says the agent intentionally documented and hard-coded the edge conventions to avoid off-by-one ambiguity at the class boundaries.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position uses the same `[start, stop)` frame indices as the neural data, so it is aligned one sample at a time.

ii.
```python
pos = position[start:stop]
...
neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```

iii. Step 5 says all requested outputs are kept as native time-varying frame sequences aligned to the imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float64)
...
(lick[start:stop] > 0).astype(np.int8)
```

iii. Step 2 documents that `lick` is a cumulative per-frame count stream, not already binary.

## 9-b. What processing is involved in computing `output` *Lick*?

i. First, trials with bad lick-sensor behavior are removed; then the remaining per-frame lick counts are binarized as `lick > 0`.

ii.
```python
bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])
...
if bad_lick[trial_i]:
    continue
...
(lick[start:stop] > 0).astype(np.int8)
```

iii. Step 4 says the paper’s final licking QC reproduces 81 bad trials at the `>30%` threshold, and Step 5 says those trials were excluded because a binary lick output cannot faithfully encode a known sensor failure.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced on the same per-trial frame interval as neural data and then binarized framewise.

ii.
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    ...
    output_trial = np.vstack(
        (
            ...,
            (lick[start:stop] > 0).astype(np.int8),
            ...
        )
    ).astype(np.int8)
```

iii. Step 2 says lick is already on the shared imaging-volume clock, so no interpolation or re-alignment was needed.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session identifier/scene string in the NWB file plus the trial ordinal, not from the framewise `reward_zone` stream.

ii.
```python
scene = nwb.identifier.rstrip("/").split("/")[-1]
initial_zone, switched_zone = parse_scene_zones(scene)
...
zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
...
np.full(T, ZONE_CODES[zone_label], dtype=np.int8)
```

iii. Step 5 says the agent chose the same conceptual source as the paper’s `get_reward_zones`: parse the scene and apply the known switch timing, while validating that against the environment stream and reward positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed to get the starting zone and, if present, the switched zone. The code then uses trial index `<30` versus `>=30` to choose the active zone, maps `A/B/C` to `0/1/2`, and repeats that per-trial value across all frames.

ii.
```python
def parse_scene_zones(scene: str) -> tuple[str, str | None]:
    match = re.search(r"Location([ABC])(?:_to_([ABC]))?$", scene)
    if match is None:
        match = re.search(r"Env\d_([ABC])_to_Env\d_([ABC])$", scene)
    if match is None:
        raise ValueError(f"Cannot parse reward zone(s) from scene {scene!r}")
    return match.group(1), match.group(2)
...
zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
np.full(T, ZONE_CODES[zone_label], dtype=np.int8)
```

iii. Step 5 explicitly lists reward-zone parsing as a key design decision and says it must support both single-environment and cross-environment identifier patterns.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse `Reward.timestamps` events.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
```

iii. Step 4 says the agent compared reward timestamps against the trial intervals and found an overall reward rate consistent with the paper.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each trial is labeled `1` if any reward timestamp falls within that trial’s `[start, stop)` interval, else `0`; the result is then repeated across all frames of the trial.

ii.
```python
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
...
np.full(T, outcomes[trial_i], dtype=np.int8)
```

iii. Step 4 says this criterion is effectively equivalent to the paper’s rewarded-vs-omitted classification on the supplied data and preserves the expected omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles data issues by validation plus a few targeted fixes: it truncates neural series to behavior length when there is an extra trailing neural row, excludes bad-lick trials, ignores negative environment placeholders when checking constancy, and raises explicit errors for inconsistent or non-finite data rather than silently imputing values.

ii.
```python
if f_series.data.shape[0] < n_behavior or fn_series.data.shape[0] < n_behavior:
    raise ValueError(f"Neural series in {plane} is shorter than behavior")
...
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
...
bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])
...
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
if env_values.size != 1:
    raise ValueError(f"Trial {trial_i} does not have one valid environment")
```

iii. Step 2 documents the one-frame dual-plane termination artifact and the bad-lick edge case; Step 4 says to ignore unmatched trailing neural rows, and to remove the 81 lick-sensor failures instead of forcing them into a binary lick label.

## 13-a. What are the most time-consuming steps of the code?

i. The agent identified raw fluorescence loading and recomputation of dF/F plus OASIS events as the main runtime cost, with per-session NWB opens and final serialization also contributing.

ii.
```python
for plane_i, plane in enumerate(plane_names):
    events, diag = process_plane(...)
...
def reference_dff(...):
    for start, stop in zip(starts, stops):
        ...

def reference_events(...):
    for start, stop in zip(starts, stops):
        events[:, start:stop] = dcnv.oasis(...)
...
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 says “Raw fluorescence and neuropil are large and OASIS must operate on each cell/trial,” and Step 7 gives runtime estimates dominated by the neural core plus full-dataset serialization.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious non-vectorized work is the repeated per-trial looping in `reference_dff`, `reference_events`, and the outer session/plane loops. The agent already vectorized across neurons within a trial, but did not eliminate the trial loops.

ii.
```python
for start, stop in zip(starts, stops):
    fneu = neuropil[:, start:stop]
    ...

for start, stop in zip(starts, stops):
    events[:, start:stop] = dcnv.oasis(...)

for session_i, path in enumerate(files):
    ...
    for plane_i, plane in enumerate(plane_names):
        events, diag = process_plane(...)
```

iii. Step 6 says the speedups came from vectorizing across neurons and filtering interneurons before OASIS, implying that the remaining unavoidable loops are over variable-length trials, sessions, and planes.

## 13-c. What processing does the code repeat multiple times?

i. It repeats trial-interval iteration in several passes over the same session: once to compute dF/F, once to deconvolve, once to mark the valid mask/correlations, once to identify outcomes and bad-lick trials, and once more to build trial arrays.

ii.
```python
for start, stop in zip(starts, stops):
    ...

speed_corr = correlations_with_speed(dff, speed, valid_mask)
...
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])
...
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    ...
```

iii. This follows from the agent’s Step 6 notes: it reduced memory by processing sequentially, but that design means several separate passes over the same trial boundaries.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Outside the core conversion path, the code spends work on optional diagnostics and some bookkeeping that the downstream decoder does not need: processing plots, diagnostic traces, signed-distance caches for plotting, and aggregate conversion statistics.

ii.
```python
if collect_diagnostics:
    diagnostics.update(
        {
            "raw_f": fluorescence[chosen].copy(),
            "raw_fneu": neuropil[chosen].copy(),
            "dff": dff[chosen].copy(),
            "events": events[0].copy(),
            **trace_processing(...),
        }
    )
...
def plot_processing(session: dict, diagnostics: dict) -> Path:
    ...

stats = dataset_statistics(data)
data["metadata"]["conversion_statistics"] = stats
```

iii. Step 7 says the plots were generated and inspected as sanity checks, not because the decoder needs them. The diagnostic structures and summary statistics are therefore extra validation work rather than required downstream inputs.
