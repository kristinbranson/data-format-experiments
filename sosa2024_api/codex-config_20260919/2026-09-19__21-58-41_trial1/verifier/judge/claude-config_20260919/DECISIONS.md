# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `/app/data` is discovered with a single recursive glob `sub-*/*.nwb`, sorted lexically, and each file is opened with `pynwb.NWBHDF5IO` (no `h5py`). All 152 files (11 subjects) are processed in one pass; there is no separate survey/pre-pass. Within each file the agent reads `nwb.subject.subject_id`, `nwb.session_id`, `nwb.identifier` (scene), the `behavior/BehavioralTimeSeries` time series (`position`, `speed`, `lick`, `environment`, `trial_start`, `teleport`, `Reward`), and the `ophys` containers (`Fluorescence`, `Neuropil`, `ImageSegmentation/PlaneSegmentation`). Trials are the paired `[trial_start, teleport)` intervals.

ii.
```python
DATA_ROOT = Path("/app/data")

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
        scene = nwb.identifier.rstrip("/").split("/")[-1]
        behavior = nwb.processing["behavior"]["BehavioralTimeSeries"].time_series
        ophys = nwb.processing["ophys"]
        starts, stops = trial_events(behavior)
```

iii. From CONVERSION_NOTES Step 2/Step 9: the data directory is one level deep with `sub-<id>` directories; 152 `behavior+ophys.nwb` assets across 11 mice were found and all 152 appear in the output metadata ("All 152 source filenames are unique and present in metadata, so no sessions were lost"). The notes justify using only `pynwb` per the task constraint, and the 12,216 paired start/teleport intervals were cross-checked against the paper's 12,376-trial lick-QC denominator, with the 160-trial difference attributed to m11 task days 1–2, for which no ophys asset exists.

## 1-b. How are the data split into subjects?

i. One subject per `sub-<id>` directory. The subject list is the sorted set of directory names with the `sub-` prefix stripped; each session's `subject_idx` is looked up from `nwb.subject.subject_id` read out of the file itself (so the directory name and the in-file subject id are implicitly cross-validated — a mismatch would raise `KeyError`).

ii.
```python
subjects = sorted({p.parent.name.removeprefix("sub-") for p in files})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[result["info"]["subject"]])   # info["subject"] = nwb.subject.subject_id
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Subject list is unique sorted mouse IDs"; "Preserve deterministic lexical subject/session order." Step 9 records 11 subjects, matching the paper's 11 switch mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is exactly one session. Sessions are ordered by the lexically sorted file paths (so grouped by subject, ascending session number). No cross-day ROI alignment is attempted; neurons are treated as independent per session.

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
...
for session_i, path in enumerate(files):
    result, diagnostics = process_session(path, args.show_processing and session_i < 2)
    neural.append(result["neural"])
    ...
    brain_region_idx.append(np.zeros(result["info"]["n_neurons"], dtype=np.int64))
```
```python
info = {"session_id": f"{subject}_ses-{session_id}", "subject": subject,
        "nwb_session_id": session_id, "scene": scene, ...}
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "each NWB file is one session." Step 9 reports 152 sessions = 14 days per mouse except m11, which begins at day 3, consistent with the paper's statement that m11 imaging started on day 3.

## 1-d. How are the data split into trials?

i. A trial is the half-open sample interval `[trial_start event, teleport event)` taken from the `trial_start` and `teleport` behavior time series. The teleport sample itself is excluded, so the inter-trial gray teleport zone is never part of a trial. The code asserts that the number of starts equals the number of teleports and that the two event streams strictly alternate. The NWB `trial number` time series is deliberately *not* used.

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
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    ...
    pos = position[start:stop]
```

iii. CONVERSION_NOTES Step 4: "NWB event arrays encode the legacy events on a 0-based time axis, so use `[start_event_index, teleport_event_index)` with no extra offset", matching the reference repo's `start-1:stop-1` slicing of its 1-based index arrays. Step 10 issue log: "One trailing false trial-number value: resolved by paired start/teleport intervals rather than unique trial-number values" (m11 ses-03). The edge-case audit confirmed strict alternation and 12,216 paired intervals in all 152 files.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level exclusion is the paper's lick-sensor QC: a trial is dropped if more than 30% of its frames have a cumulative lick count > 2. This flags exactly 81 trials (0.66%), the number reported in the Methods. Whereas the paper sets the lick vector to NaN on those trials, the agent removes the whole trial, because the target format has no NaN representation for the required categorical lick output. No minimum-trial-length filter is applied (observed min T = 96 samples). A session must retain at least 2 trials or `validate_session` raises.

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
```python
def validate_session(neural, inputs, outputs) -> None:
    if not (len(neural) == len(inputs) == len(outputs) and len(neural) >= 2):
        raise ValueError("Session trial lists are inconsistent or contain fewer than two trials")
```

iii. CONVERSION_NOTES Step 4: "Threshold >0.30 reproduces exactly 81 flagged trials; >0.35 gives 69 and >0.50 gives 44 ... Follow the paper's final stated >0.30 rule. Because lick is a required categorical output and cannot be represented as NaN, exclude those 81 whole trials (0.66%) from this multi-output dataset, rather than treating known sensor failure as 'no lick.'" Step 5 Key Decision 4 adds that no reward/omission balancing, occupancy matching or speed threshold is applied, since those were analysis-specific in the paper and would distort the requested output distributions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Raw suite2p traces: `ophys/Fluorescence` (F) and `ophys/Neuropil` (Fneu) `roi_response_series`, one series per imaging plane, restricted to ROIs marked as cells in `ophys/ImageSegmentation/PlaneSegmentation['iscell'][:,0]`. The NWB `Deconvolved` series is explicitly *not* used. Speed (`behavior['speed']`) is also used, but only as a filtering covariate.

ii.
```python
f_series = ophys["Fluorescence"].roi_response_series[plane]
fn_series = ophys["Neuropil"].roi_response_series[plane]
roi_indices = np.asarray(f_series.rois.data[:], dtype=np.int64)
local_cells = np.flatnonzero(iscell[roi_indices])
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
neuropil = np.asarray(fn_series.data[:n_behavior, local_cells], dtype=np.float32).T
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "NWB `Deconvolved` is on raw Suite2p scale (example median nonzero tens–hundreds); recomputed reference events are ~0–1 and correlate only ~0.40 median across cells with NWB `Deconvolved`. F and Fneu are available. ... Do not use the incompatible raw Suite2p `Deconvolved` export." Step 5 records the same mapping to `preprocessing.dff`.

## 2-b. How is the `neural` data processed?

i. The agent re-implements the paper's `preprocessing.dff` per plane, trial by trial: subtract `0.7 * Fneu`, add back the within-trial mean of `0.7 * Fneu` so the ratio is a true dF/F, Gaussian-smooth along time with sigma = 15 samples, take a 300-sample running minimum followed by a 300-sample running maximum (the Methods' ~20 s maximin window), form `(F - baseline)/|baseline|`, then smooth with a 2-sample Gaussian. The dF/F of retained cells is deconvolved with suite2p's OASIS at `tau = 0.7` and `fs = 15.5078125` Hz. Planes are concatenated along the neuron axis afterwards. Samples outside `[start, stop)` are never written (they stay NaN in the dF/F buffer and 0 in the events buffer) and are never emitted. The reference repo's per-mouse/per-day `keep_teleports` table is **not** implemented — the baseline window is always restricted to the lap.

ii.
```python
NEUROPIL_COEF = 0.7
OASIS_TAU_S = 0.7
FRAME_RATE = 15.5078125

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
        events[:, start:stop] = dcnv.oasis(dff[:, start:stop], batch_size=2000,
                                           tau=OASIS_TAU_S, fs=FRAME_RATE)
    return events
```
```python
events = np.concatenate(plane_events, axis=0) if len(plane_events) > 1 else plane_events[0]
```

iii. CONVERSION_NOTES Step 3: "dF/F is calculated independently within each trial: neuropil-corrected fluorescence uses a 20-s maximin baseline; dF/F is `(F - baseline) / abs(baseline)`; then a two-sample (~0.129-s s.d.) Gaussian smooth is applied. OASIS with a canonical calcium kernel produces deconvolved event activity." Step 10 check 3: "`reference_dff` (line 86) and `reference_events` (line 119) port `preprocessing.dff` (line 289): coefficient 0.7, within-trial mean neuropil restoration, sigma-15 smoothing, 300-frame min then max, `(F-F0)/abs(F0)`, sigma-2 smoothing, OASIS tau 0.7 at 15.5078125 Hz." Step 3 justifies dropping teleport samples: "Teleport data are normally excluded because laser power was reduced there in most sessions and teleport position is not part of the 0–450-cm task."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both from the paper. (1) Only ROIs with `iscell[:,0] > 0` (suite2p + manual curation) are loaded. (2) Putative interneurons are excluded: any remaining cell whose dF/F has Pearson r > 0.5 with running speed over all within-trial samples. Cells with a non-finite correlation (degenerate variance) are also dropped. This removed 402 of 138,678 curated cells (0.29%), leaving 138,276 (mean 909.7/session, range 154–2,323). No place-cell selection is applied.

ii.
```python
INTERNEURON_R_THRESHOLD = 0.5

def correlations_with_speed(dff, speed, valid_mask):
    x = dff[:, valid_mask].astype(np.float64, copy=False)
    y = speed[valid_mask].astype(np.float64, copy=False)
    x -= np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x @ y) / denominator
```
```python
iscell_table = np.asarray(ps["iscell"][:])
iscell = iscell_table[:, 0] > 0
...
local_cells = np.flatnonzero(iscell[roi_indices])
...
valid_mask = np.zeros(n_behavior, dtype=bool)
for start, stop in zip(starts, stops):
    valid_mask[start:stop] = True
speed_corr = correlations_with_speed(dff, speed, valid_mask)
keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)
kept_dff = np.ascontiguousarray(dff[keep])
events = reference_events(kept_dff, starts, stops)
```

iii. CONVERSION_NOTES Step 4: "First retain `iscell[:,0] > 0`, then recompute dF/F and exclude cells with Pearson r > 0.5 versus speed across valid track samples. Pool planes after filtering." Step 5 Key Decision 3: "manual `iscell` plus dF/F–speed r>0.5, with no place-cell-only restriction. The paper decoder used selected place-cell subtypes for its scientific question, but the general downstream decoder should receive all curated pyramidal neurons and not leak output-derived place-cell selection." Step 4 also reconciles the 154–2,323 range against the manuscript's 155–2,172, attributing the difference to the DANDI release's curation version rather than inventing an extra cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The requested alignment event is trial start, which requires no extra work: the neural rows and the behavior samples share one index axis, so slicing `events[:, start:stop]` with the same `start` produced by `trial_start` yields data that begins exactly at trial start. `off_start = 0.0`; `off_end = None` because trials have variable duration. Ten dual-plane files have one extra trailing neural row, which lies beyond the last teleport and is truncated to the behavior length before any slicing.

ii.
```python
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
...
neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```
```python
"temporal_alignment_event": "trial_start: entry onto the 0-cm start of the virtual linear track",
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 4: "Use behavior timestamps / 64.483627-ms bins as authoritative and align neural rows by index; ignore only unmatched trailing neural row beyond all trial intervals." Step 5 Key Decision 1: "retain every 64.483627-ms imaging-volume sample from start through the sample before teleport. This preserves exact neural/behavior alignment ... `off_start=0`, `off_end=None` because end offset varies." Step 10 reports plot inspection showing "no one-frame shift at trial starts/ends".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 64.483627 ms per bin (15.5078125 Hz, the per-plane imaging rate). No rebinning, resampling, or interpolation of any stream is performed — data are kept at the native acquisition rate. The frame rate is hard-coded as a constant and then validated per session against the behavior timestamp spacing, which fails loudly if any session differs. For dual-plane files the stored scanner `rate` is ~31 Hz, but each plane's rows are at half that, which is what the code assumes.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
dt = np.diff(timestamps)
if not np.allclose(dt, 1.0 / FRAME_RATE, rtol=0, atol=1e-9):
    raise ValueError(f"Unexpected behavior sample interval in {path.name}")
```
```python
"time_bin_size": TIME_BIN_MS,
"sample_rate_hz": FRAME_RATE,
```

iii. CONVERSION_NOTES Step 3: "Imaging/behavior are synchronized at the ~15.5-Hz imaging-plane rate. Dual-plane acquisitions alternate at ~31 Hz but yield ~15.5 Hz per plane." Step 5 Key Decision 1 keeps native bins to preserve alignment; Step 10 check 3 states "No temporal rebinning is applied." Step 9 records the converted bin size as 64.483627 ms, matching the reference data.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attribute of the `position` behavior time series (all behavior series share one timestamp vector; the code asserts equal lengths for `position`, `speed`, `lick`, `environment` and uniform spacing of 1/15.5078125 s).

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
n_behavior = len(position)
if not all(len(x) == n_behavior for x in (timestamps, speed, lick, environment)):
    raise ValueError("Behavior series lengths differ")
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior timestamps → `input[0]`: `timestamps[start:stop] - timestamps[start]`, seconds, one value/frame. Time-varying; begins at exactly 0." The notes treat the behavior timestamps as the authoritative clock for both streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, giving a per-frame time series that starts at exactly 0.0 and increments by 64.483627 ms. Stored as float32.

ii.
```python
input_trial = np.vstack(
    (
        timestamps[start:stop] - timestamps[start],
        np.full(T, env),
        np.full(T, trial_i),
        np.full(T, previous_outcome),
    )
).astype(np.float32)
```

iii. CONVERSION_NOTES Step 9 spot checks: "inputs began at 0 with 0.064483627-s increments". No further justification is given beyond the requested definition.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment step is needed — the same `start:stop` index range slices the neural event matrix and the timestamp vector, and the per-session assertion that behavior timestamps are uniformly spaced at the neural frame rate guarantees the two streams are on one clock.

ii.
```python
input_trial = np.vstack((timestamps[start:stop] - timestamps[start], ...))
neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```
```python
if n.shape[0] != nneurons or not (n.shape[1] == x.shape[1] == y.shape[1]):
    raise ValueError(f"Trial {trial}: inconsistent dimensions")
```

iii. CONVERSION_NOTES Step 10 check 3 (temporal alignment): "behavior timestamps define 64.483627-ms samples; neural rows use identical indices and planes are pooled over neurons." Independent `np.allclose` sanity checks in `cache/sanity_check_conversion.py` reproduced full input trials from raw NWB.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series (one value per frame; 0 = ENV1, 1 = ENV2, with negative sentinel values possible outside the track).

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float64)
...
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior `environment` → `input[1]`: 0=ENV1, 1=ENV2 ... Do not use session identifier alone because day-8 sessions switch environments at trial 30." The time series is preferred over the scene name precisely because 11 sessions change environment mid-session.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial the distinct non-negative `environment` values are taken; the code requires there to be exactly one, and that it be 0 or 1, otherwise it raises. The single value is rounded to int and broadcast across all T frames of the trial.

ii.
```python
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
if env_values.size != 1:
    raise ValueError(f"Trial {trial_i} does not have one valid environment")
env = int(round(float(env_values[0])))
if env not in (0, 1):
    raise ValueError(f"Trial {trial_i}: invalid ENV value {env}")
...
np.full(T, env),
```

iii. CONVERSION_NOTES Step 5 planned sanity checks include "Assert ... constant environment within trials"; the Step 9 consistency table reports the converted per-trial ENV1/ENV2 split as [51.067%, 48.933%], "Exact" against the raw-derived value. (Note: the Step 5 mapping table describes this as a "per-trial constant median"; the implemented rule is the unique non-negative value, which is equivalent given the verified constancy.)

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Not from any stored variable: it is the zero-based ordinal of the trial within the session, i.e. the enumerate index over the paired `trial_start`/`teleport` intervals. The NWB `trial number` time series is deliberately not used.

ii.
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    ...
    np.full(T, trial_i),
```

iii. CONVERSION_NOTES Step 5 mapping table: "Paired-trial ordinal / source `trial number` → `input[2]`: Zero-based continuous trial ordinal ... Use interval ordinal to avoid one trailing false trial-number fragment in m11 session 03." Step 10 lists the stored trial-number artefact among the issues resolved this way.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the ordinal across the trial's T frames as float32. Crucially the ordinal is the *source* index, so when a bad-lick trial is dropped the remaining trial numbers keep the gap rather than being renumbered — trial numbers remain a faithful measure of within-session experience. Values run 0–99.

ii.
```python
for trial_i, (start, stop) in enumerate(zip(starts, stops)):
    if bad_lick[trial_i]:
        continue
    ...
    np.full(T, trial_i),
```

iii. CONVERSION_NOTES Step 10 check 5: "zero-based input ordinal preserves gaps when bad trials are skipped". The verification log confirms the input range [0.0, 99.0].

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavior time series' sparse `timestamps` (reward-delivery events), compared against the behavior timestamps at the previous trial's boundaries.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
...
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Sparse `Reward.timestamps` in previous paired interval → `input[3]`: Previous raw trial reward outcome, 0=omitted, 1=rewarded". Step 4 verified equivalence to the reference `get_trial_types` conjunction (reward AND reward-zone entry): "10,342 trials have both; 1,822 have neither; 52 omissions have a zone-entry event but no reward", i.e. no trial has a reward without a zone entry, so the simpler criterion gives identical labels.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A per-trial binary outcome vector is computed once for all source trials; the input for trial *i* is `outcomes[i-1]`, with 0 for the first trial of a session. The value is broadcast across the trial's T frames. The previous trial is the true preceding *source* trial, even if that trial was removed by the lick QC.

ii.
```python
previous_outcome = int(outcomes[trial_i - 1]) if trial_i > 0 else 0
...
np.full(T, previous_outcome),
```

iii. CONVERSION_NOTES Step 5 mapping table note: "Previous outcome is based on the true preceding source trial even if that preceding trial is excluded for bad lick sensing." Step 10 check 5 confirms "first-trial previous outcome is 0". Step 12 additionally verified the input cannot leak the current outcome: "current versus previous outcome correlation is only 0.0234 and reward rates conditional on previous omission/reward are 82.74%/85.01%".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behavior time series plus the active reward zone for that trial. The active zone is determined the way the paper's own `behavior.get_reward_zones` does it: parse the A/B/C label(s) out of the scene string in `nwb.identifier` (handling both `Env#_Location<A|B|C>[_to_<A|B|C>]` and cross-environment `Env#_<A>_to_Env#_<B>` forms) and switch from the initial to the final zone at source trial ordinal 30. Zone coordinates are A = 80–130, B = 200–250, C = 320–370 cm. The `reward_zone` time series is *not* used to infer the label.

ii.
```python
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def parse_scene_zones(scene: str) -> tuple[str, str | None]:
    match = re.search(r"Location([ABC])(?:_to_([ABC]))?$", scene)
    if match is None:
        match = re.search(r"Env\d_([ABC])_to_Env\d_([ABC])$", scene)
    if match is None:
        raise ValueError(f"Cannot parse reward zone(s) from scene {scene!r}")
    return match.group(1), match.group(2)
```
```python
scene = nwb.identifier.rstrip("/").split("/")[-1]
initial_zone, switched_zone = parse_scene_zones(scene)
...
zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone
signed_distance, dist_class = distance_classes(pos, REWARD_ZONES[zone_label])
```

iii. CONVERSION_NOTES Step 4: "`get_reward_zones` parses scene and switches after 30 trials; A/B/C = 80–130/200–250/320–370 cm ... Parse zone labels from the NWB identifier and use source trial ordinal `<30` versus `>=30`". This is validated against the data two ways: "98.87% of within-trial reward samples fall in the assigned zone ±1 cm; small overshoots reflect frame sampling/auto-reward after zone end", and "Cross-environment identity changes occur exactly at trial index 30 in all 11 day-8 files" (re-verified over all 152 files in Step 10 check 5). Step 3 cites the Methods: "On day 3 (switch one), the zone was moved after 30 trials".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. A signed linear distance to the nearest point of the active 50-cm zone: `position - zone_start` before the zone, exactly 0.0 anywhere inside the zone (inclusive of both edges), and `position - zone_end` after it. This is *not* the paper's circular reward-relative coordinate (which is centred on the zone start) — it was adapted to satisfy the requested "distance to any location in the reward zone" definition.

ii.
```python
def distance_classes(position, zone):
    start, stop = zone
    distance = np.where(position < start, position - start,
                        np.where(position > stop, position - stop, 0.0))
    ...
    return distance, classes
```

iii. CONVERSION_NOTES Step 3: "The paper's RR coordinate is circular and centered at reward-zone **start**. The requested signed distance bins instead require a linear distance 'to any location in the reward zone,' so distance must be zero throughout the 50-cm zone, negative before its start, and positive after its end." Step 5 mapping table flags this as "adapted because paper RR coordinate is circular distance to zone start".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit comparisons rather than `np.digitize`: 0 for d < −50; 1 for −50 ≤ d < −10; 2 for −10 ≤ d < 0; 3 for d == 0 (anywhere in the zone); 4 for 0 < d ≤ 10; 5 for 10 < d ≤ 50; 6 for d > 50. Any sample left unclassified raises. The resulting time-weighted distribution is [0.2513, 0.1020, 0.0733, 0.2385, 0.0207, 0.0718, 0.2426].

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
if np.any(classes < 0):
    raise ValueError("Unclassified reward-zone distance")
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "use explicit comparisons, not implicit `np.digitize` defaults, and describe them in metadata. This eliminates off-by-one ambiguity at stated class edges." The class list is a direct transcription of the Decoder Task bin table, with class 3 covering the whole 50-cm zone per the "distance to any location in the reward zone" wording.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No extra alignment — `position` is sliced with the identical `start:stop` indices used for the neural events, and `validate_session` enforces that neural, input and output arrays have exactly the same number of timepoints in every trial.

ii.
```python
pos = position[start:stop]
signed_distance, dist_class = distance_classes(pos, REWARD_ZONES[zone_label])
...
neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))
```

iii. CONVERSION_NOTES Step 10 check 2 independently recomputed all six outputs from raw NWB for m11 ses-03 trials 1/5/30/79 and m17 ses-08 trials 0/1/30/79 and matched with `np.allclose`; the `--show-processing` plots overlay position, signed distance and the distance class on one time axis "and show no one-frame shift at trial starts/ends".

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the 450 cm virtual corridor), used unmodified.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float64)
...
pos = position[start:stop]
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior `position` → `output[1]` absolute position ... Direct synchronized `vr_data['pos']` equivalent".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing the trial window and discretizing; no smoothing, clipping or renormalization. Because the teleport sample is excluded from every trial, the negative positions of the inter-trial gray zone never enter the data.

ii.
```python
pos = position[start:stop]
output_trial = np.vstack((dist_class, position_classes(pos), ...)).astype(np.int8)
```

iii. CONVERSION_NOTES Step 3: teleport samples are excluded because "teleport position is not part of the 0–450-cm task"; Step 7 plot review: "Track position progresses smoothly from ~0 to ~450 cm".

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90 cm classes over the 450 cm track, again by explicit comparison with open end bins: 0 for p < 90; 1 for 90 ≤ p ≤ 180; 2 for 180 < p ≤ 270; 3 for 270 < p ≤ 360; 4 for p > 360. The open first/last bins absorb the handful of samples marginally outside [0, 450]. Resulting distribution [0.2120, 0.1767, 0.2312, 0.2264, 0.1536].

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

iii. CONVERSION_NOTES Step 5 mapping table: "Classes: `<90`, `90–180`, `180–270`, `270–360`, `>360`; deterministic shared-edge convention assigns exact 90 upward and exact internal/right endpoints to the lower stated closed range ... Exact internal boundaries are vanishingly rare in continuous samples." Key Decision 8 again motivates explicit comparisons over `np.digitize`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `start:stop` slice as the neural data; no resampling or offset. `validate_session` checks equal T.

ii.
```python
pos = position[start:stop]
...
if n.shape[0] != nneurons or not (n.shape[1] == x.shape[1] == y.shape[1]):
    raise ValueError(f"Trial {trial}: inconsistent dimensions")
```

iii. Same justification as 7-d: CONVERSION_NOTES Step 10 checks 2 and 6 (independent raw-data `np.allclose` reconstruction; shape/boundary validation; plot inspection).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (cumulative lick counts per frame).

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior cumulative `lick` → `output[3]` lick".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. First, quality control at the trial level (see 1-e): trials where >30% of frames have a cumulative count > 2 are recognised as capacitive-sensor failures and removed entirely. Second, on the remaining trials the counts are binarized as `lick > 0`. Result: 22.26% of retained frames are licks.

ii.
```python
bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])
...
if bad_lick[trial_i]:
    continue
...
(lick[start:stop] > 0).astype(np.int8),
```

iii. CONVERSION_NOTES Step 3 curation: "set the lick vector to NaN on trials where >30% of 64.5-ms frames contain cumulative lick count >2 (81 trials)"; Step 4 resolution: "Because lick is a required categorical output and cannot be represented as NaN, exclude those 81 whole trials (0.66%) ... rather than treating known sensor failure as 'no lick.'" The reference helper's "cap count >1 to 1" behaviour motivates the `> 0` binarization.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `start:stop` index slice as the neural data; no shift, no smoothing.

ii.
```python
(lick[start:stop] > 0).astype(np.int8),
```

iii. Same as 7-d/8-d: verified by independent raw-NWB `np.allclose` reconstruction (Step 10 check 2) and by the lick-binarization panel in `processing_<session>.png`, described in Step 7 as showing that "lick events align".

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The scene string in `nwb.identifier`, together with the source trial ordinal (switch at 30). Same derivation as 7-a; the `reward_zone` time series is used only as an offline validation signal, not as the label source.

ii. See 7-a (`parse_scene_zones`, `zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone`).

iii. See 7-a. CONVERSION_NOTES Step 5 mapping table: "NWB identifier scene + paired trial ordinal → `output[4]` reward zone location ... `behavior.get_reward_zones`".

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The A/B/C label is mapped to 0/1/2 and broadcast across all T frames of the trial (a per-trial variable rendered as a constant time series so all outputs are rectangular). Resulting per-frame distribution [0.3316, 0.3359, 0.3325]; per-trial [34.380%, 32.748%, 32.872%].

ii.
```python
ZONE_CODES = {"A": 0, "B": 1, "C": 2}
...
np.full(T, ZONE_CODES[zone_label], dtype=np.int8),
```
```python
"output_values": [... ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"], ...]
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "create input arrays `(4,T)` and output arrays `(6,T)`, repeating per-trial variables across T. This makes every named dimension unambiguous to the validator and decoder." Step 9 marks the A/B/C fractions as an exact match to the raw-derived expectation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` time series' `timestamps` (reward-delivery events), compared against behavior timestamps at trial boundaries. Same `outcomes` vector used for input 3.

ii.
```python
reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
```

iii. CONVERSION_NOTES Step 4: "Outcome is 1 only when sparse Reward timestamp occurs in `[start, teleport)` (equivalent to reference conjunction here), else 0. Observed reward rate 84.66% matches paper" (~15% random omissions).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per trial, 1 if any reward timestamp falls in the half-open interval `[timestamps[start], timestamps[stop])`, else 0; broadcast across the T frames. Retained data: 10,271 rewarded vs 1,864 omitted trials, i.e. 84.22% of frames rewarded.

ii.
```python
np.full(T, outcomes[trial_i], dtype=np.int8),
```

iii. CONVERSION_NOTES Step 4 (equivalence to `get_trial_types`, above) and Step 12, which verified that the outcome label cannot be inferred from the inputs: "Only 50.62% of all retained frames occur at or after reward delivery; before delivery the per-trial outcome is intentionally not causally observable ... Restricting labels to post-reward time or supplying current reward as input would improve accuracy by changing the requested task or leaking the target, so no such change was made."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Extra trailing neural rows** (10 dual-plane files have one row beyond the behavior length): the neural series is read as `[:n_behavior]`, silently truncating the unmatched row, which the audit confirmed always falls after the last teleport.
- **Neural shorter than behavior**: raises `ValueError` rather than cropping (verified never to occur across all 152 files).
- **Spurious stored `trial number` fragment** (m11 ses-03): avoided by deriving trials from paired `trial_start`/`teleport` events instead.
- **Lick-sensor failures**: the 81 paper-identified trials are dropped (1-e).
- **Environment sentinel values**: negative `environment` samples are filtered before taking the trial's unique value.
- **Degenerate speed correlations**: cells with non-finite r are excluded along with interneurons.
- Everything else is fail-loud: unequal start/teleport counts, non-alternating events, non-uniform timestamp spacing, non-finite raw fluorescence, a plane with no curated cells, all cells rejected, more than one environment per trial, an unclassifiable distance sample, non-finite converted values, inconsistent per-trial shapes, out-of-domain categorical values, or fewer than 2 trials in a session all raise.

ii.
```python
if f_series.data.shape[0] < n_behavior or fn_series.data.shape[0] < n_behavior:
    raise ValueError(f"Neural series in {plane} is shorter than behavior")
fluorescence = np.asarray(f_series.data[:n_behavior, local_cells], dtype=np.float32).T
```
```python
dt = np.diff(timestamps)
if not np.allclose(dt, 1.0 / FRAME_RATE, rtol=0, atol=1e-9):
    raise ValueError(f"Unexpected behavior sample interval in {path.name}")
```
```python
keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)
if not np.any(keep):
    raise ValueError(f"All cells rejected in {plane}")
```
```python
if not (np.all(np.isfinite(n)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
    raise ValueError(f"Trial {trial}: non-finite converted data")
domains = [range(7), range(5), range(5), range(2), range(3), range(2)]
for dim, allowed in enumerate(domains):
    values = np.unique(np.concatenate([y[dim] for y in outputs]))
    if not set(values.tolist()).issubset(set(allowed)):
        raise ValueError(f"Output {dim} has invalid values {values}")
```

iii. CONVERSION_NOTES Step 10 "Issues Found and Resolved" lists the deconvolution mismatch, the ten one-row neural overruns ("resolved by behavior-length truncation; row is after the valid behavior range and never part of a trial"), the trailing false trial-number value, and the bad-lick trials. Step 10 check 5 describes `cache/edge_case_checks.py`, which re-opened all 152 NWBs to confirm trial lengths, ordinal gaps, first-trial previous outcome, strict alternation, the trial-30 switch, and the overrun truncation.

## 13-a. What are the most time-consuming steps of the code?

i. Per the printed timings (0.46 s for the smallest single-plane session to ~2.7 s for the largest dual-plane one; 350.08 s total for 152 sessions), the dominant costs are: (1) reading the F and Fneu arrays for curated ROIs out of each NWB file (HDF5 I/O with column fancy-indexing); (2) the per-trial maximin baseline in `reference_dff` (Gaussian + min/max filters over n_cells × trial-length); (3) OASIS deconvolution in `reference_events`; and (4) pickling the 8.865 GiB result, which took 8.40 s. The script prints per-session and total wall-clock so the bottleneck is visible.

ii.
```python
for session_i, path in enumerate(files):
    t0 = time.perf_counter()
    result, diagnostics = process_session(path, args.show_processing and session_i < 2)
    ...
    print(f"[{session_i + 1:3d}/{len(files)}] {result['info']['session_id']}: "
          f"{result['info']['n_neurons']} neurons, {result['info']['n_trials']} trials, "
          f"{result['info']['n_timepoints']} samples, {elapsed:.2f}s", flush=True)
```
```python
write_seconds = time.perf_counter() - write_start
total_seconds = time.perf_counter() - overall_start
print(f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**30:.3f} GiB) "
      f"in {write_seconds:.2f}s; total {total_seconds:.2f}s", flush=True)
```

iii. CONVERSION_NOTES Step 6: "Raw fluorescence and neuropil are large and OASIS must operate on each cell/trial; recomputing the paper signal necessarily reads both streams." Step 7 estimated ~8–10 min for the full run including "152 opens, validation, and ~8.8-GiB serialization"; the actual run took 5.83 min, inside the 15-minute budget, so no further optimization was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops, and what could be done: the per-trial loops in `reference_dff` and `reference_events` (inherently per-trial, since the maximin baseline and the OASIS kernel must not cross trial boundaries, but the filters could be run once on a trial-padded array); the `valid_mask` loop, which is a pure index build that `np.add.reduceat`/`searchsorted` could replace; the `outcomes` list comprehension, which is O(n_trials × n_reward_events) and could be one `np.searchsorted` over `reward_times`; the `bad_lick` list comprehension, which could use `np.add.reduceat` on `lick > 2`; and the main per-trial assembly loop. Loops that were already vectorized: all per-cell work (dF/F over the full cell × time matrix, and the speed correlation as a single matrix–vector product instead of the reference's per-cell `np.corrcoef`).

ii.
```python
def correlations_with_speed(dff, speed, valid_mask):
    x = dff[:, valid_mask].astype(np.float64, copy=False)
    y = speed[valid_mask].astype(np.float64, copy=False)
    x -= np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x @ y) / denominator
```
```python
outcomes = np.asarray(
    [np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) for a, z in zip(starts, stops)],
    dtype=np.int8,
)
bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": "Only `iscell` columns are loaded; processing is vectorized across neurons within each trial; dF/F–speed correlations are vectorized; rejected interneurons are removed before OASIS; planes are processed sequentially; raw arrays and diagnostics are released promptly; outputs use compact int8 and inputs/neural use float32". The remaining loops are over ~80 trials per session, so they are not the bottleneck.

## 13-c. What processing does the code repeat multiple times?

i. Little. Each NWB file is opened exactly once and each F/Fneu array is read exactly once — there is no separate survey pass. Within a session the trial boundary loop is walked four times (dF/F, `valid_mask` construction, OASIS, and trial assembly), each time doing different work. In `--show-processing` mode `trace_processing` recomputes the neuropil correction and maximin baseline for a single displayed cell, duplicating work `reference_dff` already did for that cell. Interneuron filtering is applied before OASIS so no rejected cell is deconvolved.

ii.
```python
dff = reference_dff(fluorescence, neuropil, starts, stops)
valid_mask = np.zeros(n_behavior, dtype=bool)
for start, stop in zip(starts, stops):
    valid_mask[start:stop] = True
speed_corr = correlations_with_speed(dff, speed, valid_mask)
keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)
kept_dff = np.ascontiguousarray(dff[keep])
events = reference_events(kept_dff, starts, stops)
```
```python
if collect_diagnostics:
    chosen = int(np.flatnonzero(keep)[0])
    diagnostics.update({..., **trace_processing(fluorescence[chosen], neuropil[chosen], starts, stops)})
```

iii. CONVERSION_NOTES Step 6: "rejected interneurons are removed before OASIS"; Step 7 speed-up table: "Load only curated ROI columns ... Avoids processing 121,413 rejected ROIs across the full dataset and unnecessary OASIS work". The single-pass design is what brings the full conversion to 5.83 minutes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few small items, all cheap relative to the dF/F and OASIS work:
- `process_session` always builds the `diagnostics` dict holding full-length copies of `position`, `speed`, `lick`, `timestamps`, `outcomes`, `bad_lick`, `starts`, `stops` and the plane's `speed_corr`, even when `--show-processing` is off; it is discarded (`del diagnostics; gc.collect()`) immediately after the session.
- `distance_classes` always returns the continuous signed distance, which is only retained when diagnostics are being collected.
- `trace_processing` (diagnostics mode only) repeats work already done inside `reference_dff`.
- The rectangular `(4, T)` / `(6, T)` layout repeats four per-trial scalars (trial number, previous outcome, reward-zone location, reward outcome) at every timepoint, which is redundant storage the decoder does not need — though this was a deliberate format choice, not an oversight.
- `dataset_statistics` re-walks all outputs at the end to build the distribution summary, which is not part of the saved arrays (it is written to metadata).

ii.
```python
diagnostics = {
    "info": info, "plane": plane_diagnostics[0] if collect_diagnostics else {},
    "starts": starts, "stops": stops, "position": position, "speed": speed,
    "lick": lick, "timestamps": timestamps, "outcomes": outcomes,
    "bad_lick": bad_lick, "kept_ordinals": np.asarray(kept_ordinals),
    "signed_distances": signed_distances,
}
```
```python
        del diagnostics
        gc.collect()
```

iii. Not explicitly addressed in CONVERSION_NOTES; the closest statements are Step 6 ("raw arrays and diagnostics are released promptly") and Step 5 Key Decision 5, which justifies the rectangular repeated-scalar layout as making "every named dimension unambiguous to the validator and decoder".
