# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file in the data root with a single glob, `sub-*/*_behavior+ophys.nwb`, sorts them by (subject number, session number) and converts each one in turn. It does **not** use `pynwb`; it opens the files directly as HDF5 with `h5py` and reads the NWB groups by hard-coded internal paths (`processing/behavior/BehavioralTimeSeries`, `processing/ophys/...`, `general/subject/subject_id`, `general/session_id`). Everything needed is read from each file in one pass: raw `Fluorescence` and `Neuropil` traces, the `iscell` curation column, the `PlaneSegmentation/planeIdx`, and the behaviour series `position`, `speed`, `lick`, `environment`, `trial number`, `reward_zone`, `trial_start`, `teleport`, `Reward`. This yields 152 sessions from 11 mice and 12,216 source trials, of which 12,135 are retained.

ii.
```python
def build_dataset(data_root: Path, max_sessions: int | None = None) -> dict:
    files = sorted(data_root.glob("sub-*/*_behavior+ophys.nwb"), key=_natural_key)
    ...
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}", flush=True)
        arrays, info = convert_session(path)
```
```python
BEHAVIOR = "processing/behavior/BehavioralTimeSeries"
OPHYS = "processing/ophys"
...
def convert_session(path: Path) -> tuple[dict, dict]:
    with h5py.File(path, "r") as nwb:
        behavior = nwb[BEHAVIOR]
        starts, stops = _trial_bounds(behavior)
        rate_hz = _sampling_rate(behavior)
        timestamps = np.asarray(behavior["position/timestamps"])
        position = _read_behavior(behavior, "position")
        speed = _read_behavior(behavior, "speed")
        lick = _read_behavior(behavior, "lick")
        environment = _read_behavior(behavior, "environment")
        trial_number = _read_behavior(behavior, "trial number")
        rzone = _read_behavior(behavior, "reward_zone")
```

iii. From the trajectory, the AI first dumped the full HDF5 layout of a sample file (`f.visititems`) and surveyed every session's shapes, trial counts and unique values before writing the converter, then stated: "The release contains 152 sessions from the 11 switch-task mice (12,216 trials)." It chose raw `h5py` over `pynwb` for speed and for lazy per-trial slicing of the large fluorescence matrices (it verified the ophys datasets are contiguous and uncompressed, so row-range reads are cheap). Reading each file exactly once was an explicit memory strategy: "Trial-wise processing is keeping memory bounded during conversion."

## 1-b. How are the data split into subjects?

i. Subject identity is taken from inside each file (`general/subject/subject_id`), not from the directory name. The unique subject ids across all converted sessions are collected, sorted by their numeric part, and each session is mapped to an index into that list. Result: `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']` (11 mice).

ii.
```python
        subject = _decode_scalar(nwb["general/subject/subject_id"][()])
...
    subjects = sorted(set(session_subjects), key=lambda value: int(re.search(r"\d+", value).group()))
    subject_lookup = {subject: index for index, subject in enumerate(subjects)}
    subject_idx = np.asarray([subject_lookup[s] for s in session_subjects], dtype=np.int64)
```

iii. The AI did not narrate this choice, but it is the internally-stored identity rather than a path convention, so the subject label is guaranteed to be the one the NWB file itself asserts. It matches the 11 `sub-m*` directories and the paper's 11 switch mice.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Files are ordered by `(subject number, session number)` parsed from the path, so sessions appear in chronological order within each mouse. The session number is separately read from `general/session_id` and stored as `experiment_day`, which is also used to decide whether the session is a reward-switch day. No cross-session cell alignment is attempted; each session's neurons are treated as an independent population.

ii.
```python
def _natural_key(path: Path):
    subject = re.search(r"sub-m(\d+)", str(path))
    session = re.search(r"ses-(\d+)", path.name)
    return (int(subject.group(1)), int(session.group(1)))
...
        experiment_day = int(_decode_scalar(nwb["general/session_id"][()]))
```

iii. The AI verified that `general/session_id` is the experiment day by printing it alongside the NWB `identifier` string (e.g. `.../GCAMP11/23_02_2023/Env1_LocationB_to_A`) for all 152 files, and used the day number to drive the switch-day logic in the reward-zone inference. `experiment_day` is also recorded per session in `metadata['session_info']`.

## 1-d. How are the data split into trials?

i. A trial is the half-open frame interval `[trial_start event, teleport event)` — the on-track lap, excluding the teleport/ITI period. Starts are the non-zero samples of the `trial_start` behaviour series, ends are the non-zero samples of `teleport`. Two sanity checks are enforced: the number of starts must equal the number of teleports, and every teleport must strictly follow its start.

ii.
```python
def _trial_bounds(behavior) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(_read_behavior(behavior, "trial_start") > 0)
    stops = np.flatnonzero(_read_behavior(behavior, "teleport") > 0)
    if len(starts) != len(stops):
        raise ValueError(f"Unmatched trial starts ({len(starts)}) and teleports ({len(stops)})")
    if np.any(stops <= starts):
        raise ValueError("A teleport did not follow its corresponding trial start")
    return starts, stops
```

iii. The module docstring states the rule: "Only the on-track interval from each trial-start event up to (but not including) its teleport event is retained." The AI reached this after printing raw samples around each `trial_start`/`teleport` transition and confirming the trial structure (80 starts, 80 teleports per typical session), and it matches the `trial_start_inds`/`teleport_inds` pairing used throughout the paper's own `reward_relative` code.

## 1-e. How are trials filtered based on quality controls?

i. Exactly one trial-level exclusion is applied: the paper's faulty-lick-sensor criterion. A trial is dropped if more than 30% of its imaging-frame samples carry a cumulative lick count greater than 2. This removes 81 of 12,216 trials (0.66%), leaving 12,135. Every other complete start→teleport trial is kept; there is no minimum-duration filter, no speed/engagement filter, and no session- or mouse-level exclusion. Excluded trials still go through the calcium pipeline so that the session-wide interneuron screen sees the whole session; they are only dropped when the output lists are assembled.

ii.
```python
def _bad_lick_trials(lick: np.ndarray, starts: np.ndarray,
                     stops: np.ndarray) -> np.ndarray:
    """Paper criterion: >30% of frame samples have cumulative lick count >2."""
    return np.asarray([
        np.mean(lick[start:stop] > 2) > 0.30
        for start, stop in zip(starts, stops)
    ], dtype=bool)
```
```python
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        if bad_lick[trial]:
            continue
```
```python
        # Process all trials for the paper's session-wide speed-correlation
        # interneuron screen, including trials whose lick sensor was faulty.
```

iii. The AI located the criterion in the Methods ("These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2 ... ~0.65% of all imaged trials, n = 81 out of 12,376") and in the paper's notebook parameters (`correct_sensor_error = True`, `correction_thr = 0.3`, `lick_correction_thr`). It ran the criterion over the whole release before writing the converter and reported: "I found the paper's exact 81 faulty-lick trials using its published criterion (>30% of trial samples with cumulative lick count >2), so those trials cannot supply a valid lick target and will be excluded from the decoder dataset. I'll keep every other complete trial." The reasoning for dropping rather than NaN-ing is that `lick` is a required categorical decoder output and cannot be missing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw Suite2p traces: `processing/ophys/Fluorescence/<plane>/data` (F) and `processing/ophys/Neuropil/<plane>/data` (Fneu), restricted to the manually curated ROIs given by `ImageSegmentation/PlaneSegmentation/iscell[:, 0]`. The NWB `Deconvolved` array is deliberately not used. For the two-plane mice (m17, m18) the ROI-to-column mapping is resolved through each series' `rois` DynamicTableRegion rather than by assuming table row index equals response-matrix column index.

ii.
```python
        is_cell = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation/iscell"][:, 0].astype(bool)
...
        fluorescence_group = nwb[f"{OPHYS}/Fluorescence"]
        neuropil_group = nwb[f"{OPHYS}/Neuropil"]
        for plane_name in sorted(fluorescence_group.keys()):
            table_rows = np.asarray(fluorescence_group[f"{plane_name}/rois"], dtype=np.int64)
            local_keep = np.flatnonzero(is_cell[table_rows])
            if len(local_keep):
                plane_specs.append((
                    fluorescence_group[f"{plane_name}/data"],
                    neuropil_group[f"{plane_name}/data"],
                    local_keep,
                ))
```

iii. The module docstring says the conversion "follows the paper's calcium-processing methods: manually curated Suite2p ROIs are neuropil corrected, converted to trial-wise maximin dF/F, smoothed, deconvolved with OASIS". The AI read `code/src/reward_relative/preprocessing.py::dff` and the notebook parameters, which start from F and Fneu; the paper's analysed signal (`ts_key = 'events'`) is its own OASIS output, not the stored `Deconvolved` series. The two-plane mapping was a mid-run correction: "the run reached 110 sessions, then exposed a two-plane NWB layout difference in m17/m18: their ROI table concatenates both planes, while the response matrix does not use the same direct column indexing ... I'm resolving the ROI-to-column mapping from the NWB references before restarting so those sessions are pooled exactly as in the paper."

## 2-b. How is the `neural` data processed?

i. A re-implementation of the paper's `preprocessing.dff`, applied independently to each trial's frame window: subtract `0.7 × Fneu`; add the trial's mean neuropil back (so the ratio is a true dF/F and the denominator is not near zero); compute a maximin baseline (Gaussian smoothing with σ = 15 samples along time, then a 300-sample running minimum followed by a 300-sample running maximum ≈ the Methods' 20 s window); form `(F − baseline) / |baseline|`; smooth with a 2-sample σ Gaussian; deconvolve with suite2p's OASIS at `tau = 0.7 s` and the per-plane frame rate (15.5078 Hz). Cells from both planes are concatenated. Events are stored as float32. The baseline window is **always** restricted to the lap; the paper's per-mouse/per-day `teleport_sessions` table (days on which the laser was not blanked and the baseline may span the teleport) is not applied.

ii.
```python
def _dff_and_events(fluorescence: np.ndarray, neuropil: np.ndarray,
                    rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Paper's single-trial maximin dF/F and OASIS processing."""
    # Undo neuropil contamination while adding its within-trial mean back, as
    # in reward_relative.preprocessing.dff (neu_coef=0.7).
    corrected = fluorescence - 0.7 * neuropil
    corrected += 0.7 * neuropil.mean(axis=1, keepdims=True)

    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, 300, axis=1)
    baseline = maximum_filter1d(baseline, 300, axis=1)
    dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1).astype(np.float32, copy=False)
    events = dcnv.oasis(dff, 2000, 0.7, rate_hz)
    return dff, events
```
```python
        for start, stop in zip(starts, stops):
            fluorescence = np.concatenate([
                np.asarray(f_data[start:stop, :], dtype=np.float32)[:, local_keep].T
                for f_data, _, local_keep in plane_specs
            ], axis=0)
            ...
            dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
            if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(events)):
                raise ValueError("Non-finite neural activity after calcium processing")
            events_by_trial.append(events)
```

iii. The AI read `preprocessing.py::dff`, `teleport_metadata.py`, the `dayData` parameter docstring (`baseline_method='maximin'`, `ts_key='events'`, `int_thresh=0.5`), the suite2p `dcnv.oasis` signature, and the Methods paragraph, and summarised: "I'll ... apply the paper's trial-wise neuropil correction, maximin ΔF/F, two-sample Gaussian smoothing, OASIS deconvolution". The tau, neuropil coefficient and window sizes are the paper's own. Doing the pipeline one trial at a time (rather than on a NaN-masked session array as the paper does) is the AI's memory strategy and is mathematically the same operation on the same windows.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Only ROIs with `iscell == 1` (Suite2p's manual curation) are read at all. (2) Putative interneurons are then removed: the Pearson correlation between each cell's dF/F and the animal's running speed is accumulated over **all** of the session's on-track samples (including the faulty-lick trials), and any cell with r > 0.5 is dropped. The correlation is computed with streaming sums so the full session dF/F never has to be held in memory. Across the release this removes 138,276 retained cells from the curated set; the code errors out if a session has no curated cells or if the screen removes every cell.

ii.
```python
            trial_speed = speed[start:stop].astype(np.float64, copy=False)
            dff64 = dff.astype(np.float64, copy=False)
            sum_x += dff64.sum(axis=1)
            sum_x2 += np.square(dff64).sum(axis=1)
            sum_xy += dff64 @ trial_speed
            sum_y += trial_speed.sum()
            sum_y2 += np.square(trial_speed).sum()
            sample_count += len(trial_speed)

    numerator = sample_count * sum_xy - sum_x * sum_y
    denominator = np.sqrt(
        (sample_count * sum_x2 - np.square(sum_x))
        * (sample_count * sum_y2 - sum_y * sum_y)
    )
    speed_correlation = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    putative_interneuron = speed_correlation > 0.5
    keep_neuron = ~putative_interneuron
```

iii. The AI found the threshold in the paper's `dayData` attributes (`exclude_int: True`, `int_thresh: 0.5`, `int_method: 'speed'`) and in `spatial.py::is_putative_interneuron`, and stated it would apply ">0.5 speed-correlation interneuron exclusion". The comment in the code explains why the screen spans the whole session rather than only retained trials: "Process all trials for the paper's session-wide speed-correlation interneuron screen, including trials whose lick sensor was faulty."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No extra alignment step is needed. Because a trial is defined as the frame window starting at the `trial_start` sample, every emitted neural matrix already begins at the alignment event. This is recorded in the metadata as `temporal_alignment_event = 'start of trial (entry into the 450 cm corridor)'`, `off_start = 0.0`, `off_end = None` (trial length is variable, ending at the teleport).

ii.
```python
            "temporal_alignment_event": "start of trial (entry into the 450 cm corridor)",
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "teleport onset; trial end is variable relative to trial start",
```
```python
        for start, stop in zip(starts, stops):
            fluorescence = np.concatenate([
                np.asarray(f_data[start:stop, :], dtype=np.float32)[:, local_keep].T
```

iii. Implicit in the trial definition. The AI recorded `off_start = 0.0` rather than `None` to make explicit that the alignment event coincides with the first sample of every trial, and added `trial_end_event` to document that the trial end is not at a fixed offset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 64.4836 ms per bin (15.5078125 Hz), the native per-plane imaging rate. No rebinning, resampling or interpolation is applied anywhere. The rate is derived from the median spacing of the behaviour timestamps rather than from the `rate` attribute on the ophys series, because on the two-plane sessions (m17, m18) that attribute is the scanner rate, 31.0156 Hz. The code hard-asserts the derived rate against a nominal constant, so any session with a different sampling rate would abort rather than be silently mixed in.

ii.
```python
NOMINAL_RATE_HZ = 15.5078125

def _sampling_rate(behavior) -> float:
    """Use aligned timestamps, which correctly account for two-plane sessions."""
    timestamps = np.asarray(behavior["position/timestamps"])
    rate = float(1.0 / np.median(np.diff(timestamps)))
    if not np.isclose(rate, NOMINAL_RATE_HZ, rtol=0, atol=1e-4):
        raise ValueError(f"Unexpected aligned sampling rate: {rate} Hz")
    return rate
```
```python
            "time_bin_size": 1000.0 / NOMINAL_RATE_HZ,
```

iii. The AI explicitly checked this: it printed `imaging_rate`, the ophys `rate` attribute, `planeIdx` counts and the behaviour timestamp spacing for m3 (1 plane) and m17/m18 (2 planes), found the timestamp spacing is 0.06448363 s in all three, and concluded: "For the two-plane mice, the NWB timestamp spacing confirms the effective bin remains 64.48 ms despite a misleading 31 Hz imaging-plane attribute." The same per-plane rate is passed to OASIS, so the deconvolution kernel is also correct for those sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the frame index within the trial and the sampling rate derived from `position/timestamps`. The timestamps array is read and used to compute the rate; the per-sample times are then regenerated as `index / rate` rather than by subtracting the first timestamp.

ii.
```python
        timestamps = np.asarray(behavior["position/timestamps"])
        ...
        rate_hz = _sampling_rate(behavior)
...
            np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
```

iii. The AI verified the behaviour timestamps are exactly uniformly spaced (0.06448363 s) in the sessions it inspected, so index/rate and `t - t[0]` are the same quantity; regenerating from the rate keeps the input dtype and construction uniform across trials and avoids any accumulated float offset from the session's absolute clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. None beyond the division: the first sample of every trial is 0.0 s and each subsequent sample increments by 1/15.5078 s ≈ 0.0645 s. It is stored as a float32 row of the `(4, n_timepoints)` input matrix. No smoothing, clipping or normalisation.

ii.
```python
        input_trial = np.vstack([
            np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
            np.full(timepoints, trial_env, dtype=np.float32),
            np.full(timepoints, source_trial_number, dtype=np.float32),
            np.full(timepoints, previous_outcome, dtype=np.float32),
        ])
```

iii. Straightforward reading of the instruction "Time from start of trial in seconds (continuous, time-varying)". The verifier output confirms the range is [0.0, 99.6] s for the smoke session, consistent with the longest laps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the input matrix is built with `timepoints = stop - start`, the same frame window used to slice the fluorescence, so row 0 is element-wise aligned with the neural columns. Neural and behaviour streams in the NWB file share one frame index, so no interpolation or offset correction is needed. The code does not explicitly compare the neural frame count with the behaviour timestamp count (they differ by one frame in 10 of the two-plane sessions, where the ophys array is one frame *longer*; since all trial indices come from the behaviour series, the extra trailing frame is simply never read).

ii.
```python
        timepoints = stop - start
        input_trial = np.vstack([
            np.arange(timepoints, dtype=np.float32) / np.float32(rate_hz),
            ...
        ])
```

iii. The AI's `_sampling_rate` check ("Use aligned timestamps, which correctly account for two-plane sessions") shows it treated the behaviour timestamps as the already-aligned common clock for both streams, which is how the NWB release is packaged.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behaviour time series, which is 0 in ENV 1, 1 in ENV 2 and −1 during the teleport/ITI period.

ii.
```python
        environment = _read_behavior(behavior, "environment")
...
        trial_env = int(np.rint(np.median(environment[start:stop])))
```

iii. The AI printed the unique values of `environment` per trial alongside the NWB `identifier` strings (`Env1_...`, `Env2_...`, and `Env1_B_to_Env2_C` on day 8), confirming the 0/1 coding and that the value is constant within a lap. The input is named `"environment (ENV1=0, ENV2=1)"`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of the series over the trial's frames is taken, rounded to the nearest integer, and broadcast as a constant across all timepoints of the trial. Using the median over the on-track window excludes the −1 teleport code (which never falls inside a trial anyway) and is robust to any single-sample glitch.

ii.
```python
        trial_env = int(np.rint(np.median(environment[start:stop])))
        input_trial = np.vstack([
            ...,
            np.full(timepoints, trial_env, dtype=np.float32),
            ...
        ])
```

iii. The instruction specifies environment as "binary, ENV1 vs ENV2, per trial"; taking a per-trial summary rather than the raw per-sample trace enforces that the value really is per-trial, and the median makes the summary robust.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The `trial number` behaviour time series (summarised per trial by its median), not a running loop counter. This means the value stored is the trial's index in the **original** session, so the numbering is unchanged by the removal of faulty-lick trials.

ii.
```python
        trial_number = _read_behavior(behavior, "trial number")
...
        source_trial_number = float(np.median(trial_number[start:stop]))
```

iii. The AI inspected `trial number` early on (it appears in its per-session survey of unique values and in the raw sample dumps around each trial boundary) and found it constant within each start→teleport window and consistent with the trial ordering; it also records `retained_source_trials` per session in the metadata so the mapping back to source trials is recoverable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The per-trial median is taken and broadcast as a float32 constant across the trial's timepoints. No re-indexing, normalisation or scaling; the value runs 0…79 for a typical session.

ii.
```python
            np.full(timepoints, source_trial_number, dtype=np.float32),
```

iii. The instruction asks for "Trial number (continuous, per trial)"; the AI kept the raw within-session index. The verifier reported `trial number: [0.0, 79.0]` for the smoke session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behaviour time series' **timestamps** (reward delivery events have their own event clock, separate from the frame clock), combined with the frame timestamps of the trial boundaries. A per-trial binary outcome vector is built once for the whole session and then shifted by one trial.

ii.
```python
def _trial_outcomes(reward_timestamps: np.ndarray, timestamps: np.ndarray,
                    starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
        right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
        outcomes[trial] = right > left
    return outcomes
...
        outcomes = _trial_outcomes(
            np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
        )
```

iii. The AI inspected the `Reward` series and counted reward events falling inside trial time windows during its early exploration (it printed `'rew events?'` per trial for several sessions). Working in the time domain with `searchsorted` avoids having to snap reward events onto frame indices at all.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the outcome of the immediately preceding **source** trial is used (so the previous trial is the true previous lap even if that lap was later dropped as faulty-lick). The first trial of each session gets 0. The value is broadcast as a constant across the trial's timepoints, coded 0 = omitted, 1 = rewarded.

ii.
```python
        previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
...
            np.full(timepoints, previous_outcome, dtype=np.float32),
```

iii. Matches the instruction "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)". Indexing into the full source-trial outcome array (rather than into the retained-trial list) is what keeps the "previous lap" semantics correct across exclusions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The `position` behaviour series together with the per-trial reward-zone identity, which is itself inferred from the `reward_zone` behaviour series and `position` (see 10-b). The zone geometry is hard-coded from the paper: A = 80–130 cm, B = 200–250 cm, C = 320–370 cm.

ii.
```python
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_ENDS = ZONE_STARTS + 50.0
ZONE_CENTERS = (ZONE_STARTS + ZONE_ENDS) / 2.0
...
        position = _read_behavior(behavior, "position")
        rzone = _read_behavior(behavior, "reward_zone")
...
        zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
```

iii. The AI took the zone boundaries straight from the Methods: "The reward zone was a 'hidden', unmarked 50 cm span at one of three possible locations along the track ... zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm." It also inspected `code/src/reward_relative/behavior.py::get_reward_zones` and found that the `reward_zone` channel in the NWB release is an integrated event/count channel (values 0–6) rather than a zone label, so position had to be combined with it.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm to the nearest edge of the active zone: negative before the zone (`position − zone_start`), exactly 0 anywhere inside the zone, positive after the zone (`position − zone_end`). Computed per sample with `np.where`, then discretised.

ii.
```python
def _distance_class(position: np.ndarray, zone: int) -> np.ndarray:
    start, end = ZONE_STARTS[zone], ZONE_ENDS[zone]
    distance = np.where(position < start, position - start,
                        np.where(position > end, position - end, 0.0))
```

iii. This is the "distance to any location in the reward zone" the instructions ask for, and it matches the paper's reward-relative distance convention (distance 0 throughout the 50 cm zone).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the seven classes given in the instructions, using `np.select` with explicit, ordered predicates: `< −50 → 0`, `< −10 → 1`, `< 0 → 2`, `== 0 → 3`, `≤ 10 → 4`, `≤ 50 → 5`, else 6. Because `np.select` takes the first matching predicate, the boundary values fall as: −50 → class 1, −10 → class 2, exactly 0 → class 3 (in-zone), +10 → class 4, +50 → class 5. Stored as int8.

ii.
```python
    return np.select(
        [distance < -50, distance < -10, distance < 0, distance == 0,
         distance <= 10, distance <= 50],
        [0, 1, 2, 3, 4, 5], default=6,
    ).astype(np.int8)
```
```python
            ["< -50 cm", "-50 to -10 cm", "-10 to <0 cm", "in reward zone (0 cm)",
             ">0 to +10 cm", "+10 to +50 cm", "> +50 cm"],
```

iii. A direct transcription of the instruction's bin table, with the `distance == 0` predicate placed before the positive bins so that the entire reward zone (not just the single point where position equals a zone edge) is class 3. The verifier confirmed all seven classes are populated (0.225 / 0.061 / 0.103 / 0.424 / 0.045 / 0.078 / 0.064 on the smoke session).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[start:stop]`, the identical frame window used for the neural slice, so it is element-wise aligned with no offset or interpolation.

ii.
```python
        output_trial = _output_matrix(
            position[start:stop], speed[start:stop], lick[start:stop],
            int(zones[trial]), int(outcomes[trial]),
        )
```

iii. Same rationale as 3-c: neural and behaviour share one frame index in the NWB release.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (VR corridor position in cm), read once per session and sliced per trial.

ii.
```python
        position = _read_behavior(behavior, "position")
...
            position[start:stop], speed[start:stop], lick[start:stop],
```

iii. Direct; the AI verified the range by printing per-trial position traces and confirmed the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretisation — the raw cm values are binned directly, with no unwrapping, clipping or interpolation.

ii.
```python
    absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
```

iii. The instruction asks for the absolute position in the corridor discretised into 5 equal bins; the raw series is already in cm from the start of the track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with interior edges `[90, 180, 270, 360]`, i.e. five 90 cm bins spanning the 450 cm track. The first and last bins are open-ended, so the handful of samples marginally outside [0, 450] land in classes 0 and 4 rather than creating extra classes. Stored as int8.

ii.
```python
    absolute_position = np.digitize(position, [90, 180, 270, 360]).astype(np.int8)
...
            ["<90 cm", "90-180 cm", "180-270 cm", "270-360 cm", ">360 cm"],
```

iii. A direct transcription of the instruction's bin table (450 / 5 = 90 cm). The verifier confirmed all five classes are populated.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame window as the neural slice — `position[start:stop]` — so element-wise aligned with no further processing.

ii.
```python
        output_trial = _output_matrix(
            position[start:stop], speed[start:stop], lick[start:stop], ...)
```

iii. As in 7-d.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series, which stores a per-frame cumulative lick count (integer ≥ 0) from the capacitive sensor.

ii.
```python
        lick = _read_behavior(behavior, "lick")
```

iii. The AI surveyed the unique values of `lick` and used counts > 2 as the faulty-sensor criterion (1-e), showing it understood the series as a per-frame count rather than a binary flag.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised: any frame with a count greater than 0 becomes 1, otherwise 0. Stored as int8. Trials whose lick sensor was faulty are excluded from the dataset entirely (1-e) rather than being binarised from corrupt counts.

ii.
```python
    lick_binary = (lick > 0).astype(np.int8)
```
```python
            ["no lick", "lick"],
```

iii. The instruction specifies "Lick, time-varying. 0 = no, 1 = yes"; the paper likewise converts "Remaining lick counts ... to a binary vector". The trajectory ties the two together: faulty trials "cannot supply a valid lick target".

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame window, `lick[start:stop]`, element-wise aligned with the neural columns; no shifting or smoothing.

ii.
```python
        output_trial = _output_matrix(
            position[start:stop], speed[start:stop], lick[start:stop], ...)
```

iii. As in 7-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The `reward_zone` behaviour series (a non-zero event/count channel that fires only while the animal is inside the active zone), the `position` series, and the session's `experiment_day` from `general/session_id`.

ii.
```python
        rzone = _read_behavior(behavior, "reward_zone")
        experiment_day = int(_decode_scalar(nwb["general/session_id"][()]))
...
        zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
```

iii. The AI established from `behavior.py` and from direct inspection that `reward_zone` takes values 0–6 and is not a zone label, so the zone identity has to be recovered from where in the corridor the channel fires.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Two stages. Per trial, the median position over the samples where `reward_zone > 0` is taken and snapped to the nearest of the three zone centres (105, 225, 345 cm); trials with no such samples (reward omissions / no licking) are marked as unobserved. The session is then split into condition blocks — one block normally, or two blocks split after trial 30 on the switch days {3, 5, 7, 8, 10, 12, 14} — and every trial in a block is assigned the **modal** observed zone of that block. This both denoises the per-trial estimate and fills in the unobserved trials. The label is emitted as a constant int8 row (0 = A, 1 = B, 2 = C) across the trial's timepoints, and the whole per-session label vector is also recorded in `metadata['session_info']`.

ii.
```python
def _reward_zone_labels(position, rzone, starts, stops, experiment_day) -> np.ndarray:
    """Recover the active zone robustly from the aligned rzone event stream.
    ...
    """
    observed = np.full(len(starts), -1, dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        mask = rzone[start:stop] > 0
        if np.any(mask):
            event_position = float(np.median(position[start:stop][mask]))
            observed[trial] = int(np.argmin(np.abs(ZONE_CENTERS - event_position)))

    boundaries = [0, min(30, len(starts)), len(starts)] if experiment_day in SWITCH_DAYS else [0, len(starts)]
    labels = np.empty(len(starts), dtype=np.int8)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        valid = observed[left:right]
        valid = valid[valid >= 0]
        if len(valid) == 0:
            raise ValueError(f"Cannot infer reward zone for trials {left}:{right}")
        labels[left:right] = np.bincount(valid, minlength=3).argmax()
    return labels
```
```python
        np.full(timepoints, zone, dtype=np.int8),
```

iii. The switch-day set and the trial-30 boundary come from the Methods: "On day 3 (switch one), the zone was moved after 30 trials ... On day 5 (switch two) ... moved back ... on day 7. On day 8, the reward zone switch coincided with a switch into the novel environment, where the sequence of zone switches was then reversed on the same day-to-day schedule for a total of 14 days ... Each switch occurred after 30 trials." Critically, the AI then validated the inference against ground truth: it ran `_reward_zone_labels` over all 152 sessions and printed the resulting pre-/post-switch labels next to the NWB `identifier` string of each file (`Env1_LocationB_to_A`, `Env1_LocationC`, `Env2_B_to_Env1_A`, …), reporting: "Reward-zone inference was also cross-checked against every NWB session identifier and exactly matches all A/B/C and switch conditions."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` series' timestamps, via the same per-trial outcome vector used for the previous-trial input (6-a).

ii.
```python
        outcomes = _trial_outcomes(
            np.asarray(behavior["Reward/timestamps"]), timestamps, starts, stops
        )
```

iii. As in 6-a: reward delivery is an event stream with its own timestamps, so it is intersected with the trial's time window rather than read off a per-frame channel.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is coded 1 if at least one reward timestamp falls in `[t(trial_start), t(teleport))`, else 0, using two `searchsorted` calls on the sorted reward timestamps. The per-trial scalar is broadcast as a constant int8 row across the trial's timepoints, and the full per-session vector is also stored in `metadata['session_info']['reward_outcome_by_source_trial']`.

ii.
```python
        left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
        right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
        outcomes[trial] = right > left
...
        np.full(timepoints, rewarded, dtype=np.int8),
```

iii. Matches the instruction "Reward outcome, per-trial. 0 = no, 1 = yes". The resulting omission rate (~10–20% per session in the verifier output) is consistent with the paper's "reward was randomly omitted on approximately 15% of trials".

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter is deliberately fail-fast rather than repair-and-continue: it raises on unmatched trial starts/teleports, on a teleport that does not follow its start, on an unexpected sampling rate, on a session with no curated cells, on non-finite dF/F or events, on an interneuron screen that removes every cell, and on a condition block in which no trial has any reward-zone evidence. The genuinely missing data it does handle silently is (a) reward-omission trials where the `reward_zone` channel never fires — filled by the block mode in `_reward_zone_labels` — and (b) faulty lick-sensor trials, which are excluded (1-e). It does **not** explicitly reconcile the neural frame count with the behaviour timestamp count; in the ten two-plane sessions where the ophys array has one more frame than the behaviour clock, the extra trailing frame is simply never indexed, because every window comes from the behaviour series. Writing the pickle is atomic (write to `.tmp`, then `os.replace`) so a crashed run cannot leave a truncated dataset behind.

ii.
```python
    if len(starts) != len(stops):
        raise ValueError(f"Unmatched trial starts ({len(starts)}) and teleports ({len(stops)})")
    if np.any(stops <= starts):
        raise ValueError("A teleport did not follow its corresponding trial start")
...
    if not np.isclose(rate, NOMINAL_RATE_HZ, rtol=0, atol=1e-4):
        raise ValueError(f"Unexpected aligned sampling rate: {rate} Hz")
...
        if len(roi_indices) == 0:
            raise ValueError("Session has no manually curated cells")
...
            if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(events)):
                raise ValueError("Non-finite neural activity after calcium processing")
...
    if not np.any(keep_neuron):
        raise ValueError("Interneuron screen removed every cell")
...
        if len(valid) == 0:
            raise ValueError(f"Cannot infer reward zone for trials {left}:{right}")
```
```python
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)
```

iii. The AI's stated position is that a silent repair could corrupt the dataset without anyone noticing, whereas a hard failure surfaces the problem — and the failure mode did in fact surface the real m17/m18 two-plane layout issue mid-run ("The run reached 110 sessions, then exposed a two-plane NWB layout difference in m17/m18 ... No final file was overwritten"), which it then fixed properly rather than working around. The AI also ran the provided verifier on the complete output: "The complete 8.9 GB dataset now passes full-format verification with no errors or warnings: 152 sessions, 12,135 valid trials, 11 mice, 138,276 session-level CA1 neuron instances."

## 13-a. What are the most time-consuming steps of the code?

i. In order: (1) the calcium pipeline, dominated by the OASIS deconvolution and the maximin filtering, run once per trial per plane over ~900 cells × ~350 frames — this is repeated for all 12,216 source trials; (2) reading the `Fluorescence` and `Neuropil` matrices out of the NWB files (~9 GB of float32 per pass across the release), done as 12,216 × n_planes contiguous row-range reads; (3) the final `pickle.dump` of the ~9.5 GB dataset, plus the equivalent `os.replace`. The per-trial behaviour work (`np.digitize`, `np.select`, `searchsorted`) is negligible by comparison. Judging by the file timestamps of the final run, the whole 152-session conversion completed in a few minutes on this 128-core machine, so none of these steps is pathologically slow.

ii.
```python
            dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
```
```python
            fluorescence = np.concatenate([
                np.asarray(f_data[start:stop, :], dtype=np.float32)[:, local_keep].T
                for f_data, _, local_keep in plane_specs
            ], axis=0)
```
```python
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI anticipated the output size before running ("its expected size is roughly 10–20 GB because it retains frame-level activity from all curated cells") and monitored progress by session ("The full run is progressing normally (28/152 sessions)"), indicating it treated the per-session calcium processing and the final write as the cost centres. It also judged the decoder-side cost explicitly: "the provided default trainer's 200 full-data epochs are disproportionately expensive for an 8.9 GB frame-level dataset, so I'll verify the same training path on a representative subset."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Four `for trial in zip(starts, stops)` loops remain: `_trial_outcomes`, `_bad_lick_trials`, `_reward_zone_labels`, and the two main per-trial loops in `convert_session`. Of these:
- `_trial_outcomes` could be a single vectorised `np.searchsorted(reward_timestamps, timestamps[starts])` / `...[stops]` pair with no Python loop at all.
- `_bad_lick_trials` and the `observed` loop in `_reward_zone_labels` could be done with `np.add.reduceat` / segment reductions over the concatenated frame axis.
- The calcium loop could have been run once per session on a NaN-masked array (as the paper's own `dff` does) instead of once per trial, replacing ~12,216 small OASIS/filter calls with 152; that would trade the loop for a larger peak memory footprint, which is why the AI chose per-trial.
The per-trial output-assembly loop is intrinsic: trials have different lengths and must be emitted as separate arrays, so it cannot be vectorised without padding.

ii.
```python
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        left = np.searchsorted(reward_timestamps, timestamps[start], side="left")
        right = np.searchsorted(reward_timestamps, timestamps[stop], side="left")
        outcomes[trial] = right > left
```
```python
    return np.asarray([
        np.mean(lick[start:stop] > 2) > 0.30
        for start, stop in zip(starts, stops)
    ], dtype=bool)
```

iii. These loops run 80 times per session over small slices, so their cost is negligible next to the deconvolution and I/O; the AI prioritised legibility and bounded memory. The one loop where vectorisation would have mattered — the calcium pipeline — was kept per-trial on purpose: "Trial-wise processing is keeping memory bounded during conversion."

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each behaviour series is read exactly once; there is no separate survey pass. The repetition that does exist is: the trial-boundary `zip(starts, stops)` iteration is walked five separate times (outcomes, zone labels, bad-lick, calcium, output assembly) instead of once; and `speed[start:stop]` is sliced in both the calcium loop (for the interneuron screen) and the output loop (for the speed classes). The `rois` DynamicTableRegion and `is_cell` lookup are resolved once per session and cached in `plane_specs`, so the ROI subsetting is not repeated per trial.

ii.
```python
        outcomes = _trial_outcomes(...)
        zones = _reward_zone_labels(position, rzone, starts, stops, experiment_day)
        bad_lick = _bad_lick_trials(lick, starts, stops)
        ...
        for start, stop in zip(starts, stops):      # calcium + speed correlation
        ...
    for trial, (start, stop) in enumerate(zip(starts, stops)):   # input/output assembly
```

iii. The AI's design keeps a single I/O pass per file and streams the speed-correlation statistics so that the session's full dF/F never has to be materialised or re-read; the repeated cheap iterations over trial boundaries are a readability choice, each one being a self-contained, separately testable function.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three things:
- The full calcium pipeline (dF/F **and** OASIS deconvolution) is run for the 81 faulty-lick trials whose events are then thrown away. Only the dF/F of those trials is actually needed, for the session-wide interneuron screen; the deconvolution of them is wasted.
- The deconvolved events are computed and stored for **all** curated cells, then the putative interneurons (a few percent) are dropped afterwards; those cells' OASIS output is discarded.
- Minor: `n_planes` is computed only to be written into metadata, `roi_indices` is used only for its length, and the large per-session `session_info` blocks (`retained_source_trials`, `reward_zone_by_source_trial`, `reward_outcome_by_source_trial`) are provenance records the decoder never reads.
Against that, nothing large is computed and discarded: dF/F is consumed by the interneuron screen, and the events are the dataset itself.

ii.
```python
        # Process all trials for the paper's session-wide speed-correlation
        # interneuron screen, including trials whose lick sensor was faulty.
        for start, stop in zip(starts, stops):
            ...
            dff, events = _dff_and_events(fluorescence, neuropil, rate_hz)
            events_by_trial.append(events)
```
```python
    putative_interneuron = speed_correlation > 0.5
    keep_neuron = ~putative_interneuron
...
        neural_trials.append(np.ascontiguousarray(events_by_trial[trial][keep_neuron]))
```

iii. Both of the significant cases are consequences of a correctness decision the AI made deliberately: the interneuron screen must see the whole session (so every trial has to be processed) and the screen's verdict is only known after the last trial (so the events must already exist when the cells are dropped). The alternative — two passes over the fluorescence — would double the dominant I/O cost to save a few percent of compute.
