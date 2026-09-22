# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter globs every NWB file under the data directory with `sub-*/*_behavior+ophys.nwb` and sorts them by (mouse number, session number) parsed from the filename. This finds all 152 files (11 subject directories, 12–14 sessions each). Files are opened directly with `h5py` rather than `pynwb`, reading only the HDF5 paths that are needed: `processing/behavior/BehavioralTimeSeries/*` (position, speed, lick, environment, `trial number`, `trial_start`, `teleport`, `Reward`, plus the `position` and `Reward` timestamps), `processing/ophys/ImageSegmentation/PlaneSegmentation` (`iscell`, `planeIdx`) and `processing/ophys/Fluorescence|Neuropil/plane{k}/data`. Fluorescence/neuropil are read lazily, one trial slice and only the curated ROI columns at a time, so the full session-wide movie is never materialised. The loader errors out rather than silently skipping if a file is malformed (bad start/teleport pairing, unexpected sampling interval, non-contiguous plane labels, a session with fewer than two usable trials).

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse subject/session from {path}")
    return int(match.group(1)), int(match.group(2))

def convert(data_dir: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No NWB files found below {data_dir}")
```
```python
    with h5py.File(path, "r") as nwb:
        behavior = nwb["processing/behavior/BehavioralTimeSeries"]
        position = read_behavior(behavior, "position").astype(np.float64, copy=False)
        ...
        segmentation = nwb["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        fluorescence = nwb["processing/ophys/Fluorescence"]
        neuropil = nwb["processing/ophys/Neuropil"]
```
```python
        f_parts.append(
            np.asarray(
                fluorescence[f"plane{plane}"]["data"][start:stop, local_indices],
                dtype=np.float32,
            ).T
        )
```

iii. From the trajectory (steps 7–19) the agent first enumerated the data tree and the NWB internal layout, confirming 152 files, 11 `sub-` directories and the field names/units of every behavioral series and ophys group. It chose `h5py` over `pynwb` explicitly to be able to do partial, per-trial, per-column reads ("Reading only manually curated cells substantially reduces both I/O and memory"), which is what makes a single streaming pass over 152 large sessions feasible. Nothing is subsampled: all 152 sessions and all 12,216 source laps enter the pipeline.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from the file path (`sub-m<number>`). The unique mouse numbers are sorted numerically and rendered as `m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19` (11 subjects). Each session's `subject_idx` is a lookup of its parsed subject into that list.

ii.
```python
    subject_names = [f"m{mouse}" for mouse in sorted({natural_key(path)[0] for path in paths})]
    subject_lookup = {subject: index for index, subject in enumerate(subject_names)}
    ...
        subject_idx.append(subject_lookup[info["subject"]])
    ...
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
```

iii. The agent's survey (step 10) noted the release covers "11 switching-condition mice", matching the paper's n = 11 switch cohort, and the directory names are the canonical DANDI `sub-<id>` pattern. Sorting numerically (rather than lexicographically) keeps `m3, m4, m7, m11, …` in experiment order.

## 1-c. How are the data split into sessions?

i. One session per NWB file; the session number comes from the `ses-<NN>` token in the filename. All 152 files become sessions (m11 contributes 12, all other mice 14), ordered by (mouse, session). No merging or alignment of cells across days is attempted — each session's neurons are independent.

ii.
```python
    paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)
    ...
    for path in paths:
        neural_session, input_session, output_session, info = convert_session(path)
        if len(neural_session) < 2:
            raise ValueError(f"Fewer than two usable trials in {path}")
```
```python
    info = {
        "source_file": str(path.relative_to(path.parents[2])),
        "subject": f"m{mouse_number}",
        "session": session_number,
        ...
    }
```

iii. The filename pattern `sub-mX_ses-NN_behavior+ophys.nwb` makes the one-file-per-imaging-day mapping unambiguous; the agent reported "all 152 NWB sessions" (step 43) and recorded a per-session `session_info` block in the metadata so the provenance of each session is recoverable.

## 1-d. How are the data split into trials?

i. Trials are delimited by the two binary behavioral markers. `trial_start` pulses give the trial onsets, `teleport` pulses give the trial ends. Each trial is the half-open sample range `[start, stop)`: the trial-start sample is included (it is t = 0 of the alignment) and the teleport sample is excluded, so a trial spans only the 450 cm virtual corridor and never the inter-trial teleport/jitter period. The code asserts that the number of starts equals the number of teleports and that every teleport follows its start.

ii.
```python
        starts = np.flatnonzero(read_behavior(behavior, "trial_start") > 0)
        stops = np.flatnonzero(read_behavior(behavior, "teleport") > 0)

        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid start/teleport pairing in {path}")
```
```python
        for trial, (start, stop) in enumerate(zip(starts, stops)):
            # The alignment marker is included as t=0; the teleport marker is
            # excluded, leaving only the 450 cm virtual corridor.
            trial_slice = slice(int(start), int(stop))
```

iii. Step 20: "Trials will span the trial-start marker through the sample before teleport." The agent had verified in step 15 that `trial_start` and `teleport` are single-sample 0/1 pulses with equal counts (80/80 in the sessions it inspected), and that `position` runs from ~0 to ~450 cm between them and drops to −500/−50 during the teleport. It additionally cross-checks each trial index against the stored `trial number` series (see 5-a), which fails loudly if the segmentation drifts.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level quality filter is applied, taken verbatim from the paper's licking Methods: a trial is discarded if more than 30% of its frames carry a cumulative lick count greater than 2 (a sustained ≥20 Hz "lick rate" that indicates a damaged capacitive lick circuit). This removed exactly 81 of 12,216 laps (0.66%), leaving 12,135 trials. No minimum-trial-length filter, no running-speed filter and no session-level filter is applied (other than an error if a session ends up with <2 trials, which never triggers). The check is done before any neural data is read, so corrupt trials cost no I/O.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
            trial_lick = lick[trial_slice]
            if np.mean(trial_lick > 2) > LICK_ERROR_FRACTION:
                dropped_lick_trials.append(trial)
                continue
```
```python
            "trial_filtering": (
                "Trials with lick count >2 in more than 30% of corridor frames were "
                "dropped because the paper identifies these as lick-circuit errors. No "
                "running-speed filter was applied because stopped/slow frames are a "
                "requested speed class."
            ),
```

iii. The paper states: "A very small number of trials with erroneous lick detection from damage to the circuit were removed … (~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice). These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2." The agent implemented that rule literally and observed in step 46 that "the lick-quality rule removed exactly 81 trials — the same count reported in the paper," which it used as a validation that its trial segmentation matched the authors'. It explicitly declined the paper's other common restriction (speed > 2 cm/s) because "stopped frames … speed <2 cm/s is an explicit decoder class" (step 20).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw Suite2P traces: `processing/ophys/Fluorescence/plane{k}/data` (F) and `processing/ophys/Neuropil/plane{k}/data` (Fneu), restricted to the ROIs flagged by the manually curated `iscell` mask in `ImageSegmentation/PlaneSegmentation`. The stored `processing/ophys/Deconvolved` series is deliberately *not* used. For the two-plane mice (m17, m18) the ROI table is pooled across `plane0` and `plane1`, and the code maps global ROI rows back to per-plane column indices before reading.

ii.
```python
        iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
        plane_index = np.asarray(segmentation["planeIdx"], dtype=np.int64)
        plane_values = np.unique(plane_index)
        cell_indices_by_plane = [
            np.flatnonzero(iscell[plane_index == plane]) for plane in plane_values
        ]
        fluorescence = nwb["processing/ophys/Fluorescence"]
        neuropil = nwb["processing/ophys/Neuropil"]
```
```python
    f = np.concatenate(f_parts, axis=0)
    fneu = np.concatenate(fneu_parts, axis=0)
```

iii. Module docstring and step 20: "their 'Deconvolved' array is the raw Suite2P-scale trace — not the paper's trial-wise dF/F pipeline. I'll therefore reconstruct the paper's neural signal from curated `iscell` ROIs." The agent had inspected the NWB groups (steps 11–16) and read `code/src/reward_relative/preprocessing.py::dff` to confirm the authors compute their own signal from F and Fneu. The m17/m18 multi-plane layout was discovered as a bug at step 30 ("its pooled ROI table indexes across separate plane-specific fluorescence datasets") and fixed by concatenating curated cells from both planes in ROI-table order.

## 2-b. How is the `neural` data processed?

i. A reimplementation of the paper's `preprocessing.dff`, applied independently to each trial's sample range: subtract `0.7 × Fneu`; add each trial's mean neuropil back (`+0.7 × mean(Fneu)`) so the ratio is a true dF/F and the denominator is not near zero; compute a maximin baseline (Gaussian smoothing with σ = 15 samples, then a 300-sample minimum filter followed by a 300-sample maximum filter — the Methods' ~20 s window at 15.5 Hz); form `(F_corrected − baseline) / |baseline|`; smooth the result with a σ = 2-sample Gaussian. Cells from multiple planes are pooled. **The pipeline stops there: no OASIS deconvolution is performed**, so the `neural` matrices are smoothed dF/F, stored as float32 (total 9.52 GB). The baseline window is always restricted to the trial itself; the `keep_teleports` per-mouse/per-day behaviour in the paper's `teleport_metadata.py` is not reproduced.

ii.
```python
    corrected = f - NEUROPIL_COEFFICIENT * fneu
    corrected += NEUROPIL_COEFFICIENT * np.mean(fneu, axis=1, keepdims=True)
    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, 300, axis=1)
    baseline = maximum_filter1d(baseline, 300, axis=1)

    # A zero baseline is not expected for fluorescence.  The guard prevents an
    # invalid decoder file if a malformed ROI nevertheless reaches this point.
    denominator = np.abs(baseline)
    denominator[denominator == 0] = np.finfo(np.float32).eps
    dff = (corrected - baseline) / denominator
    dff = gaussian_filter1d(dff, 2, axis=1)
    return np.asarray(dff, dtype=np.float32)
```
```python
            "neural_signal": "trial-wise neuropil-corrected, maximin-baselined, Gaussian-smoothed dF/F",
```

iii. Step 20: "0.7 neuropil subtraction, per-trial 20 s maximin baseline, ΔF/F, and the reported two-frame Gaussian smoothing." The agent read the authors' `dff()` (which uses `neu_coef=0.7`, `baseline_method='maximin'`, σ = 15 smoothing, 300-sample min/max filters and a 2-sample dF/F smooth), `nansmooth`, and the Methods paragraph, and reproduced each of those steps. It gave **no justification for omitting the deconvolution step** — its framing throughout was that "the paper's relevant neural signal is trial-wise dF/F" (step 10), and the docstring only argues against the *stored* `Deconvolved` array, not against running OASIS on the computed dF/F. It likewise never discusses the teleport-period baseline exception even though it printed `teleport_metadata.py` at step 19.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Only ROIs with `iscell[:, 0] > 0` — Suite2P's manually curated cells — are read at all (138,678 cells across 152 sessions; 155–2,341 per session). (2) Putative interneurons are excluded post hoc: any cell whose trial-concatenated dF/F has a Pearson correlation with running speed above 0.5 is dropped (400 cells, 0.29% of the curated population), leaving 138,278 neurons. The correlation is computed with streaming sufficient statistics accumulated trial by trial so that a second session-wide dF/F matrix is never allocated.

ii.
```python
        iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
```
```python
        # Streaming sufficient statistics reproduce np.corrcoef(dff, speed)
        # without materialising a second session-wide dF/F matrix.
            sum_x += np.sum(dff, axis=1, dtype=np.float64)
            sum_x2 += np.sum(dff * dff, axis=1, dtype=np.float64)
            sum_xy += np.asarray(dff, dtype=np.float64) @ trial_speed
            sum_y += float(np.sum(trial_speed, dtype=np.float64))
            sum_y2 += float(np.sum(trial_speed * trial_speed, dtype=np.float64))
            n_samples += dff.shape[1]

    speed_correlations = correlations_from_sums(
        sum_x, sum_x2, sum_xy, sum_y, sum_y2, n_samples
    )
    keep_neurons = speed_correlations <= INTERNEURON_R_THRESHOLD
    neural_trials = [trial[keep_neurons] for trial in neural_trials]
```

iii. Step 20: "I'll also exclude the additional dF/F–speed-correlation >0.5 interneuron candidates," and step 24 reports the one-session dry run "reproduced the expected curated cell count (155 cells before the paper's extra interneuron screen)" — 155 is the lower end of the paper's stated 155–2,172 pyramidal cells per session. The Methods specify exactly this second filter: "Additional putative interneurons were detected for exclusion … by a Pearson correlation of >0.5 between their dF/F timeseries and the animal's running speed, excluding 0.42 ± 0.85% of cells." The agent's 0.29% is in that range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions ask for alignment to trial start, which the trial segmentation already provides: the first sample of every trial is the `trial_start` pulse, so sample 0 of every neural matrix is t = 0 relative to the alignment event. No re-slicing, padding or cropping around the event is done; trials keep their native, variable length (96 to 3,359 frames) and end at the sample before teleport. `off_start = 0.0` and `off_end = None` record this.

ii.
```python
            trial_slice = slice(int(start), int(stop))
            ...
            dff = paper_dff_trial(
                fluorescence, neuropil, cell_indices_by_plane, int(start), int(stop)
            )
```
```python
            "temporal_alignment_event": "entry into the virtual corridor (trial-start marker)",
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "teleport onset (excluded from each trial)",
```

iii. Step 43: the artifact "preserves native variable lap durations." The agent treated the trial-start marker itself as the alignment event and did not truncate to a common length, since the decoder consumes ragged per-trial matrices and truncating would discard the long tail of slow laps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling of any kind. Everything stays on the native acquisition grid: 15.5078125 Hz per plane, i.e. a 64.4836 ms bin, identical for all trials and all sessions (for the two-plane mice m17/m18 the 31 Hz scanner rate corresponds to the same 15.5 Hz per-plane rate, and the behavioral timestamps are already on the per-plane grid). The frame rate is a hard-coded constant, but the code validates it against the median behavioral inter-sample interval in every session and aborts on mismatch.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
        frame_intervals = np.diff(timestamps)
        if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, rtol=0, atol=1e-6):
            raise ValueError(f"Unexpected behavior sampling interval in {path}")
```
```python
            "time_bin_size": TIME_BIN_MS,
            "sampling_rate_hz": FRAME_RATE_HZ,
```

iii. Step 20: "I'll retain native synchronized 64.48 ms frames." The agent had measured the rate directly (step 15 output: "rate median 15.507812500000881") and noted that behavior and imaging are already on a common, uniform grid, so any rebinning would only lose resolution and risk introducing a behavior/neural offset.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attached to the `position` behavioral time series (all `BehavioralTimeSeries` members except `Reward` share this frame-aligned clock).

ii.
```python
        timestamps = np.asarray(behavior["position"]["timestamps"], dtype=np.float64)
```

iii. The agent inspected the timestamps of every behavioral series (steps 12–16) and confirmed they are a single, uniform, frame-locked clock, so any of them is interchangeable; `position` was used as the canonical one and the `Reward` series (which carries its own event timestamps) is the only one handled separately.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp of the trial's first sample, so every trial starts at exactly 0.0 s and increments by ~0.0645 s per bin. Stored as float32 in input row 0. Observed range across the dataset: 0 to 216.5 s.

ii.
```python
            time_from_start = timestamps[trial_slice] - timestamps[start]
            ...
            inputs = np.vstack(
                [
                    time_from_start,
                    ...
                ]
            ).astype(np.float32)
```

iii. Direct implementation of "Time from start of trial in seconds" with the trial-start marker as the origin; no other processing is needed because the clock is already in seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral samples are the same samples: both are indexed with the same `[start, stop)` range, the neural frames being the rows of the fluorescence array at those indices. No interpolation or offset correction is applied. The alignment assumption is validated indirectly by the sampling-interval check and, per-trial, by the `np.vstack` of behaviour-length and `dff.shape[1]`-length rows, which raises if the two streams ever disagree in length.

ii.
```python
        frame_intervals = np.diff(timestamps)
        if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, rtol=0, atol=1e-6):
            raise ValueError(f"Unexpected behavior sampling interval in {path}")
```
```python
            inputs = np.vstack(
                [
                    time_from_start,
                    np.full(dff.shape[1], env),
                    ...
                ]
            ).astype(np.float32)
```

iii. The NWB release stores behaviour already resampled onto the imaging frame clock (the agent verified identical lengths and a constant 64.48 ms interval), so the streams are aligned by construction and the offset is zero.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series, which takes the values 0 (ENV 1), 1 (ENV 2) and −1 (during the teleport period).

ii.
```python
        environment = read_behavior(behavior, "environment")
        ...
            env_values = environment[trial_slice]
```

iii. The agent surveyed the unique values of every behavioral series (step 12/17) and found `environment ∈ {−1, 0, 1}`, matching the paper's two-environment design with a sentinel for the inter-trial period; the input name in the output records the mapping (`"environment type (ENV1=0, ENV2=1)"`).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within a trial the label is constant, so the code takes the rounded median of the non-negative samples (ignoring any −1 sentinel), errors if a trial has no valid label, and broadcasts the scalar across all timepoints of the trial as input row 1. This correctly handles day-8 sessions where the environment changes between the pre- and post-switch blocks of the same session (11 of the 152 sessions contain both 0 and 1).

ii.
```python
            valid_env = env_values[env_values >= 0]
            if valid_env.size == 0:
                raise ValueError(f"No environment label in {path}, trial {trial}")
            env = float(np.rint(np.median(valid_env)))
            ...
                    np.full(dff.shape[1], env),
```

iii. The instruction lists environment as a per-trial binary variable; taking the median of the valid samples makes the per-trial value robust to any stray sentinel at a trial boundary, and the explicit `>= 0` mask encodes the agent's observation that −1 marks the non-environment (teleport) period.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The stored `trial number` behavioral time series, cross-validated against the sequential index of the trial-start marker. The rounded median of `trial number` within each trial is used as the value, and the code raises if it ever differs from the loop index, i.e. the two definitions must agree exactly (they do, for all 152 sessions).

ii.
```python
        trial_number_stream = read_behavior(behavior, "trial number")
        ...
            source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
            if source_trial_number != trial:
                raise ValueError(
                    f"Unexpected trial number {source_trial_number} for marker {trial} in {path}"
                )
```

iii. Using the stored field with a hard consistency assertion against the marker-derived index means the converter would fail loudly if its trial segmentation ever drifted from the authors', instead of silently emitting mislabelled trials. The values are zero-based within a session (0 to 99 across the dataset), as recorded in the input name `"trial number (zero-based)"`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the median/rounding described above; the scalar is broadcast across all timepoints of the trial as input row 2. Note that because it is the *source* trial number, the 81 lick-corrupted trials leave gaps in the sequence of retained trials rather than renumbering the remaining ones.

ii.
```python
                    np.full(dff.shape[1], source_trial_number),
```

iii. Trial number is specified as a continuous per-trial input; keeping the source numbering preserves the true experiential position of each lap in the session (and therefore the correct relationship to the 30-trial reward switch) rather than compressing it after filtering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavioral time series' event `timestamps`, compared against the behavioral clock at each trial's start/end sample.

ii.
```python
        reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
...
def reward_outcomes(timestamps, starts, stops, reward_timestamps):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
    return outcomes
```

iii. The agent established (step 16) that `Reward` is a sparse event series with its own timestamps rather than a per-frame series, and that reward positions cluster tightly inside the active zone, so an interval containment test on the event times is the natural per-trial outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial rewarded/omitted vector is shifted by one with a 0 prepended for the first trial of the session, then broadcast across all timepoints as input row 3. The shift is over *source* trials, so a retained trial that follows a discarded lick-corrupt trial still sees that trial's true outcome. Across the dataset 84.2% of trials are rewarded and 15.8% omitted, matching the paper's ~15% random omission rate.

ii.
```python
        outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
        previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.float32)
        ...
                    np.full(dff.shape[1], previous_outcomes[trial]),
```
```python
            "first_trial_previous_outcome": (
                "Set to 0 because the outcome of the pre-imaging warm-up trial is unavailable."
            ),
```

iii. The instruction defines the variable as binary (omitted = 0, rewarded = 1) per trial. The first-trial convention is stated explicitly in the metadata: the preceding lap is not in the recording, so 0 is used as the neutral default.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From `position` together with the trial's active reward zone. The active zone is *not* read from the `reward_zone` series; instead it is inferred from where rewards were actually delivered. For each of the two trial blocks (trials 0–29 and trials ≥30, the paper's 30-trial switch point) the code interpolates `position` at every reward event time in the block, takes the median, and picks the nearest of the three known zone centres (A 80–130, B 200–250, C 320–370 cm → centres 105/225/345). Every trial in the block, including omission trials, inherits that label.

ii.
```python
ZONE_BOUNDS = np.asarray([[80.0, 130.0], [200.0, 250.0], [320.0, 370.0]])
ZONE_CENTERS = ZONE_BOUNDS.mean(axis=1)
SWITCH_TRIAL = 30  # zero-based: trials 0--29 are the initial condition
```
```python
    boundaries = [(0, min(SWITCH_TRIAL, n_trials)), (min(SWITCH_TRIAL, n_trials), n_trials)]

    for first, last in boundaries:
        if first == last:
            continue
        lo = timestamps[starts[first]]
        hi = timestamps[stops[last - 1]]
        block_rewards = reward_timestamps[(reward_timestamps >= lo) & (reward_timestamps <= hi)]
        if block_rewards.size == 0:
            raise ValueError(f"No reward deliveries in trial block [{first}, {last})")
        reward_positions = np.interp(block_rewards, timestamps, position)
        median_position = float(np.median(reward_positions))
        label = int(np.argmin(np.abs(ZONE_CENTERS - median_position)))
        labels[first:last] = label
```

iii. Docstring: "The task uses one condition for trials 0–29 and another condition from trial 30 onward on switch days. Inferring one label per block makes omitted trials well-defined and is more robust than assigning a zone independently from the sparse delivery events. On stay days both blocks infer the same label." The zone coordinates are the paper's ("zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm") and the block boundary is the paper's ("Each switch occurred after 30 trials"). The agent validated the inference in step 24 ("the inferred day-3 switch was B→A as indicated by reward positions") and in step 46 reported the resulting A/B/C fractions are "each ≈ one-third"; it also tracked the worst-case margin (max |median reward position − zone centre| = 24.8 cm, well inside the 60 cm that would be needed to mislabel).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed linear distance from the animal's position to the nearest point of the active zone: negative before the zone (`position − zone_start`), exactly 0 anywhere inside the zone, positive after it (`position − zone_stop`). Computed vectorised over the whole trial, then discretised (7-c).

ii.
```python
def distance_classes(position: np.ndarray, zone: int) -> np.ndarray:
    """Bin signed distance to the closest point in the active reward zone."""
    zone_start, zone_stop = ZONE_BOUNDS[zone]
    distance = np.where(
        position < zone_start,
        position - zone_start,
        np.where(position > zone_stop, position - zone_stop, 0.0),
    )
```
```python
            "distance_definition": (
                "Signed linear distance to the nearest point in the active reward zone: "
                "negative before the zone, zero throughout the zone, positive after it."
            ),
```

iii. The instruction asks for "distance to *any* location in the reward zone" with a dedicated "0 cm" class, which only makes sense as a nearest-point distance that saturates at 0 across the whole 50 cm zone — this is what the code implements and what the metadata states.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit masks: 0 for < −50; 1 for [−50, −10); 2 for [−10, 0); 3 for exactly 0 (the default, i.e. inside the zone); 4 for (0, 10]; 5 for (10, 50]; 6 for > 50. Stored as int8. Resulting class fractions: 0.251 / 0.102 / 0.073 / 0.238 / 0.021 / 0.072 / 0.243.

ii.
```python
    result = np.full(position.shape, 3, dtype=np.int8)  # exactly/in the zone
    result[distance < -50.0] = 0
    result[(distance >= -50.0) & (distance < -10.0)] = 1
    result[(distance >= -10.0) & (distance < 0.0)] = 2
    result[(distance > 0.0) & (distance <= 10.0)] = 4
    result[(distance > 10.0) & (distance <= 50.0)] = 5
    result[distance > 50.0] = 6
```
```python
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "in reward zone (0 cm)", "> 0 to 10 cm", "> 10 to 50 cm", "> 50 cm"],
```

iii. The edges are exactly those given in the instructions; "in the zone" is made the default class so that the 0 cm category is reached only through the saturating distance definition, and the `output_values` strings document the half-open conventions chosen at the boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same-sample alignment: `position` is sliced with the identical `[start, stop)` range used for the fluorescence, so distance class *t* corresponds to neural frame *t*. Nothing is shifted or interpolated.

ii.
```python
            trial_position = position[trial_slice]
            ...
            outputs = np.vstack(
                [
                    distance_classes(trial_position, int(zones[trial])),
                    ...
                ]
            )
```

iii. Behaviour in the NWB release is already resampled onto the imaging frame clock, so common indexing is the alignment; the `np.vstack` against `dff.shape[1]`-length rows would fail if the streams ever had different lengths.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioral time series, in cm along the virtual corridor.

ii.
```python
        position = read_behavior(behavior, "position").astype(np.float64, copy=False)
        ...
            trial_position = position[trial_slice]
```

iii. `position` is the VR track coordinate; within the converted trial windows it runs from ~0 to ~450 cm (the −50/−500 values occur only during the excluded teleport period, which the agent verified when choosing the trial boundaries).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretisation — the raw cm values are used directly.

ii.
```python
                    np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. The corridor coordinate is already the requested quantity; the 450 cm track length is recorded in the metadata as `track_length_cm`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins via `np.digitize` with interior edges [90, 180, 270, 360], giving classes 0–4 with open first and last bins so the handful of samples marginally outside [0, 450] fall into the end classes. Class fractions: 0.212 / 0.177 / 0.231 / 0.226 / 0.154.

ii.
```python
                    np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```
```python
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to < 360 cm", ">= 360 cm"],
```

iii. The instruction specifies "5 equal-sized bins spanning the 450 cm track" with exactly these boundaries; the `output_values` labels state the open ends explicitly.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same-sample alignment through the shared `trial_slice`; no shift.

ii.
```python
            trial_position = position[trial_slice]
```

iii. Same reasoning as 7-d: the behavioural and imaging streams share one frame-locked clock in the NWB release.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioral time series (a per-frame cumulative lick count, integer valued 0–~5+).

ii.
```python
        lick = read_behavior(behavior, "lick")
        ...
            trial_lick = lick[trial_slice]
```

iii. The agent's survey showed `lick` is a small non-negative integer count per imaging frame rather than a binary flag, which is what motivated both the >2-count corruption test and the binarisation below.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised: any non-zero count becomes 1. Separately, whole trials whose lick trace matches the paper's circuit-damage signature are dropped from the dataset entirely (see 1-e) rather than having their lick values NaN'd. Resulting class balance: 77.7% no-lick, 22.3% lick.

ii.
```python
                    (trial_lick > 0).astype(np.int8),
```
```python
            if np.mean(trial_lick > 2) > LICK_ERROR_FRACTION:
                dropped_lick_trials.append(trial)
                continue
```

iii. The instruction specifies a binary lick output; the paper likewise states "Remaining lick counts were converted to a binary vector". Dropping rather than NaN-ing the corrupt trials is the agent's adaptation, since the target format has no missing-value representation for a categorical output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same-sample alignment through the shared `trial_slice`; no shift or smoothing.

ii.
```python
            trial_lick = lick[trial_slice]
```

iii. Same reasoning as 7-d.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. As in 7-a: inferred per pre/post-switch block from the positions at which rewards were actually delivered (`Reward` timestamps interpolated into `position`), matched to the paper's A/B/C zone coordinates. The `reward_zone` behavioral series is not used.

ii. See 7-a (`infer_zone_by_trial`).

iii. See 7-a. The agent's stated reason for the block-median approach over per-trial assignment is that reward events are sparse and absent on omission trials, so a block-level estimate "makes omitted trials well-defined and is more robust."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The inferred block label (0 = A, 1 = B, 2 = C) is broadcast across all timepoints of every trial in the block as output row 4, and also recorded per session in `metadata['session_info'][…]['reward_zone_by_block']` together with the median reward position used to infer it. Class fractions are 0.332 / 0.336 / 0.332.

ii.
```python
                    np.full(dff.shape[1], zones[trial], dtype=np.int8),
```
```python
        "reward_zone_by_block": block_zone_labels,
        "median_reward_position_cm_by_block": block_reward_medians,
```
```python
            "reward_zone_inference": (
                "Nearest of the known A/B/C zone centers to the median actual reward "
                "delivery position, independently for trials 0-29 and trials >=30. "
                "Omission trials inherit the active block label."
            ),
```

iii. Step 46: "A/B/C each ≈ one-third", which the agent used as evidence the inference was not systematically biased — consistent with the paper's counterbalanced assignment of starting zones and switch sequences across mice.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` series' event timestamps, tested for containment in each trial's time window.

ii.
```python
        reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
        outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
```

iii. See 6-a: `Reward` is a sparse event series, so a containment test against the trial's start/end timestamps is the direct way to recover the per-trial outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, 1 if any reward event time falls within `[timestamps[start], timestamps[stop]]` (the interval is closed at the teleport sample), else 0; broadcast across all timepoints as output row 5. 84.2% rewarded / 15.8% omitted.

ii.
```python
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
```
```python
                    np.full(dff.shape[1], outcomes[trial], dtype=np.int8),
```

iii. Binary per-trial outcome as specified. The agent checked the resulting omission rate against the paper's "~15% of trials" statement (step 46) as a validation of the reward parsing.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter is deliberately fail-fast rather than repair-oriented: it validates and raises instead of patching. Guards present are (a) start/teleport pairing and ordering; (b) median behavioural sampling interval vs the expected 64.4836 ms; (c) contiguity of imaging plane labels; (d) no reward events in a trial block; (e) missing environment label in a trial; (f) stored `trial number` disagreeing with the marker-derived index; (g) fewer than two usable trials in a session; (h) a zero dF/F baseline, which is clamped to machine epsilon instead of producing inf/NaN. Genuinely corrupt data is handled in one place only: the 81 lick-circuit trials, which are dropped. Notably there is **no** crop for a neural/behaviour length mismatch — the code indexes the fluorescence arrays with behaviour-derived indices, which happens to be safe here because the only mismatch in the release is 10 m17/m18 sessions where imaging has one *extra* frame at the end.

ii.
```python
        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid start/teleport pairing in {path}")
        frame_intervals = np.diff(timestamps)
        if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, rtol=0, atol=1e-6):
            raise ValueError(f"Unexpected behavior sampling interval in {path}")
```
```python
        if not np.array_equal(plane_values, np.arange(len(plane_values))):
            raise ValueError(f"Non-contiguous imaging plane labels in {path}: {plane_values}")
```
```python
    denominator = np.abs(baseline)
    denominator[denominator == 0] = np.finfo(np.float32).eps
```

iii. Step 30 shows the philosophy working as intended: the m17 multi-plane layout made the single-plane read fail, and the agent noted "the single-plane read correctly failed before writing a partial file," then fixed the loader and reran from scratch. The remaining assertions encode every invariant the agent verified during its survey, so any file that violates them stops the conversion rather than silently producing mislabelled trials.

## 13-a. What are the most time-consuming steps of the code?

i. (1) Reading fluorescence and neuropil from HDF5 — done per trial with fancy (point-list) indexing on the ROI axis, ~12,200 paired reads over 152 files, which dominates wall time and is the reason the whole run is I/O bound. (2) The dF/F computation: four full passes (Gaussian, min filter, max filter, Gaussian) over each trial's `n_cells × T` float32 block, for ~2.6 billion cell-samples in total. (3) Pickling and writing the 9.52 GB result in one `pickle.dump`. (4) The final `[trial[keep_neurons] for trial in neural_trials]` rebuild, which copies an entire session's neural data.

ii.
```python
            np.asarray(
                fluorescence[f"plane{plane}"]["data"][start:stop, local_indices],
                dtype=np.float32,
            ).T
```
```python
    with args.output.open("wb") as stream:
        pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent sized the output ahead of time (step 17 computed "cell-time full … GB f32") and chose per-trial, curated-columns-only reads specifically to keep peak memory bounded: "Reading only manually curated cells substantially reduces both I/O and memory." It accepted the I/O cost of many small reads in exchange for never holding a session-wide movie in RAM, and it made the conversion a single pass over each file.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The per-trial HDF5 read inside `paper_dff_trial`: h5py point-selection on the column axis, repeated ~80 times per session, is far slower than one contiguous `data[:, :]` read (or one `data[:, min:max]` block) followed by in-memory column subsetting — this is the single largest avoidable cost. (2) `reward_outcomes` loops over trials doing an O(n_rewards) mask each time; `np.searchsorted(reward_timestamps, timestamps[starts])` vs `…[stops]` would give all outcomes in two vectorised calls. (3) The per-trial conversion loop applies `distance_classes` / `np.digitize` / `(lick > 0)` trial by trial; these could be computed once over the whole session array and then split, though variable trial lengths make that awkward and the saving is negligible next to the neural work. The streaming correlation accumulation is already vectorised (a matrix–vector product per trial).

ii.
```python
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
```

iii. The agent did not discuss vectorisation; the per-trial structure follows directly from the paper's per-trial dF/F baselining, which must be computed on each lap independently and therefore cannot be a single session-wide filter pass.

## 13-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and each trial's fluorescence read exactly once; the reward-zone inference, the behaviour extraction and the neural processing all happen in that single pass, so there is no separate survey/convert double read. What is repeated: the small behavioural arrays (`position`, `speed`, `lick`, `environment`, `trial number`) are re-sliced per trial (cheap); `np.mean(fneu)` is recomputed per trial (required — the paper's baseline is per trial); and the entire neural output is rebuilt once at the end by the `keep_neurons` list comprehension, which touches every element a second time and transiently doubles the session's memory footprint.

ii.
```python
    keep_neurons = speed_correlations <= INTERNEURON_R_THRESHOLD
    neural_trials = [trial[keep_neurons] for trial in neural_trials]
```

iii. The single-pass design was enabled by the agent's choice to infer reward zones from the reward event stream (which is cheap to read up-front) rather than from a statistic that requires the neural or full behavioural data, so no pre-survey over the corpus was needed.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three small items. (1) dF/F is computed for all curated cells including the 400 later excluded as putative interneurons — unavoidable, since the exclusion criterion is itself a function of the dF/F. (2) The five per-trial-constant variables (environment, trial number, previous outcome, reward zone, reward outcome) are tiled across every timepoint even though the target format permits a shape-`(d,)` per-trial vector; this inflates the input/output arrays but is negligible (int8/float32) next to the neural data and makes the decoder's per-timestep interface uniform. (3) `block_reward_medians`, `kept_source_trials` and the other `session_info` bookkeeping are recorded but never consumed by the decoder — they exist for provenance. The dominant cost, storing full-resolution float32 dF/F for 138,278 neuron-sessions (9.52 GB), is not wasted work but is the price of the no-rebinning decision in 2-e.

ii.
```python
                    np.full(dff.shape[1], env),
                    np.full(dff.shape[1], source_trial_number),
                    np.full(dff.shape[1], previous_outcomes[trial]),
```
```python
        "kept_source_trials": kept_source_trials,
        "reward_zone_by_block": block_zone_labels,
        "median_reward_position_cm_by_block": block_reward_medians,
```

iii. The agent's stated aim (module docstring) was that "Decisions which are specific to this decoder are documented in `metadata` in the resulting pickle", so the per-session bookkeeping is intentional auditability rather than oversight; the time-tiling of per-trial constants follows the target format's "If at all possible, make it time-varying" guidance.
