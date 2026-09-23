# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `/app/data` is discovered with a single recursive glob `sub-*/*.nwb`, sorted deterministically, and each file is opened with `pynwb.NWBHDF5IO(..., load_namespaces=True)` inside a context manager. All 152 files (11 subjects) are processed in one pass; there is no separate survey pass. Within each file the AI reads `processing['behavior']['BehavioralTimeSeries']` (position, speed, lick, environment, `trial number`, scanning, reward_zone, autoreward, trial_start, teleport, plus the irregular `Reward` event series) and `processing['ophys']` (`Fluorescence`, `Neuropil`, `ImageSegmentation/PlaneSegmentation`). No `h5py` access is used anywhere. Trials are then cut out of these session-length arrays with the `trial_start`/`teleport` event indices.

ii.
```python
DATA_ROOT = Path("/app/data")

def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    if sample:
        wanted = {("m11", 3), ("m11", 5)}
        files = [p for p in files if session_key(p) in wanted]
        if len(files) != 2:
            raise RuntimeError(f"Expected two sample sessions, found {len(files)}")
    return files
```
```python
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        ...
        behavior = nwb.processing["behavior"]["BehavioralTimeSeries"]
        ophys = nwb.processing["ophys"]
        fluorescence_container = ophys["Fluorescence"]
        neuropil_container = ophys["Neuropil"]
        segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]
```
```python
def load_behavior(behavior, common_length: int) -> dict[str, np.ndarray]:
    names = ["position", "speed", "lick", "environment", "trial number", "scanning",
             "reward_zone", "autoreward", "trial_start", "teleport"]
    arrays = {name: np.asarray(behavior[name].data[:common_length]) for name in names}
    arrays["timestamps"] = np.asarray(behavior["position"].timestamps[:common_length])
    return arrays
```

iii. From CONVERSION_NOTES Step 2: "152 files are organized as `sub-<mouse>/sub-<mouse>_ses-<day>_behavior+ophys.nwb`. Subjects m3, m4, m7, m12–m15, and m17–m19 each have days 1–14; m11 has days 3–14." and "All inspection used `pynwb.NWBHDF5IO(..., load_namespaces=True)`. No `h5py` access was used." Step 4 explicitly resolves the scope question: the paper's main analyses use only the 77 switch days, but the AI converts "all 152 provided sessions because the requested 'full dataset' and task outputs are defined for every condition", tagging switch/stay status in `session_info` instead. Step 9 reconciles 12,216 discovered trials with the paper's 12,376 by noting m11's two missing imaging days (12,376 − 160 = 12,216).

## 1-b. How are the data split into subjects?

i. Subject identity is read from the NWB metadata field `nwb.subject.subject_id` and cross-checked against the subject parsed from the file path with a regular expression; a mismatch raises. The set of subject ids over all converted sessions becomes `data['subjects']`, sorted numerically by the integer after `m`, and `subject_idx` is the index of each session's subject in that list.

ii.
```python
def session_key(path: Path) -> tuple[str, int]:
    match = re.search(r"sub-(m\d+)_ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse session path: {path}")
    return match.group(1), int(match.group(2))
```
```python
    subject_from_path, day = session_key(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        if subject != subject_from_path:
            raise ValueError(f"Subject mismatch in {path}: {subject} vs {subject_from_path}")
```
```python
    subjects = sorted({x["subject"] for x in converted_sessions}, key=lambda x: int(x[1:]))
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    ...
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[x["subject"]] for x in converted_sessions],
                                  dtype=np.int16),
```

iii. CONVERSION_NOTES Step 2 lists "Subjects | 11: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19", and Step 3 records the paper's "counterbalanced across mice (n = 11 mice)", so the directory/metadata count is checked against the paper. Using the in-file `subject_id` rather than only the path makes the dataset self-describing; the path cross-check guards against a mis-filed NWB.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The session (experiment day) number is parsed from the filename (`ses-<NN>`) and kept as `task_day` in the per-session metadata, together with a derived `is_switch_day` flag for days {3, 5, 7, 8, 10, 12, 14}. Sessions are not merged or aligned across days (no cross-day ROI alignment is attempted).

ii.
```python
SWITCH_DAYS = {3, 5, 7, 8, 10, 12, 14}
...
    subject_from_path, day = session_key(path)   # -> ("m11", 3)
...
        session_id = f"sub-{subject}_ses-{day:02d}"
...
        session_meta = {
            "session_id": session_id,
            "source_file": str(path),
            "nwb_identifier": nwb.identifier,
            "scene": scene,
            "task_day": day,
            "is_switch_day": day in SWITCH_DAYS,
            ...
        }
```

iii. CONVERSION_NOTES Step 3: "14 task days, one imaging session/day except m11 began imaging day 3; seven switch days (3,5,7,8,10,12,14)". The day number matters because the reward-zone schedule and the environment switch are day-dependent, so it is retained in metadata rather than discarded. Step 9 confirms "152 total; 77 switch" against the paper's 77 switch sessions.

## 1-d. How are the data split into trials?

i. Trial boundaries come from the two binary behavior event series. A trial starts at the frame where `trial_start > 0` and ends at (exclusive) the frame where `teleport > 0`, i.e. the lap from entry at 0 cm to the teleport onset; the teleport/ITI period is not part of any trial. The code asserts that the number of starts equals the number of teleports (≥2) and that every teleport index is strictly greater than its start index. All arrays are first trimmed to `common_length = min(behavior length, all fluorescence/neuropil row counts)`, so trial indices are valid in every stream.

ii.
```python
        behavior_len = len(behavior["position"].data)
        fluorescence_series = list(fluorescence_container.roi_response_series.values())
        neuropil_series = list(neuropil_container.roi_response_series.values())
        common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                            [x.data.shape[0] for x in neuropil_series])
        b = load_behavior(behavior, common_length)
        starts = np.flatnonzero(b["trial_start"] > 0)
        stops = np.flatnonzero(b["teleport"] > 0)
        if len(starts) != len(stops) or len(starts) < 2:
            raise ValueError(f"Invalid trial event counts in {path}: {len(starts)}/{len(stops)}")
        if np.any(stops <= starts):
            raise ValueError(f"Non-positive trial interval in {path}")
```
```python
        for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
            ...
            position = b["position"][start:stop]
            speed = b["speed"][start:stop]
            lick = b["lick"][start:stop]
```

iii. CONVERSION_NOTES Step 2: "There is no NWB `trials` table. Trial intervals are encoded by matched `trial_start` and `teleport` events (all 152 files have equal counts)". Step 4 addresses the one real trap — the reference repo slices `start-1:stop-1` because its legacy index arrays are one-based: "Slice NWB directly as `[start_event_index:teleport_event_index]`. Applying the legacy −1 offset to explicit NWB indices would introduce a pre-start frame." Step 4 also validates the resulting windows empirically: "trial starts have median position 1.75 cm, final included frames median 448.94 cm", i.e. the slice really is one full 0→450 cm lap.

## 1-e. How are trials filtered based on quality controls?

i. Four per-trial exclusion criteria, evaluated on the original (pre-exclusion) trial list:
1. fewer than two samples between start and teleport;
2. any sample with `scanning != 1` (imaging not running);
3. any non-finite value among position, speed, lick, or timestamps;
4. the paper's lick-sensor failure rule: more than 30% of the trial's frames have a cumulative lick count > 2.

In the full run only criterion 4 fires, removing exactly 81 of 12,216 trials (12,135 retained). Sessions with fewer than two retained trials would raise. Excluded trials are recorded with their reason in `session_info['excluded_trials']`, and — importantly — the *original* trial index is used for trial number and previous-outcome bookkeeping so that an exclusion does not renumber the task history.

ii.
```python
        for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
            reason = None
            if stop - start < 2:
                reason = "fewer than two samples"
            elif not np.all(b["scanning"][start:stop] == 1):
                reason = "outside valid scanning period"
            required = np.vstack((b["position"][start:stop], b["speed"][start:stop],
                                  b["lick"][start:stop], timestamps[start:stop]))
            if not np.all(np.isfinite(required)):
                reason = "non-finite required behavior"
            lick_bad_fraction = float(np.mean(b["lick"][start:stop] > 2))
            if lick_bad_fraction > 0.30:
                reason = "corrupt lick sensor (>30% frames with count >2)"
            if reason is not None:
                excluded_trials.append({"original_trial_index": trial_idx, "reason": reason})
                continue
```
```python
        if len(neural_trials) < 2:
            raise ValueError(f"Fewer than two retained trials in {path}")
```

iii. Methods (quoted in CONVERSION_NOTES Step 3): "A very small number of trials with erroneous lick detection from damage to the circuit were removed … (~0.65% of all imaged trials, n = 81 out of 12,376 trials removed across 11 switch mice). These trials were detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". Step 4 records that the AI compared the manuscript's 0.30 threshold against the repo's 0.35 (`dayData`) and chose the published one: "Exact >30% rule identifies 81/12,216 trials; >35% identifies 69 … Prefer the explicitly reported >30% criterion. Exclude these trials from the joint decoder, because retaining them with fabricated lick labels would corrupt one output; loss is only 0.66%." Step 10 adds that "the current source trial number is retained after exclusions, while previous outcome comes from the immediately preceding **source** trial", so dropping a trial does not silently relabel task history.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The raw Suite2p traces: `ophys/Fluorescence` (F) and `ophys/Neuropil` (Fneu) ROI response series, restricted to the ROIs marked as cells in `ophys/ImageSegmentation/PlaneSegmentation['iscell'][:,0] == 1`. The NWB `Deconvolved` series is deliberately **not** used. For the two-plane animals (m17, m18) the `plane0` and `plane1` response series are both read and their ROIs mapped back onto the global `PlaneSegmentation` row index so all cells are pooled into one matrix.

ii.
```python
        fluorescence_container = ophys["Fluorescence"]
        neuropil_container = ophys["Neuropil"]
        segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]
        ...
        iscell = np.asarray(segmentation["iscell"].data[:])
        manual_ids = np.flatnonzero(iscell[:, 0] == 1)
        ...
        fluorescence = load_selected_roi_series(
            fluorescence_container, manual_ids, common_length, len(segmentation))
        neuropil = load_selected_roi_series(
            neuropil_container, manual_ids, common_length, len(segmentation))
```
```python
def load_selected_roi_series(container, roi_ids, common_length, n_segmentation_rows):
    """Load selected global segmentation rows from one or more plane series."""
    out = np.empty((len(roi_ids), common_length), dtype=np.float32)
    filled = np.zeros(len(roi_ids), dtype=bool)
    for name in sorted(container.roi_response_series):
        series = container[name]
        region_ids = np.asarray(series.rois.data[:], dtype=int)
        ...
        local = np.flatnonzero(np.isin(region_ids, roi_ids))
        global_selected = region_ids[local]
        destination = np.searchsorted(roi_ids, global_selected)
        out[destination] = np.asarray(series.data[:common_length, local]).T
        filled[destination] = True
    if not np.all(filled):
        raise ValueError(...)
    return out
```

iii. CONVERSION_NOTES Step 4 documents a direct empirical test rather than an assumption: "NWB `Deconvolved` has raw-scale values (example max 2,484) and only modest correlation (0.25–0.67 in five cells) to reconstructed reference events (max 0.53) … Recompute reference dF/F/events from NWB `Fluorescence` and `Neuropil`; do not treat NWB `Deconvolved` as the final paper signal. This is the major resolved processing discrepancy." Step 7 records that the multi-plane case was found and fixed during validation: "a performance probe revealed that m17/m18 response series are split into `plane0` and `plane1`. The initial loader assumed only `plane0`; it was fixed to concatenate all plane series through each `DynamicTableRegion`."

## 2-b. How is the `neural` data processed?

i. The paper's own dF/F + deconvolution pipeline is reimplemented, per trial:
1. subtract `0.7 × Fneu`, then add back the trial-mean of `0.7 × Fneu` so the denominator is a true baseline rather than a near-zero number;
2. maximin baseline: Gaussian smooth along time with σ = 15 samples (NaN-robust), then a 300-sample running minimum followed by a 300-sample running maximum (≈20 s window at 15.5 Hz);
3. `dF/F = (F_corrected − baseline) / |baseline|`;
4. Gaussian smooth dF/F with σ = 2 samples;
5. after the interneuron filter, deconvolve each trial with Suite2p's OASIS at `tau = 0.7` and the per-plane frame rate 15.5078125 Hz; negative values are clipped to 0 and stored as float32.

Cells from both imaging planes are pooled into one neuron axis. The baseline window is always restricted to the lap (the reference `dff` default `keep_teleports=False`); the repo's per-mouse/per-day `teleport_metadata.py` table (days on which the laser was not blanked, so the baseline may span the teleport) is not applied.

ii.
```python
def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Reference nansmooth behavior without propagating missing values."""
    nan_mask = np.isnan(values)
    clean = values.copy(); clean[nan_mask] = 0.0
    weights = np.ones(values.shape, dtype=np.float64); weights[nan_mask] = 0.001
    smooth = ndimage.gaussian_filter1d(clean, sigma, axis=-1)
    smooth_weights = ndimage.gaussian_filter1d(weights, sigma, axis=-1)
    return smooth / smooth_weights


def calculate_reference_dff(fluorescence, neuropil, trial_starts, trial_ends):
    """Reproduce preprocessing.dff for trial periods, before deconvolution."""
    dff = np.full((n_cells, n_time), np.nan, dtype=np.float64)
    for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
        # Reference operation: subtract 0.7*Fneu, then add its per-trial mean
        # before baseline division so the denominator is not artificially small.
        f = fluorescence[:, start:stop].astype(np.float64, copy=True)
        fn = neuropil[:, start:stop].astype(np.float64, copy=False)
        corrected = f - 0.7 * fn + 0.7 * np.nanmean(fn, axis=1, keepdims=True)
        baseline = gaussian_smooth(corrected, 15)
        baseline = ndimage.minimum_filter1d(baseline, 300, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
        trial_dff = (corrected - baseline) / np.abs(baseline)
        trial_dff = gaussian_smooth(trial_dff, 2)
        dff[:, start:stop] = trial_dff
    return dff, example
```
```python
FRAME_RATE_HZ = 15.5078125
...
            trial_dff = dff[:, start:stop]
            events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
            events[events < 0] = 0
```

iii. CONVERSION_NOTES Step 3 quotes the Methods: "Raw F is neuropil corrected, then a per-trial baseline is estimated by maximin with a 20 s window. dF/F is `(F-baseline)/abs(baseline)`, smoothed with a two-sample (~0.129 s s.d.) Gaussian, then OASIS-deconvolved with a canonical calcium kernel." Step 1 identifies the source function: "`dff` … Restrict fluorescence to trial intervals; subtract 0.7× neuropil; estimate per-trial maximin baseline (15-frame smoothing then 300-frame min/max filters); calculate dF/F; smooth by 2 frames; optionally OASIS-deconvolve." Step 10's independent re-derivation from raw NWB for five m11 day-3 trials passed `np.allclose(rtol=1e-6, atol=1e-6)` on the complete neuron×time matrices. The `keep_teleports` table is never mentioned in the notes — the AI ran the reference `dff` with `keep_teleports=False` during exploration and carried that setting forward without discussing the per-day exception.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Only ROIs with Suite2p manual curation label `iscell[:,0] == 1` are loaded at all (138,678 of 312,110 candidate ROIs). (2) Putative interneurons are then dropped: any cell whose trial-restricted dF/F has Pearson correlation with running speed greater than 0.5 (402 cells, 0.35 ± 0.61% per session), leaving 138,276 neurons. The correlation is computed vectorised over cells on all finite samples of the session. Cells with an undefined (NaN) correlation are kept. Place-cell selection is deliberately *not* applied.

ii.
```python
        iscell = np.asarray(segmentation["iscell"].data[:])
        if iscell.ndim != 2 or iscell.shape[1] < 1:
            raise ValueError(f"Unexpected iscell shape in {path}: {iscell.shape}")
        manual_ids = np.flatnonzero(iscell[:, 0] == 1)
```
```python
def speed_correlations(dff: np.ndarray, speed: np.ndarray) -> np.ndarray:
    valid = np.isfinite(dff[0]) & np.isfinite(speed)
    x = dff[:, valid]
    y = speed[valid].astype(np.float64, copy=False)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y)
    numerator = x_centered @ y_centered
    denominator = np.sqrt(np.sum(x_centered * x_centered, axis=1) *
                          np.sum(y_centered * y_centered))
    return numerator / denominator
```
```python
        correlations = speed_correlations(dff, b["speed"])
        keep = ~(correlations > 0.5)
        kept_ids = manual_ids[keep]
        dff = dff[keep]
```

iii. CONVERSION_NOTES Step 3: "Additional putative interneurons are excluded when Pearson correlation between dF/F and running speed exceeds 0.5 (0.42 ± 0.85% of cells across mice/days)." Step 2 warns about a real trap in the data: "`iscell` [is] two-column Suite2p … (`[:,0]` is the curated 0/1 label; `[:,1]` is probability). It is essential not to interpret the probability column as another cell label." Step 4 justifies *not* filtering to place cells: "Do not restrict to place cells: the new multi-output task includes movement, licking, and outcome, for which non-place pyramidal cells remain relevant; place-cell classification is analysis-specific and computationally stochastic." Step 10 checks the result against the paper: "402/138,678 cells were excluded. The session-wise percentage is 0.348 ± 0.609% … reasonably consistent with the paper's 0.42 ± 0.85%."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No extra alignment step is needed. Behavior and imaging are already synchronised in the NWB export at one behavior sample per imaging frame, so the same integer index range `[trial_start_frame, teleport_frame)` slices the neural, input and output streams. Every trial therefore begins exactly at the alignment event (`time_from_trial_start_s[0] == 0`), and `off_start = 0.0`, `off_end = None` (variable trial length) are recorded in metadata. The code asserts per trial that the neural, input and output time axes have identical length.

ii.
```python
            time_from_start = timestamps[start:stop] - timestamps[start]
            ...
            trial_dff = dff[:, start:stop]
            events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
            ...
            if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
                raise AssertionError(f"Alignment shape mismatch in {path}, trial {trial_idx}")
```
```python
            "temporal_alignment_event": "trial_start (entry into the 0-cm virtual corridor)",
            "off_start": 0.0,
            "off_end": None,
```

iii. CONVERSION_NOTES Step 4: "Use behavioral timestamps as authoritative and trim every stream to its common length. This preserves the actual one-row-per-aligned-frame representation." Step 10 check 5: "across 12,135 converted trials, zero neural/input/output shape failures, zero nonzero/negative trial-start times, zero nonmonotonic time vectors". Step 12 also records visual confirmation from `processing_sub-m11_ses-03.png`: "Raw F/Fneu, maximin baseline, dF/F/OASIS, position and active zone, trial-start time, speed/lick, and all discretized outputs share the same samples."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data are kept at the native imaging-frame resolution: 15.5078125 Hz per plane → 64.48362720403023 ms per bin. **No rebinning or resampling is applied** — no temporal downsampling, no spatial (10 cm) binning. For the two-plane animals the stored `rate` on the response series is the scanner rate (31.015625 Hz), and the per-plane rate 15.5078125 Hz is the one used both for the time bin and for the OASIS kernel; the row count of those series already matches the 15.5 Hz behavior stream. The bin size is a module constant derived from the frame rate rather than measured per session.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ      # 64.48362720403023 ms
...
            "time_bin_size": TIME_BIN_MS,
            "time_bin_size_units": "ms",
```
```python
    print(f"Reference time bin: {TIME_BIN_MS:.9f} ms", flush=True)
```

iii. CONVERSION_NOTES Step 3: "Neural data time bin | ~64.5 ms (15.5 Hz); two-plane acquisition interleaved at ~31 Hz gives ~15.5 Hz/plane". Step 5 key decision 4: "**Native temporal sampling**: Do not resample. All behavior timestamps have a 64.483627 ms median interval, already synchronized to one neural row; trim ten +1-row neural sessions to the common length." Step 10 reference comparison: "Binning | ~15.5-Hz events; spatial analyses separately use 10-cm bins | native ~64.484-ms imaging frames | Correct for the requested temporal decoder; spatial 10-cm binning would conflict with trial-start time alignment and specified categorical cut points."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attribute of the `position` behavior time series (all regularly sampled behavior series in `BehavioralTimeSeries` carry an identical timestamp vector, so the choice of series is immaterial). The vector is trimmed to `common_length` along with the data arrays.

ii.
```python
    arrays["timestamps"] = np.asarray(behavior["position"].timestamps[:common_length])
```
```python
        timestamps = b["timestamps"]
```

iii. CONVERSION_NOTES Step 2 establishes that all regular behavior series are on the same synchronized imaging-frame clock: "`behavior/BehavioralTimeSeries` has aligned series: `position` (cm), `speed` (cm/s), cumulative-per-frame `lick`, `environment`, `trial number`, binary/event-count `trial_start`, `teleport`, `reward_zone`, `autoreward`, `scanning`, plus irregular event series `Reward`". Step 4: "Use behavioral timestamps as authoritative". (I independently confirmed on `sub-m11_ses-03` that `position`, `speed`, `lick`, `environment` and `trial number` share byte-identical timestamp vectors, so this is equivalent to using any other series.)

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial's first timestamp is subtracted, giving seconds elapsed since the alignment event; the first value of every trial is exactly 0 and the row is stored as float32. No rescaling, resampling or normalisation.

ii.
```python
            time_from_start = timestamps[start:stop] - timestamps[start]
            inputs = np.vstack((
                time_from_start,
                np.full(stop - start, environment),
                np.full(stop - start, trial_number),
                np.full(stop - start, previous_outcome),
            )).astype(np.float32)
```

iii. Straightforward; the AI's justification is the metadata contract it set for itself in Step 5 key decision 8 (`alignment = trial_start / entry to 0-cm corridor`, `off_start = 0.0`). Step 10 check 5 verifies the invariant across the whole dataset: "zero nonzero/negative trial-start times, zero nonmonotonic time vectors".

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `[start:stop)` frame index range is used for the timestamps and for the dF/F/event matrix, after every stream has been trimmed to a single `common_length`. A per-trial assertion requires `n_timepoints` to be identical across neural, input and output. No interpolation or shifting is performed.

ii.
```python
        common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                            [x.data.shape[0] for x in neuropil_series])
        b = load_behavior(behavior, common_length)
```
```python
            if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
                raise AssertionError(f"Alignment shape mismatch in {path}, trial {trial_idx}")
```

iii. CONVERSION_NOTES Step 2: "Ten two-plane sessions have one more neural row than behavior and require truncation to the common length." Step 10: "Ten multi-plane files and one single-plane file have one extra neural row; common-length trimming removes only that unmatched trailing row. Every included trial remains inside the common synchronized interval."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series (0 = ENV 1, 1 = ENV 2; the value is −1 outside laps, during the teleport period).

ii.
```python
    names = ["position", "speed", "lick", "environment", "trial number", "scanning",
             "reward_zone", "autoreward", "trial_start", "teleport"]
...
            environment = mode_value(b["environment"][start:stop])
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`environment` | `input[1]` | Mode within trial, preserve 0=ENV1 / 1=ENV2, repeated through trial | `behavior.get_trial_types` | Binary per-trial context." Step 2 records that 73 files are Env1-only, 68 Env2-only and 11 contain both (the day-8 environment switch), which is consistent with the paper's design.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The modal value of the `environment` samples within the trial is taken and broadcast as a constant row over all timepoints of the trial (stored as float32). Taking the mode rather than the raw per-sample values makes the value robust to a stray −1 sentinel at a lap boundary; the environment never actually changes within a lap, so the row is constant.

ii.
```python
def mode_value(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute mode of empty/non-finite values")
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[np.argmax(counts)])
...
            environment = mode_value(b["environment"][start:stop])
            inputs = np.vstack((
                time_from_start,
                np.full(stop - start, environment),
                ...
```

iii. Step 5 key decision 6: "**Mixed time-varying/per-trial variables**: Store all four inputs and all six outputs as 2-D arrays with per-trial labels repeated over time. This is necessary because one NumPy array cannot mix 1-D and 2-D rows and makes alignment explicit." Step 10 reference comparison: "modal categorical values are robust to frame representation". Step 5 planned check: "Confirm all 11 day-8 environment transitions occur at original trial index 30"; Step 9 reports environment range [0, 1] in the converted data.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The NWB `trial number` behavior time series (not the loop counter). Its value is −1 during the teleport/ITI and the within-lap value is the zero-based lap index.

ii.
```python
            trial_number = mode_value(b["trial number"][start:stop])
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`trial number` | `input[2]` | Mode within trial (native zero-based number), repeated through trial | `behavior.get_trial_types` pattern | Continuous per-trial covariate." Step 9 reports "Trial-number range | … | [0, 99] | [0, 99] | Yes", matching the 100-trial maximum session. Step 10 notes the deliberate choice to keep the *source* numbering after exclusions: "The current source trial number is retained after exclusions … Thus removal of a corrupt-lick trial does not silently relabel task history." (I independently confirmed on six random sessions that the per-trial modal `trial number` is exactly `0..n_trials-1` in `trial_start` order, so this is equivalent to using the loop index.)

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Modal value within the trial, broadcast as a constant float32 row over the trial's timepoints. No renumbering after trial exclusion, no normalisation, no per-session offset (numbering restarts at 0 in each session).

ii.
```python
            trial_number = mode_value(b["trial number"][start:stop])
            inputs = np.vstack((
                time_from_start,
                np.full(stop - start, environment),
                np.full(stop - start, trial_number),
                np.full(stop - start, previous_outcome),
            )).astype(np.float32)
```

iii. Same as 5-a: Step 5 mapping plus Step 10's statement that source-trial numbering is preserved through exclusions. Step 10 check 5 verified "zero failures of per-trial variable constancy" across all 12,135 trials.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The irregular `Reward` event series in `BehavioralTimeSeries`, specifically its `timestamps` (reward-delivery times in seconds). A per-trial reward outcome vector is computed once for *all* source trials by testing whether any reward timestamp falls inside `[t(trial_start), t(teleport))`; the previous-trial input is that vector shifted by one source trial.

ii.
```python
        reward_times = np.asarray(behavior["Reward"].timestamps[:])
        timestamps = b["timestamps"]
        outcomes = np.array([
            np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
            for start, stop in zip(starts, stops)
        ], dtype=np.int16)
```

iii. CONVERSION_NOTES Step 5 mapping: "`Reward.timestamps` | `input[3]` | Map deliveries to all original start-to-teleport intervals; for trial i use outcome i−1; first trial=0; repeat through trial | `behavior.get_trial_types` | 0=previous omitted/no preceding trial, 1=previous rewarded. Excluding a corrupt-lick trial does not alter the next trial's true predecessor." Step 4 records that the AI checked the delivery counts: 10,342 of 10,345 `Reward` events fall inside a lap; the other three "occur during unusually long inter-trial gaps in two m4 sessions" and are intentionally left unassigned.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `previous_outcome = outcomes[trial_idx - 1]` using the *source* trial index, with 0 for the first trial of a session (no predecessor). It is broadcast as a constant float32 row over the trial. The predecessor is never taken from the previous *retained* trial, so an excluded corrupt-lick trial does not corrupt the history of the following trial. There is no carry-over across session boundaries.

ii.
```python
            previous_outcome = int(outcomes[trial_idx - 1]) if trial_idx > 0 else 0
            inputs = np.vstack((
                time_from_start,
                np.full(stop - start, environment),
                np.full(stop - start, trial_number),
                np.full(stop - start, previous_outcome),
            )).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping (quoted above) and Step 10 check 5: "zero preceding-outcome mismatches for adjacent source trials" across the whole dataset; Step 10 check 3 independently reconstructed "preceding source-trial reward outcome" for m11 day-3 trials 0, 5, 29, 30 and 79 and all passed `np.allclose`, "includ[ing] the first-trial default and trials immediately before/after the switch".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Two things: the `position` behavior time series, and the active reward zone for that trial. The active zone is **not** inferred from the `reward_zone` behavior stream; it is derived deterministically from the NWB file `identifier` (the VR scene name, e.g. `Env1_LocationB_to_A`, `Env1_C_to_Env2_A`, `Env2_LocationC`) combined with the paper's rule that a switch happens after the first 30 trials. Zone coordinates are the paper's: A = 80–130 cm, B = 200–250 cm, C = 320–370 cm.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def scene_from_identifier(identifier: str) -> str:
    return identifier.rstrip("/").split("/")[-1]

def scene_zone_labels(scene: str) -> list[str]:
    labels: list[str] = []
    for token in scene.split("_"):
        if token in ZONE_BOUNDS:
            labels.append(token)
        elif token.startswith("Location") and token[-1:] in ZONE_BOUNDS:
            labels.append(token[-1])
    labels = list(dict.fromkeys(labels))
    if len(labels) not in (1, 2):
        raise ValueError(f"Could not derive one or two reward zones from scene {scene!r}: {labels}")
    return labels

def zones_by_trial(scene: str, n_trials: int) -> list[str]:
    labels = scene_zone_labels(scene)
    if len(labels) == 1:
        return [labels[0]] * n_trials
    if n_trials <= 30:
        raise ValueError(f"Switch scene {scene!r} has only {n_trials} trials")
    return [labels[0]] * 30 + [labels[1]] * (n_trials - 30)
```
```python
        scene = scene_from_identifier(nwb.identifier)
        zones = zones_by_trial(scene, len(starts))
```

iii. This is a direct port of the reference repo's `behavior.get_reward_zones(sess, rz_dict=None, change_trial=30)`, which the AI identified in Step 1: "`get_reward_zones` … Assign per-trial reward zones A/B/C using scene/switch identity; canonical zones are A/X=80–130, B/Y=200–250, C/Z=320–370 cm." Step 3 quotes the Methods for both the coordinates ("zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm") and the switch timing ("Each switch occurred after 30 trials"). Step 4: "Parse scene identifier with reference zone coordinates; switch after 30 trials. Validate environment stream and zone-entry positions." Step 10 check 4 verifies the absence of an off-by-one: "Trials 29/30 explicitly establish the absence of a zone-switch off-by-one error."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance in cm from the animal's position to the nearest point of the active zone: `position − zone_start` when before the zone (negative), `position − zone_end` when past it (positive), and exactly 0.0 everywhere inside the 50 cm zone. It is computed vectorised over the trial with nested `np.where`, then discretized (see 7-c). No smoothing, no speed masking, no circular wrapping (the paper's circular reward-relative coordinate is deliberately not used here).

ii.
```python
def reward_distance(position: np.ndarray, zone: str) -> np.ndarray:
    start, stop = ZONE_BOUNDS[zone]
    return np.where(position < start, position - start,
                    np.where(position > stop, position - stop, 0.0))
...
            distance = reward_distance(position, zone)
            outputs = np.vstack((
                discretize_reward_distance(distance),
                ...
```

iii. CONVERSION_NOTES Step 3: "Reward-relative paper analyses center circular coordinates on reward-zone **start**. The requested signed distance 'to any location in the reward zone' instead requires zero throughout the entire 50 cm zone, negative distance before its start, and positive distance after its end." Step 10: "Requested seven/five/five-class discretizations replace the paper's circular continuous coordinate as explicitly required."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned by explicit boolean masks rather than `np.digitize`:
`< −50 → 0`; `[−50, −10] → 1`; `(−10, 0) → 2`; `== 0 → 3`; `(0, 10] → 4`; `(10, 50] → 5`; `> 50 → 6`.
The instruction's ambiguous endpoints (−10, +10) are resolved in favour of the lower class. A boundary unit test runs at startup on every invocation of the script.

ii.
```python
def discretize_reward_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int16)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```
```python
def check_discretizer_boundaries() -> None:
    dist = np.array([-51, -50, -10, -9.999, -0.1, 0, 0.1, 10, 10.001, 50, 50.001])
    expected_dist = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6])
    if not np.array_equal(discretize_reward_distance(dist), expected_dist):
        raise AssertionError("Reward-distance boundary test failed")
...
def main() -> None:
    args = parse_args()
    check_discretizer_boundaries()
```

iii. CONVERSION_NOTES Step 5: "Boundary policy: −50 starts bin1; −10 remains bin1; +10 remains bin4; +50 remains bin5; zero is exactly bin3." Step 10 edge-case review: "Distance/position/speed boundary unit tests cover every exact cut point. In particular, distance −10 belongs to class 1, 0 to class 3, +10 to class 4, +50 to class 5." The `OUTPUT_VALUES` labels record the same thresholds in the pickle.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is indexed with the identical `[start:stop)` frame slice as the neural events, so no alignment step is required; the per-trial shape assertion enforces it.

ii.
```python
            position = b["position"][start:stop]
            ...
            distance = reward_distance(position, zone)
            ...
            trial_dff = dff[:, start:stop]
            events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
            ...
            if events.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
                raise AssertionError(f"Alignment shape mismatch in {path}, trial {trial_idx}")
```

iii. CONVERSION_NOTES Step 7 plot review: "Raw F/Fneu, corrected F/baseline, smooth dF/F/OASIS, monotonic 0→450 cm trial trajectories, zone overlays, frame-aligned inputs, source speed/lick, and every discretized output are mutually aligned. Distance is exactly class 3 throughout the zone. No temporal shift or discretization anomaly was visible."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the 450 cm virtual corridor), sliced to the trial.

ii.
```python
            position = b["position"][start:stop]
            outputs = np.vstack((
                discretize_reward_distance(distance),
                discretize_position(position),
                ...
```

iii. CONVERSION_NOTES Step 2: "`behavior/BehavioralTimeSeries` has aligned series: `position` (cm) …" and "Valid on-track position spans approximately 0–450 cm." Step 5 mapping: "`position` | `output[1]` | 5 bins: <90, [90,180), [180,270), [270,360], >360".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing to the trial and discretizing. The raw cm values are used directly — no re-zeroing, no unwrapping, no smoothing, no clipping. Because trials are cut from lap start to teleport, only on-track samples (~0–450 cm) are present; the small overshoots at the track ends fall into the open first/last bins.

ii.
```python
            position = b["position"][start:stop]
            ...
            outputs = np.vstack((
                discretize_reward_distance(distance),
                discretize_position(position),
                ...
            )).astype(np.int16, copy=False)
```

iii. CONVERSION_NOTES Step 5: "Tiny synchronized endpoint overshoots remain in extreme bins." Step 4 records the empirical check on the slice: "trial starts have median position 1.75 cm, final included frames median 448.94 cm". Step 9 reports the resulting distribution [.212, .177, .231, .226, .154], close to the uniform occupancy expected for a 450 cm track cut into five equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track, assigned with explicit boolean masks: `< 90 → 0`; `[90, 180) → 1`; `[180, 270) → 2`; `[270, 360] → 3`; `> 360 → 4`. The default fill is 0, so anything below 90 cm (including small negative overshoots) lands in class 0. Note the deliberate reading of the instruction text "3: 270 to 360 cm / 4: > 360 cm": exactly 360 cm is class 3. Covered by the startup boundary unit test.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int16)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```
```python
    pos = np.array([89.9, 90, 179.9, 180, 269.9, 270, 360, 360.1])
    if not np.array_equal(discretize_position(pos), [0, 1, 1, 2, 2, 3, 3, 4]):
        raise AssertionError("Position boundary test failed")
```

iii. CONVERSION_NOTES Step 10 edge-case review: "position 360 and speed 40 remain in their inclusive classes as specified." The `OUTPUT_VALUES` entry `["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to 360 cm", "> 360 cm"]` documents the same convention in the saved file.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical `[start:stop)` frame slice as the neural data; no resampling or shifting. Enforced by the per-trial shape assertion and confirmed by the `--show-processing` overlays.

ii.
```python
            position = b["position"][start:stop]
            ...
            trial_dff = dff[:, start:stop]
```

iii. Same justification as 2-d/7-d: CONVERSION_NOTES Step 4 ("behavioral timestamps as authoritative … one-row-per-aligned-frame") and Step 10 check 4, which reconstructed position independently from raw NWB for five trials and passed `np.allclose(rtol=0, atol=0)`.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series, which stores a cumulative lick count per imaging frame (observed values run up to ~6).

ii.
```python
            lick = b["lick"][start:stop]
            ...
                (lick > 0).astype(np.int16),
```

iii. CONVERSION_NOTES Step 1: "Licks are cumulative counts per imaging frame in the original synchronized stream; reference code clips values >1 to 1 for event presence." Step 5 mapping: "`lick` | `output[3]` | `lick>0` → 1 else 0 | `glmUtils.get_timeseries_data`, lick Methods".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a positive count becomes 1, otherwise 0. No temporal smoothing, no conversion to a lick *rate* (the paper's spatially binned, occupancy-normalised lick rate is not used, because the requested output is a binary per-timepoint variable). Separately, whole trials whose lick sensor was faulty by the paper's criterion (>30% of frames with cumulative count > 2) are excluded from the dataset entirely rather than having their lick values set to NaN.

ii.
```python
            lick = b["lick"][start:stop]
            outputs = np.vstack((
                ...
                (lick > 0).astype(np.int16),
                ...
```
```python
            lick_bad_fraction = float(np.mean(b["lick"][start:stop] > 2))
            if lick_bad_fraction > 0.30:
                reason = "corrupt lick sensor (>30% frames with count >2)"
```

iii. CONVERSION_NOTES Step 3: "Lick count is converted to binary per frame. Reference lick-only analysis invalidates corrupt trials, but for the requested joint decoder those trials can retain other outputs while lick samples/trials need explicit handling." Step 4 settles it: "Exclude these trials from the joint decoder, because retaining them with fabricated lick labels would corrupt one output; loss is only 0.66% and is exactly reproduced in available sessions." The resulting lick distribution is [.777, .223].

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop)` frame slice as the neural data; the lick stream is already synchronized to imaging frames in the NWB export, so no shift or interpolation is applied.

ii.
```python
            lick = b["lick"][start:stop]
            ...
            trial_dff = dff[:, start:stop]
```

iii. Step 10 check 4 independently recomputed binary lick from raw NWB for five trials and passed `np.allclose(rtol=0, atol=0)`; Step 7 plot review confirms lick overlays its discretized version with no visible offset.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The NWB file `identifier` (VR scene name) plus the trial index, exactly as in 7-a — not from the `reward_zone` behavior stream (which is loaded but unused). Single-location scenes (`Env1_LocationB`) give one zone for every trial; switch scenes (`Env1_LocationB_to_A`, `Env1_C_to_Env2_A`) give the first label for trials 0–29 and the second for trials 30+.

ii. See 7-a (`scene_from_identifier`, `scene_zone_labels`, `zones_by_trial`), plus:
```python
        scene = scene_from_identifier(nwb.identifier)
        zones = zones_by_trial(scene, len(starts))
...
        for trial_idx, (start, stop, zone) in enumerate(zip(starts, stops, zones)):
```

iii. See 7-a. Step 5 mapping: "NWB identifier scene + trial index | `output[4]` | Parse active zone; switch after 30 original trials; A/B/C → 0/1/2; repeat through trial | `behavior.get_reward_zones`". Note the zone list is built over the *original* trial list, so an excluded trial does not shift the switch point.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The label is mapped to an integer via `"ABC".index(zone)` (A → 0, B → 1, C → 2) and broadcast as a constant int16 row across the trial's timepoints.

ii.
```python
            outputs = np.vstack((
                discretize_reward_distance(distance),
                discretize_position(position),
                discretize_speed(speed),
                (lick > 0).astype(np.int16),
                np.full(stop - start, "ABC".index(zone), dtype=np.int16),
                np.full(stop - start, outcomes[trial_idx], dtype=np.int16),
            )).astype(np.int16, copy=False)
```
```python
OUTPUT_VALUES = [
    ...
    ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
    ...
]
```

iii. Step 5 key decision 6 (per-trial labels repeated over time so a single 2-D array can hold mixed time-varying and per-trial outputs). Step 9 checks the distribution against the balanced counterbalancing of the design: "Reward-zone distribution | zones A/B/C | trial reward location | trial counts [4172, 3974, 3989] | [.332, .336, .333] | Yes".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The irregular `Reward` event series' `timestamps` (reward-delivery times), compared against the trial's start and teleport timestamps. The `Reward` *data* values (volumes, all identical) are not used, nor is `autoreward`.

ii.
```python
        reward_times = np.asarray(behavior["Reward"].timestamps[:])
        timestamps = b["timestamps"]
        outcomes = np.array([
            np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
            for start, stop in zip(starts, stops)
        ], dtype=np.int16)
```

iii. CONVERSION_NOTES Step 2 describes the stream: "plus irregular event series `Reward` (mL at delivery timestamps)". Step 4: "Define outcome only from deliveries inside each trial: 84.66% rewarded / 15.34% omission, matching paper [~15% random omissions]."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A reward-delivery timestamp falling in the half-open interval `[t(trial_start), t(teleport))` marks the trial as rewarded (1), otherwise omitted (0). The value is broadcast as a constant int16 row over all timepoints of the trial — including timepoints *before* the delivery, which the AI explicitly identified as a ceiling on decodability rather than a bug. Three `Reward` events that fall in unusually long inter-trial gaps in two m4 sessions are deliberately left unassigned to any trial.

ii.
```python
        outcomes = np.array([
            np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
            for start, stop in zip(starts, stops)
        ], dtype=np.int16)
...
                np.full(stop - start, outcomes[trial_idx], dtype=np.int16),
...
            "reward_events_total": len(reward_times),
            "rewarded_trials": int(np.sum(outcomes)),
            "unmapped_reward_events": int(len(reward_times) - np.sum(outcomes)),
```

iii. CONVERSION_NOTES Step 9: "Three reward events occur after teleport during unusually long inter-trial intervals and are not assigned to either adjacent trial, matching explicit trial boundaries and the paper's definition of trial outcome." Step 12 investigates the low decoding accuracy (0.579 vs 0.5 chance) and concludes it is intrinsic, not a bug: "of 2,169,754 bins in rewarded trials, 865,799 (39.90%) occur before the first reward delivery. The required per-trial label marks those bins 'rewarded' even though current neural activity cannot yet reveal a random future omission … the paper's GLM uses a different *causal* predictor that switches from 0 to 1 only after reward delivery; changing this task's explicitly per-trial outcome to that signal would violate the requested output definition."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several categories:
- **Neural/behavior length mismatch** (10 two-plane files plus 1 single-plane file have one extra neural row): every stream is trimmed to `common_length = min(behavior, all F, all Fneu)` before anything else; both source lengths are recorded in `session_info`.
- **Degenerate trials**: trials with <2 samples, with any `scanning != 1` sample, or with any non-finite position/speed/lick/timestamp are dropped and logged with a reason.
- **Corrupt lick sensor**: the paper's 81 trials are dropped and logged (see 1-e).
- **Rewards outside any trial**: the three ITI deliveries are simply not mapped to a trial; the count is recorded as `unmapped_reward_events`.
- **Multi-plane ROI bookkeeping**: the loader validates that ROI region ids are in range and that every selected segmentation row is filled by exactly one plane series; otherwise it raises.
- **Hard failures**: subject mismatch, unequal start/teleport counts, non-positive trial intervals, an empty `iscell`, an empty post-interneuron-filter cell set, non-finite dF/F or OASIS output, mismatched time axes, and out-of-range class values all raise rather than being silently patched. A final `validate_converted()` re-checks the whole assembled dictionary.

ii.
```python
        common_length = min([behavior_len] + [x.data.shape[0] for x in fluorescence_series] +
                            [x.data.shape[0] for x in neuropil_series])
```
```python
            if stop - start < 2:
                reason = "fewer than two samples"
            elif not np.all(b["scanning"][start:stop] == 1):
                reason = "outside valid scanning period"
            required = np.vstack((...))
            if not np.all(np.isfinite(required)):
                reason = "non-finite required behavior"
            ...
            if reason is not None:
                excluded_trials.append({"original_trial_index": trial_idx, "reason": reason})
                continue
```
```python
            if not np.all(np.isfinite(trial_dff)):
                raise ValueError(f"Non-finite dF/F in {path}, trial {trial_idx}")
            events = dcnv.oasis(trial_dff, 2000, 0.7, FRAME_RATE_HZ).astype(np.float32)
            events[events < 0] = 0
            if not np.all(np.isfinite(events)):
                raise ValueError(f"Non-finite OASIS events in {path}, trial {trial_idx}")
```
```python
def validate_converted(data: dict) -> None:
    ...
            for out_idx, names in enumerate(OUTPUT_VALUES):
                values = outputs[out_idx]
                if values.min() < 0 or values.max() >= len(names):
                    raise AssertionError(f"Session {s}, trial {tr}, output {out_idx}: invalid class")
```

iii. CONVERSION_NOTES Step 4 lists each discovered anomaly with its resolution (extra neural row, three ITI rewards, updated m17/m18 curation). Step 10's edge-case review states: "Ten multi-plane files and one single-plane file have one extra neural row; common-length trimming removes only that unmatched trailing row … Three rewards outside all start-to-teleport intervals remain unmapped rather than leaking into an adjacent outcome. All NaN/invalid/scanning checks pass on retained intervals." Step 5 key decision 5 lists the trial-validity contract.

## 13-a. What are the most time-consuming steps of the code?

i. The AI instrumented the script with per-session and projected-total timing printed to stdout, and reports the full conversion at 442.9 s (7.38 min) for 152 sessions, with per-session times from 0.59 s (155-cell m11 session) to ~8 s (2,323-cell two-plane session) and a 7.38 s pickle write of the 9.53 GB output. The dominant costs identified are: (1) NWB I/O — reading the F and Fneu matrices for the curated ROIs; (2) the per-trial maximin baseline (a σ=15 Gaussian plus two 300-sample rank filters over cells×frames); (3) per-trial OASIS deconvolution; (4) the final pickle write. The AI's own summary is that cost scales with cell count: "Loading 92 GB of NWB assets and applying filters cell×frame is intrinsically expensive."

Notably, unlike the reference solution the AI makes only **one** pass over the NWB files (no separate survey pass), because it does not need a data-driven reward-zone estimate.

ii.
```python
def convert_session(path: Path, show_processing: bool = False) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    ...
    elapsed = time.perf_counter() - t0
    session_meta["conversion_seconds"] = elapsed
```
```python
        elapsed = time.perf_counter() - total_start
        projected = elapsed / (idx + 1) * len(files)
        print(
            f"  kept {metadata['retained_trials']}/{metadata['source_trials']} trials, "
            f"{metadata['retained_cells']}/{metadata['manual_cells']} curated cells; "
            f"session {metadata['conversion_seconds']:.2f}s; projected total {projected / 60:.2f} min",
            flush=True,
        )
```
```python
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.outpicklefile} ({args.outpicklefile.stat().st_size / 1e9:.3f} GB) "
          f"in {time.perf_counter() - write_start:.2f}s", flush=True)
```

iii. CONVERSION_NOTES Step 7 gives the extrapolation the instructions asked for, explicitly scaled by session size rather than by a flat average: "Small sample sessions (plots included) 1.33–1.47 s; Representative 1,143-cell session 2.30 s; Large 2,323-cell/two-plane session 7.90 s; Full conversion estimate ~8–10 min for 152 sessions plus pickle write (below 15 min)." Step 9 confirms the realised 7.38 min, within the estimate. The timing is per-session rather than per-stage, so the breakdown among I/O, baseline and OASIS is inferred rather than measured.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised what it could and states so; the loops that remain are:
- `calculate_reference_dff`: a Python loop over trials, because the maximin baseline is defined per trial. It could be made a single pass over the session with segment-aware filtering, but at the cost of considerable complexity.
- the main per-trial loop in `convert_session`: the per-trial OASIS call is inherently sequential (a C-level call per trial), but the behavior-side work inside it (`reward_distance`, the three discretizers, the lick binarisation, `mode_value`) could all be computed once on the full session array and then sliced.
- the `outcomes` list comprehension over trials, which could be a single `np.searchsorted` of reward times into the trial-boundary timestamps.
- `mode_value`, called twice per trial, performs a `np.unique` sort on each trial slice; environment and trial number could be read as single samples or computed session-wide.
- `load_selected_roi_series` loops over plane series (at most two iterations — not worth vectorising).

What was already vectorised: the speed correlation over all cells (one matrix–vector product instead of a per-cell `np.corrcoef`, which is what the reference code does), and all discretization via boolean masks.

ii.
```python
    for trial_idx, (start, stop) in enumerate(zip(trial_starts, trial_ends)):
        f = fluorescence[:, start:stop].astype(np.float64, copy=True)
        ...
        baseline = gaussian_smooth(corrected, 15)
        baseline = ndimage.minimum_filter1d(baseline, 300, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
```
```python
        outcomes = np.array([
            np.any((reward_times >= timestamps[start]) & (reward_times < timestamps[stop]))
            for start, stop in zip(starts, stops)
        ], dtype=np.int16)
```
Already vectorised:
```python
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y)
    numerator = x_centered @ y_centered
    denominator = np.sqrt(np.sum(x_centered * x_centered, axis=1) * np.sum(y_centered * y_centered))
```
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int16)
    out[(position >= 90) & (position < 180)] = 1
    ...
```

iii. CONVERSION_NOTES Step 6: "Code speedups added: Filter manual cells before signal processing; vectorize speed correlations and all class transforms; process baseline trial-wise; OASIS only retained non-interneurons; immediately store final float32 trial events; process one session at a time; avoid image acquisition reads." Step 7: "Vectorized correlation/discretization | Removes per-cell/per-frame Python loops." The AI does not enumerate the remaining trial loops as a residual inefficiency; its implicit position is that the per-trial structure is required because the baseline, dF/F and deconvolution are all defined per trial and trials have variable length.

## 13-c. What processing does the code repeat multiple times?

i. Little at the file level — each NWB file is opened exactly once and each array read once, so the survey/convert double-read that the reference solution incurs does not occur here. The repetition that remains is small and internal:
- `gaussian_smooth` always builds a full-size `weights` array and runs a **second** Gaussian filter over it to normalise for NaNs, even though the per-trial slices it is called on never contain NaNs. This doubles the cost of both smoothing passes for no benefit.
- `mode_value` re-sorts a trial slice once for `environment` and again for `trial number`, twice per trial.
- `np.vstack` builds a 4×T copy of position/speed/lick/timestamps on every trial purely to run one `np.isfinite` check, and those same arrays are then sliced again individually.
- `plot_example`/`plot_payload` copies of the first trial's traces are made on every session regardless of whether `--show-processing` is set.
- `validate_converted` re-walks all 12,135 trials re-checking shapes and class ranges that were already asserted inside the per-trial loop.

ii.
```python
def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    nan_mask = np.isnan(values)
    clean = values.copy(); clean[nan_mask] = 0.0
    weights = np.ones(values.shape, dtype=np.float64)
    weights[nan_mask] = 0.001
    smooth = ndimage.gaussian_filter1d(clean, sigma, axis=-1)
    smooth_weights = ndimage.gaussian_filter1d(weights, sigma, axis=-1)   # never needed per-trial
    return smooth / smooth_weights
```
```python
            environment = mode_value(b["environment"][start:stop])
            trial_number = mode_value(b["trial number"][start:stop])
```
```python
            required = np.vstack((b["position"][start:stop], b["speed"][start:stop],
                                  b["lick"][start:stop], timestamps[start:stop]))
            if not np.all(np.isfinite(required)):
```
```python
            if plot_payload is None:
                plot_payload = (events[:3].copy(), {
                    "position": position.copy(), "speed": speed.copy(), "lick": lick.copy(),
                    "zone": zone, "original_trial_index": trial_idx,
                }, inputs.copy(), outputs.copy())
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Loading 92 GB of NWB assets and applying filters cell×frame is intrinsically expensive. Building full-session event matrices would also duplicate final trial storage. Code speedups added: … immediately store final float32 trial events; process one session at a time; avoid image acquisition reads." The AI's stated design principle is to avoid duplicate I/O and duplicate storage; it does not discuss the small residual recomputations above, and they are individually negligible against the measured 7.38 min runtime.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little, but not nothing:
- `load_behavior` reads the `reward_zone` and `autoreward` series into memory on every session, and **neither is ever used** — the reward zone is derived from the scene identifier instead, and `autoreward` was checked during exploration and found uninformative.
- The NaN-weight branch of `gaussian_smooth` (see 13-c) is dead work for every call in the pipeline.
- dF/F is computed for all curated cells including the 402 later dropped as putative interneurons — but this is unavoidable, since the correlation that identifies them is computed on the dF/F itself. (OASIS, the more expensive step, is correctly run only on kept cells.)
- First-trial plotting copies are materialised even when plotting is off.
- `session_info` accumulates rich provenance (per-trial exclusion reasons, plane lists, source lengths, unmapped reward counts) that the decoder never reads; this is deliberate documentation, not waste.
- `speed` is computed and stored as `output[2]`; it is required by the Decoder Task specification, so it is not discarded.

ii.
```python
def load_behavior(behavior, common_length: int) -> dict[str, np.ndarray]:
    names = ["position", "speed", "lick", "environment", "trial number", "scanning",
             "reward_zone", "autoreward", "trial_start", "teleport"]   # reward_zone, autoreward unused
    arrays = {name: np.asarray(behavior[name].data[:common_length]) for name in names}
```
```python
        dff, plot_example = calculate_reference_dff(fluorescence, neuropil, starts, stops)
        correlations = speed_correlations(dff, b["speed"])    # needs dF/F of all curated cells
        keep = ~(correlations > 0.5)
        dff = dff[keep]                                        # OASIS runs only on kept cells
```

iii. CONVERSION_NOTES Step 5 key decision 10 explains the intent behind loading extra streams: "**Unused but checked variables**: `scanning`, `trial_start`, `teleport`, `reward_zone`, `autoreward`, segmentation plane IDs, and all timestamps are used for validity/provenance checks; raw pixel masks/background/acquisition placeholder are not decoder variables." In the final script, though, only `scanning`, `trial_start` and `teleport` are actually consulted; `reward_zone` and `autoreward` are read and dropped. Step 6 states the deliberate avoidance of larger waste: "OASIS only retained non-interneurons … avoid image acquisition reads."
