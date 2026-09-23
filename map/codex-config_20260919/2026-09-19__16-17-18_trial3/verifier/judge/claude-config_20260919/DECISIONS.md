# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI:000363 NWB release, one HDF5-backed `.nwb` file per session under `/app/data/sub-<id>/`. The AI enumerates every session with a single sorted glob and opens the files **directly with `h5py`** rather than with `pynwb`, reading only the specific datasets it needs (`units/classification`, `units/anno_name`, `units/spike_times` + `spike_times_index`, `units/obs_intervals` + `obs_intervals_index`, `units/is_good_trials`, `intervals/trials/*`, `acquisition/BehavioralEvents/*`, `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`). Files are opened in three separate passes: `discover_usable_files()` opens all 174 files to find those with ≥1 classifier-good unit; `main()` opens all kept files again to build the global brain-region vocabulary; `convert_session()` opens each file a third time to do the actual conversion. Subject, session, trials and units are then read from within each file.

ii.
```python
DATA_DIR = Path("/app/data")

def discover_usable_files() -> tuple[list[str], list[dict]]:
    """Return sorted sessions with at least one classifier-good unit."""
    files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*.nwb")))
    usable, excluded = [], []
    for path in files:
        with h5py.File(path, "r") as nwb:
            classification = nwb["units/classification"][()]
            n_good = int(np.count_nonzero(classification == b"good"))
            if n_good:
                usable.append(path)
            else:
                excluded.append({...})
    return usable, excluded
```

```python
def convert_session(path: str) -> SessionResult:
    ...
    with h5py.File(path, "r") as nwb:
        classification = nwb["units/classification"][()]
        unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
        ...
        trials = nwb["intervals/trials"]
        go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
```

```python
    for number, path in enumerate(files, 1):
        result = convert_session(path)
```

iii. From CONVERSION_NOTES Step 2/Step 6: the dandiset is 174 assets / 28 mice, one NWB file per session, so the directory listing is the complete set of sessions and sorting makes the order deterministic. The AI explicitly rejected a full `pynwb` object-model load as an efficiency decision — "Loading via a full NWB object model would materialize irrelevant waveform/electrode data" — and chose "direct `h5py` access; load only small metadata/video arrays and selected ragged spike slices". It also notes the NWB release is preferred over the reference repository's intermediate MATLAB exports because "direct NWB access avoids lossy intermediate MATLAB exports" (Step 10 reference-comparison table).

## 1-b. How are the data split into subjects (mice)?

i. The subject id is parsed from the **file name** (`sub-440956_ses-...nwb` → `"440956"`), not from `general/subject/subject_id` inside the file. `subjects` is the sorted set of unique ids over the processed file list, and `subject_idx` is each session's index into that list. Result: 28 subjects, 3–10 sessions each.

ii.
```python
def subject_from_path(path: str) -> str:
    return Path(path).name.split("_")[0].removeprefix("sub-")
```

```python
    subjects = sorted({subject_from_path(path) for path in files})
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    ...
        subject_idx.append(subject_lookup[result.subject])
    ...
        "subjects": subjects, "subject_idx": np.asarray(subject_idx, dtype=np.int32),
```

iii. CONVERSION_NOTES Step 2: "There are 28 `sub-<mouse>/` directories ... Subject identifiers are numeric strings encoded in the directory and NWB subject metadata", and Step 5 mapping row: "`general/subject/subject_id` / path subject → `subjects`, `subject_idx`; deterministic sorted unique subject strings and per-session indices. Expected 28 subjects among 173 sessions." The DANDI file-naming convention derives the `sub-` prefix from the NWB subject id, so the two sources are the same string.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is done. The session id is taken from the filename up to the `_behavior` suffix (e.g. `sub-440956_ses-20190207T120657`). Session order follows the sorted file list (chronological within a subject because the filename embeds the acquisition timestamp). One session is dropped before conversion (`sub-440958_ses-20190216T162508`), leaving 173. Per-session provenance is recorded in `metadata['session_info']` and the dropped session in `metadata['excluded_sessions']`.

ii.
```python
def session_id_from_path(path: str) -> str:
    return Path(path).name.split("_behavior")[0]
```

```python
    files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*.nwb")))
```

```python
        info = {
            "session_id": session_id,
            "source_file": str(Path(path).relative_to(DATA_DIR)),
            "subject": subject_from_path(path),
            ...
        }
```

iii. CONVERSION_NOTES Step 2: "174 HDF5-backed NWB files ... each file is one recording session." Step 4 reconciles the count with the papers: "Exclude only the unlabeled/no-curated-unit session. This exactly reconciles session count" (papers report 173 behavioral sessions). Determinism is explicitly a design goal (Key Decision 10: "Sort source paths").

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`). The AI does not use every row: it first determines which contiguous block of trial-table rows is backed by ephys, by mapping each `units/obs_intervals` start time onto `trials/start_time` with `searchsorted` + `allclose`, and asserting the mapped rows are contiguous and the same width as the `is_good_trials` mask. Trial events (go cue, sample onsets) are read as global timestamp arrays and indexed by the selected trial-table row indices — i.e. the code assumes `go_start_times` has exactly one event per trial-table row (this assumption is checked indirectly by `tone_onsets < go_times`, but never asserted directly).

ii.
```python
        good_trial_matrix = np.asarray(nwb["units/is_good_trials"][unit_indices, :], dtype=bool)
        n_ephys_trials = good_trial_matrix.shape[1]
        valid_prefix = np.all(good_trial_matrix, axis=0)

        trials = nwb["intervals/trials"]
        n_trial_table = int(trials["id"].shape[0])
        ...
        obs_index = np.asarray(nwb["units/obs_intervals_index"], dtype=np.int64)
        first_unit = int(unit_indices[0])
        obs_lo = 0 if first_unit == 0 else int(obs_index[first_unit - 1])
        obs_hi = int(obs_index[first_unit])
        obs_intervals = np.asarray(nwb["units/obs_intervals"][obs_lo:obs_hi], dtype=np.float64)
        if obs_intervals.shape[0] != n_ephys_trials:
            raise ValueError(f"{session_id}: observation intervals do not match trial-mask width")
        trial_starts_table = np.asarray(trials["start_time"], dtype=np.float64)
        ephys_trial_idx = np.searchsorted(trial_starts_table, obs_intervals[:, 0])
        if (np.any(ephys_trial_idx >= n_trial_table)
                or not np.allclose(trial_starts_table[ephys_trial_idx], obs_intervals[:, 0])):
            raise ValueError(f"{session_id}: cannot map observation intervals to trial table")
        if np.any(np.diff(ephys_trial_idx) != 1):
            raise ValueError(f"{session_id}: ephys-backed trial rows are not contiguous")
```

```python
        go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
        go_times = go_all[selected_trial_idx]
```

iii. CONVERSION_NOTES Step 4 (discrepancy table): "In 9 NWBs, `is_good_trials.shape[1]` is shorter than `intervals/trials` (by 5–376 trials). Unit `obs_intervals` map the 93,310 neural trials to the exact contiguous trial-table block; one session is a suffix (rows 125–629), while the others begin at row 0 ... Map every neural-mask column to the trial table through exact observation-interval start times. Discard the 1,060 behavioral-only rows outside those blocks." Step 10, Iteration 2 records this as a bug the AI found and fixed: "Prefix slicing would silently attach wrong behavior to spikes."

## 1-e. How are trials filtered based on quality controls?

i. Four successive filters, all documented as data-validity (not behavioural) filters:
   1. Rows outside the ephys observation block are dropped (1,060 rows).
   2. Rows where **any** retained classifier-good unit has `is_good_trials == False` are dropped (509 rows).
   3. `auto_water` **or** `free_water` trials are dropped (3,731 rows).
   4. Trials whose whole curated population has zero spikes over the entire 4 s window are dropped (2 rows) — a post-hoc no-coverage guard applied after binning.
   A session must retain ≥2 trials (otherwise `ValueError`). Early-lick, `ignore`/no-response and photostimulation trials are deliberately **kept** because they are required decoder labels/inputs. Net: 94,990 source rows → 89,068 converted trials.

ii.
```python
        candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
        auto_water = np.asarray(trials["auto_water"])[candidate_trial_idx] != 0
        free_water = np.asarray(trials["free_water"])[candidate_trial_idx] != 0
        water_trial = auto_water | free_water
        selected_trial_idx = candidate_trial_idx[~water_trial]
        if selected_trial_idx.size < 2:
            raise ValueError(f"{session_id}: fewer than two valid non-water trials")
```

```python
        # A small number of NWB trial-mask prefixes extend one trial beyond the
        # actual spike recording. ... a completely zero population over four
        # seconds is a definitive no-coverage marker, not a plausible silent trial.
        zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
        n_zero_spike_trials = int(zero_spike_trials.sum())
        if n_zero_spike_trials:
            keep = ~zero_spike_trials
            neural_rates = neural_rates[keep]; inputs = inputs[keep]; outputs = outputs[keep]
            ...
        if selected_trial_idx.size < 2:
            raise ValueError(f"{session_id}: fewer than two trials with neural coverage")
```

iii. CONVERSION_NOTES Step 3/4: the data paper excludes early-lick and no-response trials and the method paper's `get_regular_trial_mask` additionally excludes photoinhibition and auto/free-water trials, but "This decoder explicitly requires early-lick, ignore/no-lick, and photostimulation variables, so excluding them would destroy required target/input classes; retain them unless raw timing/validity is unusable." For water trials: "exclude auto/free-water trials because those variables are not requested, outcome semantics are experimentally altered, and the reference explicitly removes them" (Step 4). For `is_good_trials`: "`is_good_trials` is explicitly described in NWB as a manual per-probe/per-unit trial-validity annotation ... Honor the NWB validity annotation and retain a rectangular session population by dropping trials where any retained curated neuron is invalid. This is conservative and required by the instruction to exclude invalid periods." The all-zero guard came from Step 10 Iteration 1 (the sample verifier flagged an all-zero boundary trial).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, global session clock) indexed by `units/spike_times_index`, restricted to the units selected by `units/classification == b"good"`; the go-cue times come from `acquisition/BehavioralEvents/go_start_times/timestamps`. `units/anno_name` supplies the per-neuron brain-region label. No other neural representation is used (this is extracellular ephys, so no dF/F).

ii.
```python
        neural_rates = bin_selected_units(
            nwb["units/spike_times"], np.asarray(nwb["units/spike_times_index"]), unit_indices, go_times
        )
```

```python
    for out_unit, raw_unit in enumerate(unit_indices):
        lo = 0 if raw_unit == 0 else int(spike_index[raw_unit - 1])
        hi = int(spike_index[raw_unit])
        spikes = np.asarray(spike_times_ds[lo:hi], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 1: "This is extracellular electrophysiology, so delta-F/F is not applicable. The native neural variable is per-unit spike times." Step 2 notes that, unlike the MATLAB export consumed by the reference repo, "NWB spike times are not stored per trial or pre-aligned; they must be sliced by `go_start_times`."

## 2-b. How is the `neural` data processed?

i. Spike times of each good unit are converted to per-bin **firing rates in Hz**. For each unit, every spike is assigned to a trial by `searchsorted` over the (checked non-overlapping) trial window starts, converted to a go-relative time, restricted to `[-2.5, 1.5)`, floored into one of 80 50-ms bins, and accumulated with a single `np.bincount` over the flattened (trial, bin) index. Counts are divided by 0.05 s. Stored as `float32`, shape (n_neurons, 80) per trial. No smoothing, z-scoring, baseline subtraction or rate threshold is applied.

ii.
```python
def bin_selected_units(spike_times_ds, spike_index, unit_indices, go_times):
    """Assign each selected-unit spike to its non-overlapping trial/bin."""
    window_starts = go_times + OFF_START
    window_ends = go_times + OFF_END
    if np.any(window_starts[1:] < window_ends[:-1]):
        raise ValueError("requested trial windows overlap")
    n_trials = go_times.size
    rates = np.zeros((n_trials, unit_indices.size, N_TIME), dtype=np.float32)
    for out_unit, raw_unit in enumerate(unit_indices):
        ...
        trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
        possible = (trial_idx >= 0) & (trial_idx < n_trials)
        spikes, trial_idx = spikes[possible], trial_idx[possible]
        rel = spikes - go_times[trial_idx]
        inside = (rel >= OFF_START) & (rel < OFF_END)
        rel, trial_idx = rel[inside], trial_idx[inside]
        bin_idx = np.floor((rel - OFF_START) / BIN_SIZE_S).astype(np.int64)
        good_bin = (bin_idx >= 0) & (bin_idx < N_TIME)
        flat_idx = trial_idx[good_bin] * N_TIME + bin_idx[good_bin]
        counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_SIZE_S
    return rates
```

iii. CONVERSION_NOTES Step 1 identifies the reference's `sliding_histogram` as counting spikes in half-open `[center-width/2, center+width/2)` bins and dividing by bin width to produce Hz. Step 4: "Decoder specification overrides width/stride: use 80 adjacent 50-ms half-open bins spanning `[-2.5, 1.5)` and divide counts by 0.05 s to Hz. This preserves reference counting/scaling conventions." Key Decision 4: "Values are firing rates in Hz, float32, never smoothed or z-scored." The metadata field `spike_bin_convention` records "left-closed, right-open non-overlapping bins".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == b"good"` are kept — the verdict of the spike-sorting QC classifier — with **no** thresholds on any individual QC metric and **no** application of the method paper's 2-Hz analysis threshold. `units/unit_quality` (the Kilosort `good`/`multi` label) is not used. The AI additionally requires that every retained unit have a non-empty, non-`nan` Allen CCF `anno_name`; this is implemented as a hard `ValueError` (validation), not as a filter. A session with zero classifier-good units is excluded up front. Result: 69,453 units over 173 sessions, mean 401.5 / median 390 per session.

ii.
```python
        classification = nwb["units/classification"][()]
        unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
        region_names = [decode_text(x).strip() for x in nwb["units/anno_name"][unit_indices]]
        if any((not x) or x.lower() == "nan" for x in region_names):
            raise ValueError(f"{session_id}: curated unit lacks a CCF annotation")
```

```python
            n_good = int(np.count_nonzero(classification == b"good"))
            if n_good:
                usable.append(path)
            else:
                excluded.append({"session_id": ..., "reason": "no classifier-good units / missing classifier labels", ...})
```

iii. CONVERSION_NOTES Step 3 neuron-curation rules: "Kilosort2 clusters were characterized by 15 QC metrics. Five region-specific logistic classifiers ... produced the final unit list. Individual metrics overlap strongly, and the white paper explicitly rejects simple one-metric thresholds. Use the NWB `classification == "good"` label." Step 3 also rejects the 2-Hz cut: "the method paper excludes firing-rate neurons below 2 Hz for video-to-neural prediction, but ... the 2-Hz analysis-specific threshold is not automatically applicable." Step 4 documents the 69,453 vs 69,943 gap against the paper: "The embedded release labels are authoritative for reproducibility ... Use 69,453, not an invented metric threshold."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so alignment is just arithmetic: each trial's go-cue timestamp is taken from `go_start_times`, the fixed relative bin grid is added to it to give absolute bin centers/windows, and spikes are expressed as `spike_time - go_time` before binning. No resampling, interpolation, or per-stream offset correction is applied, and the same go-relative grid is reused for the inputs and the tongue output.

ii.
```python
        go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
        go_times = go_all[selected_trial_idx]
        centers_global = go_times[:, None] + BIN_CENTERS[None, :]
```

```python
    window_starts = go_times + OFF_START
    window_ends = go_times + OFF_END
    ...
        rel = spikes - go_times[trial_idx]
        inside = (rel >= OFF_START) & (rel < OFF_END)
```

iii. CONVERSION_NOTES Step 2: "Raw spike timestamps and behavioral/video timestamps share the NWB session clock." Step 10 reference-comparison: "reference spike arrays are go-cue-relative ... [ours] uses NWB absolute go timestamps, then subtracts/adds fixed relative edges — equivalent go-cue alignment, independently verified on raw timestamps." Metadata: `temporal_alignment_event = "auditory go cue onset"`, `off_start = -2.5`, `off_end = 1.5`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, 80 non-overlapping bins spanning `[-2.5, +1.5)` s relative to the go cue, with centers `-2.475 … +1.475` s. The grid is built once at module level from `np.linspace(-2.5, 1.5, 81)` and reused for every trial, session, and data stream, so all trials have exactly 80 timepoints. This is a deliberate departure from the reference pipeline's overlapping 40-ms/3.4-ms sliding window — the spikes themselves are re-binned from raw spike times directly into the 50 ms grid (no two-stage rebinning). `metadata['time_bin_size'] = 50.0` ms.

ii.
```python
OFF_START, OFF_END, BIN_SIZE_S = -2.5, 1.5, 0.050
BIN_EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = BIN_CENTERS.size
```

```python
        bin_idx = np.floor((rel - OFF_START) / BIN_SIZE_S).astype(np.int64)
```

iii. Key Decision 3: "Edges are `np.arange(-2.5, 1.5 + 0.05, 0.05)` (81 edges), producing exactly 80 timepoints at centers `-2.475 ... 1.475` s. All neural/input/tongue streams use these centers/windows." Step 4: "Decoder specification overrides width/stride ... This preserves reference counting/scaling conventions." The AI verified in Step 10 that "the first/last centers are exactly −2.475/+1.475 s" and that all rates are non-negative multiples of 20 Hz (integer counts / 0.05 s).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample/tone onsets) together with `intervals/trials/start_time` and the trial's go cue. Crucially, the AI takes the **first** sample-start event at or after the trial's start time — i.e. the first tone of the trial — not the last tone before the go cue. It validates that this event precedes the go cue.

ii.
```python
        sample_starts = np.asarray(nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"])
        first_sample_idx = np.searchsorted(sample_starts, trial_starts, side="left")
        if np.any(first_sample_idx >= sample_starts.size):
            raise ValueError(f"{session_id}: missing sample-start event")
        tone_onsets = sample_starts[first_sample_idx]
        if np.any(tone_onsets >= go_times):
            raise ValueError(f"{session_id}: first sample-start not before go")
```

iii. CONVERSION_NOTES Step 4: "Every ephys-backed trial has at least one `sample_start_times` event in `[trial start, go)`. Early trials often have 2–14; most no-early trials have one ... Define tone onset as the first sample-start event in the behavioral trial, matching 'time from tone onset' rather than the final replay. This intentionally makes elapsed time informative on replay/early-lick trials." Step 3 explains the mechanism: "Early licking can trigger replay of the sample/delay sequence, explaining repeated raw event timestamps." Metadata field: `tone_onset_definition = "first sample_start event within the behavioral trial"`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of every bin center is computed (`go + bin_center`) and the trial's tone-onset timestamp is subtracted, giving a continuous, monotonically increasing ramp in seconds that advances by exactly 0.05 s per bin. Stored as `float32` in row 0 of the `(2, 80)` input array. No clipping, normalisation, or binarisation is applied; the observed range over the full dataset is `[-1.5, 11.9]` s (the large positive values are early-lick replay trials).

ii.
```python
        centers_global = go_times[:, None] + BIN_CENTERS[None, :]
        ...
        elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
        ...
        inputs = np.stack((elapsed_from_tone, stim_on), axis=1).astype(np.float32)
```

iii. Step 5 mapping table: "For every 50-ms bin center, compute global bin-center time minus that trial's first tone/sample onset, in seconds (float32). Continuous time-varying input explicitly requested; unlike the generic format example, this is elapsed time, not an onset indicator." Step 9 consistency table: "[-1.5, 11.9] s over bin centers — Yes; long positive values are early-lick replay trials." Step 12 diagnostic 3: "Tone time ramps by exactly 50 ms/bin."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same go-cue-anchored bin grid used for the firing rates — `centers_global = go_times[:, None] + BIN_CENTERS[None, :]` — so bin *k* of the input covers the same interval as bin *k* of the neural array by construction. Camera, event and spike timestamps all live on the same NWB session clock, so no interpolation or offset correction is needed.

ii.
```python
        centers_global = go_times[:, None] + BIN_CENTERS[None, :]
        ...
        elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
```

```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2   # shared by neural, inputs, tongue
```

iii. Key Decision 3: "All neural/input/tongue streams use these centers/windows." Step 10 Check 3 verified this independently from the raw NWB files: "independently locate first sample onset and stimulation interval in raw NWB; verify `np.allclose` for elapsed-time and photostim vectors", reported as passing.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the paired event streams `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `.../photostim_stop_times/timestamps`, used directly on the global clock. The AI deliberately does **not** use the trials-table columns `photostim_onset` / `photostim_duration` (which are strings with `'N/A'` on unstimulated trials), having verified the two agree.

ii.
```python
        stim_starts = np.asarray(nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"])
        stim_stops = np.asarray(nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"])
        stim_on = photostim_state(centers_global, stim_starts, stim_stops)
```

iii. CONVERSION_NOTES Step 4: "NWB global start/stop events match trial columns; example starts occur at −1.2 s and stop at −0.7 s relative to go ... Use the NWB start/stop event pairs directly on the global clock and sample binary state at each decoder bin center." Step 2 records that the trials table stores "photostimulation onset/duration/power (`N/A` on unstimulated trials)", and Step 3 that photoinhibition occupies "the final 0.5 s of delay and ends before go" in ~25% of trials in 17 VGAT mice.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, stored `float32`) time series: for each bin center, `searchsorted` finds the most recent stimulation onset and the bin is marked 1 if the center falls in the half-open interval `[start, stop)` of that event, 0 otherwise. Sessions with no stimulation events return an all-zero array. The number of starts and stops must match or the session raises.

ii.
```python
def photostim_state(centers_global, starts, stops):
    if starts.size == 0:
        return np.zeros(centers_global.shape, dtype=np.float32)
    if starts.size != stops.size:
        raise ValueError(f"photostim starts/stops mismatch: {starts.size} vs {stops.size}")
    flat = centers_global.ravel()
    interval_idx = np.searchsorted(starts, flat, side="right") - 1
    safe_idx = np.clip(interval_idx, 0, starts.size - 1)
    on = (interval_idx >= 0) & (flat >= starts[safe_idx]) & (flat < stops[safe_idx])
    return on.reshape(centers_global.shape).astype(np.float32)
```

iii. Step 5 mapping table: "Binary float32 state at each global bin center, using half-open `[start, stop)` intervals. Retain stimulated trials because photostimulation is a decoder input." This follows the instruction that a time input should be represented as a binary time series; the half-open convention matches the neural binning convention. Metadata: `photostimulation_definition = "binary state at bin center from NWB start/stop events"`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation intervals stay on the absolute session clock and are queried at the *same* absolute bin centers `centers_global` that define the neural bins, so alignment is automatic and identical to the neural grid. Plot panel 8 of `--show-processing` overlays the session-mean firing rate against the fraction of trials with stimulation on, as an alignment sanity check.

ii.
```python
        centers_global = go_times[:, None] + BIN_CENTERS[None, :]
        ...
        stim_on = photostim_state(centers_global, stim_starts, stim_stops)
```

```python
    ax[7].plot(BIN_CENTERS, payload["neural"].mean(axis=(0, 1)), color="tab:green")
    ax7 = ax[7].twinx(); ax7.fill_between(BIN_CENTERS, 0, payload["stim_on"].mean(axis=0), ...)
```

iii. Step 7 plot review: "photostimulation occupies late delay and ends before go ... No alignment anomaly was seen", consistent with the paper's description of photoinhibition in the final 0.5 s of the delay.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not stored in the file, so it is derived from two trials-table columns: `trial_instruction` (`left`/`right`) and `outcome` (`hit`/`miss`/`ignore`). The AI cross-validated this derivation against the actual lick event streams before adopting it.

ii.
```python
        outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
        instructions = np.asarray(trials["trial_instruction"])[selected_trial_idx]
```

```python
def map_choice(instructions, outcomes):
    result = np.empty(outcomes.size, dtype=np.int8)
    for i, (instruction_raw, outcome_raw) in enumerate(zip(instructions, outcomes)):
        instruction, outcome = decode_text(instruction_raw), decode_text(outcome_raw)
        if outcome == "ignore":
            result[i] = 2
        elif outcome == "hit":
            result[i] = 0 if instruction == "left" else 1
        elif outcome == "miss":
            result[i] = 1 if instruction == "left" else 0
        else:
            raise ValueError(f"unexpected outcome {outcome!r}")
    return result
```

iii. CONVERSION_NOTES Step 4: "Instruction + outcome implies choice and agrees with the first directional lick in the 1.5-s answer period on 92,976 / 93,310 ephys-backed trials (99.64%); event discrepancies include tracker/event omissions and occasional rapid opposite licks ... Use the task-authoritative mapping: hit→instruction side, miss→opposite side, ignore→no lick. It is complete, matches paper definitions, and is more robust than selecting the first lick event."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, as `int8`, written into row 0 of a `(4, 80)` per-trial output array and repeated (broadcast) across all 80 bins so that the per-trial labels can share one dense array with the time-varying tongue output. `output_values[0] = ["left", "right", "no lick"]`. Full-data distribution: 0.429 / 0.422 / 0.149.

ii.
```python
        outputs = np.empty((selected_trial_idx.size, 4, N_TIME), dtype=np.int8)
        outputs[:, 0, :] = map_choice(instructions, outcomes)[:, None]
```

```python
        "output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
        "output_values": [
            ["left", "right", "no lick"], ["ignore", "miss", "hit"], ["no", "yes"], [...],
        ],
```

iii. Key Decision 5: "Store every trial output as a `(4, 80)` integer array. Per-trial choice/outcome/early labels are repeated across time so the time-varying tongue output can coexist in the required dense representation." Step 10 Check 3 verified "per-trial labels are constant across time" over all 89,068 trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of `intervals/trials`, which already contains exactly the three requested categories (`ignore`, `miss`, `hit`).

ii.
```python
        outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
```

iii. Step 2: the trials table has "outcome (`hit`/`miss`/`ignore`)"; Step 5 mapping table: "Direct categorical mapping: ignore 0, miss 1, hit 2; repeat across time. Preserve all three required classes." No derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The byte strings are decoded and mapped through a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit` (matching the order given in the instructions), stored as `int8` in row 1 and repeated across all 80 bins. Full-data distribution: 0.149 / 0.166 / 0.685 (i.e. 68.5% hit, versus the paper's 84% correct on its much more restricted regular-trial subset).

ii.
```python
def map_outcome(outcomes):
    code = {"ignore": 0, "miss": 1, "hit": 2}
    return np.asarray([code[decode_text(x)] for x in outcomes], dtype=np.int8)
```

```python
        outputs[:, 1, :] = map_outcome(outcomes)[:, None]
```

iii. Step 9 consistency table: "84% correct only for selected control/non-early analysis trials ... [ours] 0.149/0.166/0.685 — Consistent given broader retained trials", i.e. the difference from the paper's reward rate is explained by deliberately retaining early-lick, ignore and photostimulation trials.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of `intervals/trials`, which holds the strings `no early` / `early`.

ii.
```python
        early = np.asarray(trials["early_lick"])[selected_trial_idx]
```

iii. Step 5 mapping table: "`early_lick` → `output[..., 2, :]`; `no early`→0, `early`→1; repeat across time. Preserve rather than applying paper's exclusion." Step 3 records that the data paper excluded early-lick trials from its analyses, which the AI explicitly declines to do because early lick is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Decoded and mapped through a fixed dictionary to `0 = no`, `1 = yes`, stored as `int8` in row 2 and repeated across all 80 bins. Full-data distribution: 0.884 / 0.116.

ii.
```python
def map_early(early):
    code = {"no early": 0, "early": 1}
    return np.asarray([code[decode_text(x)] for x in early], dtype=np.int8)
```

```python
        outputs[:, 2, :] = map_early(early)[:, None]
```

iii. Step 12 gives extra justification for keeping the class: "the class has adequate prevalence (11.6%), its three raw-trial spot checks match exactly, its sample/full validation accuracies agree (0.7568/0.7567)". The source distribution (10,805 early / 84,185 no-early) was recorded in Step 2 and matches.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` is `(n_frames, 3)` = tongue x, tongue y, DeepLabCut likelihood, with matching global `timestamps` (~294 Hz nominal, 3.4 ms inter-frame). Column 1 supplies the value, column 2 decides visibility, and column 0 is used only in the 2-D velocity-outlier computation.

ii.
```python
        tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
        tracking_ts = np.asarray(tracking["timestamps"], dtype=np.float64)
        tracking_data = np.asarray(tracking["data"], dtype=np.float64)
        clean_y, visible, outlier, q40, q60, tongue_summary = clean_tongue_tracking(tracking_ts, tracking_data)
```

iii. Step 2: "`(frames, 3)` float64 columns documented as tongue x, tongue y, and DeepLabCut likelihood, with global float64 timestamps. Camera0 tongue/jaw/nose tracking exists in all 174 sessions ... Tongue coordinates are always numeric even when tracking confidence is essentially zero, so visibility must be determined from the likelihood column rather than finite/NaN coordinates."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps, in `clean_tongue_tracking` + `sample_tongue_categories`:
   1. **Visibility mask**: a frame counts as visible only if `likelihood >= 0.9` (and its timestamp/coordinates are finite).
   2. **Velocity-outlier cleaning** (from the method paper): 2-D speed is computed between consecutive visible frames ≤10 ms apart; frames whose speed exceeds mean + 5 SD are flagged and their x and y are replaced by linear interpolation from the surrounding clean visible frames. Low-likelihood frames are *not* imputed — they stay invisible.
   3. **Per-session percentiles**: the 40th and 60th percentiles of the cleaned **visible raw frame** y-values over the whole session.
   4. **Sample-and-hold to the bin grid**: for each bin center, the last raw frame at or before that center is taken (tolerance 5.1 ms ≈ 1.5 frames); no averaging within the bin is performed.

ii.
```python
DLC_VISIBLE_THRESHOLD = 0.9

def clean_tongue_tracking(timestamps, data):
    """Apply confidence masking and the paper's five-sigma velocity cleaning."""
    xy = np.asarray(data[:, :2], dtype=np.float64).copy()
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    finite = np.isfinite(timestamps) & np.all(np.isfinite(xy), axis=1)
    visible = finite & (likelihood >= DLC_VISIBLE_THRESHOLD)
    dt = np.diff(timestamps); dxy = np.diff(xy, axis=0)
    valid_velocity = (dt > 0) & (dt <= 0.010) & visible[:-1] & visible[1:]
    speed = np.full(timestamps.shape, np.nan, dtype=np.float64)
    speed[1:][valid_velocity] = np.linalg.norm(dxy[valid_velocity], axis=1) / dt[valid_velocity]
    velocity_values = speed[np.isfinite(speed)]
    if velocity_values.size:
        velocity_threshold = float(velocity_values.mean() + 5.0 * velocity_values.std())
        outlier = visible & (speed > velocity_threshold)
    ...
    # Only high-confidence velocity outliers are imputed. Low-confidence frames
    # remain invisible and therefore become output class 3.
    for dim in range(2):
        xy[outlier, dim] = np.interp(timestamps[outlier], timestamps[base_valid], xy[base_valid, dim])
    clean_y = xy[:, 1]
    q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
```

```python
def sample_tongue_categories(centers_global, timestamps, clean_y, visible, q40, q60):
    """Use the final raw frame at/before each center, as in reference alignment."""
    flat = centers_global.ravel()
    idx = np.searchsorted(timestamps, flat, side="right") - 1
    safe_idx = np.clip(idx, 0, timestamps.size - 1)
    age = flat - timestamps[safe_idx]
    near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
    is_visible = near & visible[safe_idx]
```

iii. Step 3/Step 4: the method paper "rejected marker velocity outliers beyond five standard deviations and imputed nearby frames; when tongue was occluded/retracted, it replaced tongue position with its mean. That imputation supported continuous movement regression, whereas this task explicitly requires a categorical 'not visible' state, so raw DLC likelihood must be preserved and used instead of mean imputation." Key Decision 6: "A DLC likelihood cutoff of 0.9 is justified by the strongly bimodal source distribution and conventional high-confidence DLC use. Percentiles are calculated only over visible, cleaned y samples across the whole session; invisible coordinates must not contaminate thresholds." Key Decision 7: "Follow the reference marker alignment's sample-and-hold convention (last raw frame at/before a requested time), with a one-frame tolerance. Do not average visible and invisible samples inside a 50-ms bin because that would blur the required not-visible class." (The reference code `align_markers_between_lims` does take "the embedding at the last time point within the time range", on a 3.4 ms grid.)

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the 4-class scheme specified in the instructions, with per-session percentile edges: `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, and `3` if the sampled frame is not visible (low likelihood) or no frame falls within the tolerance window. `q40`/`q60` are per-session and are recorded in `metadata['session_info']`. Full-data distribution: 0.062 / 0.032 / 0.065 / 0.841 — i.e. among visible bins, 39 % / 20 % / 41 %, as expected.

ii.
```python
    categories = np.full(flat.shape, 3, dtype=np.int8)
    y = clean_y[safe_idx]
    categories[is_visible & (y < q40)] = 0
    categories[is_visible & (y >= q40) & (y <= q60)] = 1
    categories[is_visible & (y > q60)] = 2
    return categories.reshape(centers_global.shape), safe_idx.reshape(centers_global.shape)
```

```python
        outputs[:, 3, :] = tongue_category
```

iii. Step 5 mapping table: "compute session 40th/60th percentiles from cleaned visible y frames ... Codes: y<q40→0, q40≤y≤q60→1, y>q60→2, low confidence/no nearby frame→3. Raw low-confidence coordinates are finite but invalid; do not mean-impute occlusion because class 3 is explicitly required." Step 5 planned check: "tongue visible classes approximate 40%/20%/40% over visible session frames (trial-center sampling can shift these fractions)"; Step 9 confirms the realised fractions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and the go cue, so the tongue output is evaluated at the identical absolute bin centers `centers_global = go + BIN_CENTERS` used for the firing rates. Each bin takes the last camera frame at or before its center, with a 5.1 ms staleness tolerance; when the video is off (e.g. the inter-trial interval, or before the trial-gated clip starts) no frame is within tolerance and the bin falls into class 3. No interpolation or resampling onto a new time base is used.

ii.
```python
        centers_global = go_times[:, None] + BIN_CENTERS[None, :]
        ...
        tongue_category, tongue_frame_idx = sample_tongue_categories(
            centers_global, tracking_ts, clean_y, visible, q40, q60
        )
```

```python
    idx = np.searchsorted(timestamps, flat, side="right") - 1
    age = flat - timestamps[safe_idx]
    near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
```

iii. Step 10 reference-comparison: "video functions align to `go_times`, generally taking the last nearby frame ... [ours] tongue uses last frame at/before each center within 5.1 ms — equivalent go-cue alignment, independently verified on raw timestamps." Step 10 Check 5 lists "video-frame tolerance at bin edges" among the boundary cases explicitly verified, and Check 2 reports an independent raw-file `np.allclose` comparison of "all 80 tongue classes" and the session quantiles.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct cases, handled by exclusion where nothing was recorded and by an explicit category where the measurement legitimately has no value:
   - **Session with no QC labels** (`sub-440958_ses-20190216T162508`, 1,852 units with `classification`/`anno_name` = NaN): detected in `discover_usable_files()` and excluded before conversion, and recorded in `metadata['excluded_sessions']`.
   - **Behavioural trials with no ephys**: excluded via the `obs_intervals` → trial-table mapping (1,060 rows).
   - **Per-unit invalid trials**: excluded via `is_good_trials` (509 rows).
   - **Irregular reward trials**: `auto_water`/`free_water` excluded (3,731 rows).
   - **Residual no-coverage trials**: trials whose entire curated population is silent for the full 4 s window are dropped (2 rows).
   - **Untracked tongue**: low-likelihood or absent frames are not imputed; they become output class 3.
   - Everything else is treated as fail-fast: unexpected outcome strings, missing sample events, a tone not preceding the go cue, mismatched photostim start/stop counts, non-contiguous ephys blocks, missing CCF annotations, and non-finite converted values all raise `ValueError`/`AssertionError` and abort the run rather than silently dropping a session.

ii.
```python
        if any((not x) or x.lower() == "nan" for x in region_names):
            raise ValueError(f"{session_id}: curated unit lacks a CCF annotation")
```

```python
        zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
        ...
        if selected_trial_idx.size < 2:
            raise ValueError(f"{session_id}: fewer than two trials with neural coverage")
```

```python
    if not np.all(np.isfinite(neural_rates)) or not np.all(np.isfinite(inputs)):
        raise AssertionError(f"{session_id}: non-finite converted values")
```

```python
    # in-memory validation of every session/trial before pickling
def validate_in_memory(data: dict) -> None:
    ...
            assert np.all((y[0] >= 0) & (y[0] <= 2)) and np.all((y[1] >= 0) & (y[1] <= 2))
            assert np.all((y[2] >= 0) & (y[2] <= 1)) and np.all((y[3] >= 0) & (y[3] <= 3))
```

iii. Step 4/Step 10: "Exclude only the unlabeled/no-curated-unit session. This exactly reconciles session count." Step 10 Iteration 1: "the first sample verifier exposed an all-population-zero boundary trial with no usable spike coverage. Added a coverage guard." The final accounting is given explicitly: "94,990 raw trial-table rows − 620 rows in the unlabeled session − 1,060 behavioral-only rows outside ephys observation blocks − 509 per-unit-invalid trials − 3,731 auto/free-water trials − 2 all-zero coverage edges = 89,068 converted trials." For the tongue, Step 4: "low-likelihood frames must map to class 3 ('not visible'), not be mean-imputed."

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented and reported per-session and serialization timing. The full conversion of 173 sessions took **148.64 s** (130.42 s of session work + 14.56 s pickling 10.83 GiB); per-session times ranged from ~0.27 s to ~1.7 s and scale with unit count (e.g. 787 neurons → 1.72 s). Within a session the dominant costs are the HDF5 reads (the ragged `spike_times` slices and the ~680k × 3 tongue-tracking array) and the per-unit Python loop in `bin_selected_units`. The two pre-passes over all 174 files (`discover_usable_files` and the brain-region vocabulary loop) add further I/O that the AI did not time separately. The AI chose `h5py` specifically to avoid materialising waveform/electrode data.

ii.
```python
def convert_session(path: str) -> SessionResult:
    start_clock = time.perf_counter()
    ...
        elapsed = time.perf_counter() - start_clock
```

```python
    write_clock = time.perf_counter()
    with output_path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_clock; total_seconds = time.perf_counter() - total_clock
    print(f"Saved {output_path} ({output_path.stat().st_size / 2**30:.3f} GiB) in {write_seconds:.2f} s; "
          f"total conversion {total_seconds:.2f} s", flush=True)
```

iii. Step 6: "Loading via a full NWB object model would materialize irrelevant waveform/electrode data. Per-trial/per-unit histogram calls would also multiply Python overhead across approximately 69k units and 93k trials. Storing float64 firing rates would double the dominant output size." Step 7 estimated "approximately 2–3 minutes, well below 15 minutes", which the full run confirmed (148.64 s), so no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops that remain are:
   - the per-unit loop in `bin_selected_units` — inherent to the ragged `spike_times` storage (each unit is a separate contiguous slice); all trials and bins are already handled in one vectorised `searchsorted` + `bincount` per unit;
   - the `for dim in range(2)` interpolation loop in `clean_tongue_tracking` (2 iterations, trivial);
   - the per-trial Python loops in `map_choice`, `map_outcome`, `map_early` and the `region_names` list comprehension — these iterate over trials/units and could be replaced by vectorised `np.where`/lookup-table indexing, but cost microseconds relative to the I/O;
   - the final list comprehensions that slice the session arrays into per-trial views.
   Notably, the tongue discretisation is fully vectorised across all trials and bins at once (no per-trial loop), and `photostim_state` likewise.

ii.
```python
    for out_unit, raw_unit in enumerate(unit_indices):
        ...
        trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
        ...
        counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_SIZE_S
```

```python
def map_choice(instructions, outcomes):
    result = np.empty(outcomes.size, dtype=np.int8)
    for i, (instruction_raw, outcome_raw) in enumerate(zip(instructions, outcomes)):
        ...
```

```python
    flat = centers_global.ravel()               # tongue: all trials x bins at once
    idx = np.searchsorted(timestamps, flat, side="right") - 1
```

iii. Step 6 speedups: "assign all spikes from one unit to non-overlapping trial windows with vectorized `searchsorted` + `bincount`; precompute all timing grids". The per-unit loop cannot be collapsed further because each unit has a different number of spikes stored in one ragged buffer, so there is no single sorted array to search against.

## 10-c. What processing does the code repeat multiple times?

i. Every NWB file is opened and partially re-read up to **three** times: `units/classification` is read in `discover_usable_files()`, again in the `main()` region-vocabulary loop, and a third time inside `convert_session()`; `units/anno_name` is read twice (vocabulary loop and conversion). `discover_usable_files()` scans all 174 files even in `--sample` mode, where only 2 are converted. Within a session nothing is recomputed: the bin grid is built once at module level, the tongue cleaning/percentiles are computed once per session, and the spike buffer slice for each unit is read once. The `plot_payload` dict is assembled on every session even when `--show-processing` is off.

ii.
```python
def discover_usable_files():                       # pass 1 over all 174 files
    for path in files:
        with h5py.File(path, "r") as nwb:
            classification = nwb["units/classification"][()]
```

```python
    for path in files:                             # pass 2 over all kept files
        with h5py.File(path, "r") as nwb:
            good = nwb["units/classification"][()] == b"good"
            region_set.update(decode_text(x).strip() for x in nwb["units/anno_name"][good])
    brain_regions = sorted(region_set)
```

```python
    for number, path in enumerate(files, 1):       # pass 3: actual conversion
        result = convert_session(path)
```

iii. The AI does not flag this redundancy in CONVERSION_NOTES; its stated efficiency decisions are "stream one HDF5 session at a time" and "avoid loading waveforms or raw uncurated spikes unnecessarily" (Key Decision 10), both of which hold. The two pre-passes exist to (a) exclude un-QC'd sessions before conversion and report them in `metadata['excluded_sessions']`, and (b) build a deterministic global `brain_regions` vocabulary before any session is assembled. Each pre-pass reads only the small `classification`/`anno_name` string datasets, so the measured cost stayed inside the 15-minute budget.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few small items:
   - **`plot_payload` is built for every session** (all 173) although it is only consumed for the first 2 sessions under `--show-processing`; it is otherwise discarded when `result` is rebound on the next loop iteration.
   - **Tongue x is cleaned but never used as an output** — the velocity-outlier interpolation loop runs over both x and y, and the 2-D speed uses x, but only `clean_y` reaches the output. (The x channel does genuinely feed the 5-SD speed statistic, so only the interpolation of x is wasted.)
   - **`tongue_frame_idx`** is computed and returned on every session purely for the diagnostic plot.
   - **Large per-session provenance in metadata**: `retained_source_trial_indices_zero_based` stores all 89,068 retained row indices as Python ints, and `time_bin_centers_seconds` duplicates a derivable grid; neither is read by the decoder.
   - `discover_usable_files()` scanning all sessions in `--sample` mode (see 10-c).
   Everything else that is computed (neural, inputs, outputs, subject/region indices, metadata) is written to the pickle and used.

ii.
```python
        plot_payload = {
            "session_id": session_id, ..., "tracking_ts": tracking_ts,
            "tracking_y_raw": tracking_data[:, 1], "tracking_likelihood": tracking_data[:, 2],
            "tracking_y_clean": clean_y, "tracking_frame_idx": tongue_frame_idx, ...
        }
```

```python
        if args.show_processing and number <= 2:
            plot_path = Path("/app") / f"processing_{result.info['session_id']}.png"
            plot_processing(result.plot_payload, plot_path)
```

```python
    for dim in range(2):                     # x is interpolated but only y is used
        xy[outlier, dim] = np.interp(timestamps[outlier], timestamps[base_valid], xy[base_valid, dim])
```

```python
            "retained_source_trial_indices_zero_based": selected_trial_idx.astype(int).tolist(),
            ...
            "n_timepoints": int(N_TIME), "time_bin_centers_seconds": BIN_CENTERS.tolist(),
```

iii. The AI's stated rationale for the extra bookkeeping is provenance and auditability — Key Decision 9 requires metadata to "include units, bin convention, categorical codebooks, tongue rules, curation summary, and per-session source/retention/percentile information", and the Step 10 whole-data audit uses `retained_source_trial_indices_zero_based` to check that "retained source row IDs are strictly increasing and fall inside each ephys observation block". The plot payload is mostly array *views* rather than copies, so the wasted work is small; none of it was flagged as a bottleneck in the AI's timing analysis.
