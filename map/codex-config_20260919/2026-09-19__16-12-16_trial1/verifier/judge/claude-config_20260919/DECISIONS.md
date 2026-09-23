# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB/HDF5 file per session laid out as `/app/data/sub-<id>/<session>.nwb`. It discovers every session with a single sorted `glob` over that layout (174 files found) and opens each file **directly with `h5py`** rather than with `pynwb`, reading the raw HDF5 paths (`units/...`, `intervals/trials/...`, `acquisition/BehavioralEvents/...`, `acquisition/BehavioralTimeSeries/...`, `general/subject/subject_id`). Each file is opened once inside a `with` block and every quantity for that session is derived in that single pass; sessions are accumulated into per-session lists and assembled at the end. `--sample` stops after the first 2 usable sessions; `--full` (default) processes all of them.

ii.
```python
DATA_ROOT = Path("/app/data")
...
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
if not paths:
    raise FileNotFoundError(f"No NWB files under {DATA_ROOT}")
target_sessions = 2 if args.sample else None
...
for path in paths:
    make_plot = args.show_processing and len(converted_sessions) < 2
    result = convert_session(path, make_plot=make_plot)
    if result is None:
        continue
    converted_sessions.append(result)
    if target_sessions is not None and len(converted_sessions) >= target_sessions:
        break
```

```python
def convert_session(path: str, make_plot: bool) -> dict | None:
    ...
    with h5py.File(path, "r") as nwb:
        good_units, annotations = get_good_units(nwb)
        ...
        selection = map_observed_trials(nwb, good_units)
        trial_indices = selection.indices
        tone_onsets = final_tone_onsets(nwb, trial_indices)
        inputs, absolute_centers = construct_inputs(nwb, trial_indices, tone_onsets)
        rates = bin_spikes(nwb, good_units, trial_indices)
```

iii. From CONVERSION_NOTES Step 2: "`/app/data/dandiset.yaml` identifies DANDI:000363 version 0.230822.0128 … There are 28 `sub-*` directories containing 174 NWB/HDF5 session files (about 50 GiB total)." The directory listing is therefore the complete set of sessions and a sorted glob is both sufficient and deterministic ("Sort file paths, subjects, and region strings to make output reproducible", Step 5 Key Decision 10). Raw `h5py` access was chosen for speed and to allow the ragged NWB index datasets (`spike_times_index`, `obs_intervals_index`) to be sliced manually: "Read each session's concatenated spike vector once; use NumPy `searchsorted` against all trial edges for each unit" (Step 6). Step 10 Check 8 documents the deviation from the reference loader: "Converter reads native NWB HDF5 instead of reference MATLAB export but retrieves the corresponding units, spikes, trials/events, tracking, QC, and anatomy."

## 1-b. How are the data split into subjects?

i. The subject label for a session is taken from the **parent directory name** (`sub-440956`, including the `sub-` prefix). The AI cross-checks this against `general/subject/subject_id` inside the file and raises if they disagree (accepting either the bare numeric id or the `sub-`-prefixed form). At assembly, `subjects` is the sorted unique set of those labels and `subject_idx` is each session's index into that list. The result is 28 subjects with 3–10 sessions each.

ii.
```python
subject_id = Path(path).parent.name
source_subject = decode_scalar(nwb["general/subject/subject_id"][()])
if source_subject not in {subject_id, subject_id.removeprefix("sub-")}:
    raise ValueError(
        f"Path subject {subject_id} disagrees with NWB subject {source_subject}"
    )
```

```python
subjects = sorted({x["subject"] for x in converted_sessions})
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray(
    [subject_to_idx[x["subject"]] for x in converted_sessions],
    dtype=np.int64,
),
```

iii. Step 5's variable-mapping table lists "`general/subject/subject_id` / `sub-*` directory → `subjects`, `subject_idx`; Deterministic sorted unique subject IDs and per-session index" with the note "Cross-check file subject and NWB subject agree." The BIDS-style directory name is the canonical DANDI subject folder and is derived from the NWB subject id, so it is a faithful animal identifier; the explicit cross-check guards against a mislaid file. Step 2 records "28 `sub-*` directories" and "Subject session counts range from 3 to 10", which the conversion reproduces exactly.

## 1-c. How are the data split into sessions?

i. One NWB file is one session — no grouping or splitting is performed. The session id is the file stem with the modality suffix stripped (`sub-440956_ses-20190207T120657`). Session order in all output lists follows the sorted file paths, which because the filename embeds the acquisition timestamp is chronological within each subject. Per-session provenance (`session_id`, `source_file`, `subject`, unit/trial counts, recording span, tongue thresholds, label counts) is stored in `metadata['session_info']`. 173 of 174 files reach the output; one is skipped for having no classifier-good units.

ii.
```python
session_id = Path(path).stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

```python
info = {
    "session_id": session_id,
    "source_file": os.path.relpath(path, "/app"),
    "subject": subject_id,
    "n_units": int(len(good_units)),
    "n_trials_original": selection.n_original,
    ...
}
```

```python
"session_info": [x["info"] for x in converted_sessions],
```

iii. Step 2: "There are 28 `sub-*` directories containing 174 NWB/HDF5 session files"; each file carries its own trials table, units table and behavioral streams, so the file boundary *is* the session boundary. Step 4 resolves the one exception: the session whose `classification`/`anno_name` are unset "has 0 classifier-good labels and every annotation is blank … Exclude the zero-good session, yielding 173 usable sessions", which matches the papers' "173 analyzed behavioral sessions".

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table `intervals/trials` (`start_time`, `stop_time`, plus the categorical columns). The AI asserts that the number of `go_start_times` events equals the number of trial rows, so each trial has exactly one go cue and the trial ↔ go-cue mapping is unambiguous. Trials retained for output are referenced throughout by integer row indices (`trial_indices`) into that table, and the same index vector is used to slice the neural, input and output streams, which guarantees they stay in register.

ii.
```python
trial_group = nwb["intervals/trials"]
starts = trial_group["start_time"][:]
stops = trial_group["stop_time"][:]
n_original = len(starts)
...
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
if len(go_times) != n_original:
    raise ValueError("Go-cue count does not match trial-table length")
```

```python
if np.any(np.diff(observed_trial_indices) <= 0):
    raise ValueError("Observed trials are not strictly ordered")
```

iii. Step 2: "`intervals/trials` contains one row per trial … Every session has exactly one go-start timestamp per trial." Because the trials table is authoritative and one-to-one with the go cue, the AI uses it directly instead of re-deriving trial boundaries from event streams. Step 4 notes that the sample/delay events are *not* one-per-trial ("5,534 trials have repeats" of `sample_start_times`, because an early lick replays the epoch), which is why the go cue rather than the sample event is used as the trial key.

## 1-e. How are trials filtered based on quality controls?

i. **No behavioural quality filter is applied** — early-lick, `ignore` (no-response), photostimulated and auto/free-water trials are all deliberately retained because they are mandated decoder inputs/outputs. Trials are filtered only for absence or invalidity of *neural* data, in four successive steps:

1. **Observed trials**: `units/obs_intervals` for the good units is matched exactly (start *and* stop, `atol=1e-8`) to trial-table rows; unobserved behavioural trials are dropped. The AI verifies every good unit has the identical interval sequence (count check on all units, full sequence check on first/middle/last), so a single fixed neuron set per session is well defined. 94,370 → 93,310 trials.
2. **Per-unit validity**: `units/is_good_trials` (unit × observed-trial boolean). A trial is kept only if **every** retained good unit is flagged valid on it. 93,310 → 92,801 (−509).
3. **Window coverage**: the full requested [go−2.5 s, go+1.5 s] window must lie inside the overall observed recording span. 92,801 → 92,786 (−15).
4. **Population-all-zero**: after binning, any trial with zero spikes across *all* units in the whole 4 s window is dropped as missing rather than silent data. 92,786 → 90,363 (−2,423).

A session is dropped if fewer than 2 trials survive (raised as an error rather than a skip).

ii.
```python
observed_trial_indices = np.empty(len(common_obs), dtype=np.int64)
for j, (obs_start, obs_stop) in enumerate(common_obs):
    matches = np.flatnonzero(
        np.isclose(starts, obs_start, atol=1e-8, rtol=0)
        & np.isclose(stops, obs_stop, atol=1e-8, rtol=0)
    )
    if len(matches) != 1:
        raise ValueError(...)
    observed_trial_indices[j] = matches[0]

unit_trial_validity = nwb["units/is_good_trials"]
...
all_units_valid = np.all(unit_trial_validity[good_units, :], axis=0)
after_unit_qc = observed_trial_indices[all_units_valid]
...
full_window = (
    (go_times[after_unit_qc] + TIME_EDGES[0] >= recording_start - 1e-9)
    & (go_times[after_unit_qc] + TIME_EDGES[-1] <= recording_stop + 1e-9)
)
selected = after_unit_qc[full_window]
if len(selected) < 2:
    raise ValueError(f"Only {len(selected)} fully valid trials remain")
```

```python
# A few source observation rows extend beyond the actual end of every
# retained unit's spike stream despite being flagged good in NWB.  A
# four-second population-silent window across hundreds of units is a
# missing-recording signature, not physiology; exclude it explicitly.
neural_data_present = np.any(rates > 0, axis=(1, 2))
n_all_zero_neural_trials = int(np.sum(~neural_data_present))
if n_all_zero_neural_trials:
    trial_indices = trial_indices[neural_data_present]
    ...
if len(trial_indices) < 2:
    raise ValueError("Fewer than two trials remain after neural-data QC")
```

iii. Step 4: "Reference 'regular trial' filtering is analysis-specific and removes precisely several categories that are required decoder outputs/inputs here (early lick, no response, and stimulation). It therefore cannot be applied wholesale." Step 5 Key Decision 4 explains the `obs_intervals` mapping: "In nine NWBs it corresponds to only the trials listed in each unit's ragged `obs_intervals` (for example 160 observed of 480 behavioral trials). All good units within a session share exactly the same interval sequence." For `is_good_trials`, Step 4 cites the task instruction "Check for variables indicating valid data periods — exclude invalid data" and resolves: "Exclude any trial for which any retained good unit is marked invalid, preserving a fixed valid neuron set per session" (64,612 false of 37.68 M pairs, affecting 4 sessions). The all-zero rule was found empirically in Step 7: "Direct raw inspection showed every unit's spike stream ended at or before 1107.443 s, whereas the requested trial window was [1110.140, 1114.140) s despite its NWB observation flag being true", and generalised in Step 9 to avoid "treatment of unavailable ephys as zero firing".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a single concatenated ragged buffer of absolute session-clock spike times) together with `units/spike_times_index` (per-unit end offsets). Only units whose `units/classification` string is `'good'` contribute. The second ingredient is `acquisition/BehavioralEvents/go_start_times/timestamps`, which positions the bin edges. `units/anno_name` supplies each retained unit's brain-region label.

ii.
```python
def get_good_units(nwb: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    classifications = decode_array(nwb["units/classification"])
    good_indices = np.flatnonzero(classifications == "good")
    annotations = decode_array(nwb["units/anno_name"])[good_indices]
    if np.any(annotations == ""):
        raise ValueError("A classifier-good unit has an empty CCF annotation")
    return good_indices.astype(np.int64), annotations
```

```python
spike_index = nwb["units/spike_times_index"][:].astype(np.int64)
previous = np.r_[0, spike_index[:-1]]
# One sequential read is substantially faster than thousands of HDF5 reads.
all_spikes = nwb["units/spike_times"][:]
```

iii. Step 2: "`units` is a ragged NWB Units table. `spike_times` are absolute continuous-session seconds with `spike_times_index` row endpoints. It includes 272,227 sorted units and classifier labels." Spike times are the only neural representation in the file, so firing rates must be computed from them; this is electrophysiology, so "delta-F/F is not applicable" (Step 1).

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz with no smoothing, normalisation or baseline subtraction. For each retained unit the AI slices its spike segment out of the single concatenated buffer, runs one `np.searchsorted` against the (n_trials × 81) matrix of absolute bin edges, and differences adjacent positions to get counts per bin; counts are divided by the 0.05 s bin width. Output dtype is `float32`; the per-trial arrays handed to the format are `(n_neurons, 80)`. Two validity assertions follow: rates must be finite and non-negative, and `rate × 0.05` must be integral (i.e. really a spike count).

ii.
```python
def bin_spikes(nwb, good_units, trial_indices) -> np.ndarray:
    """Bin spikes to Hz; return trial x neuron x time float32."""
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
    if np.any(np.diff(absolute_edges.ravel()) < 0):
        raise ValueError("Trial windows overlap or are not time ordered")
    ...
    rates = np.empty(
        (len(trial_indices), len(good_units), len(TIME_CENTERS)), dtype=np.float32
    )
    for out_unit, source_unit in enumerate(good_units):
        spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
        edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
        rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Invalid firing rates")
    # Rates from integer counts must be integer after multiplying by bin width.
    if not np.allclose(rates * BIN_WIDTH_S, np.rint(rates * BIN_WIDTH_S)):
        raise ValueError("Firing rates are inconsistent with 50-ms spike counts")
    return rates
```

iii. Step 1 identifies the reference's `sliding_histogram`, which "Counts spikes in half-open windows and divides by bin width to yield Hz". Step 5's mapping row for `neural` says: "count absolute spike timestamps in 80 adjacent half-open bins `[go-2.5, go+1.5)` of width 0.05 s; divide counts by 0.05 to Hz". Step 10 Check 10: "Both align to go and use half-open spike intervals divided by width. Requested 50-ms adjacent bins override paper 40-ms/3.4-ms or 200-ms sliding variants."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit curation is exactly the authors' spike-sorting QC classifier verdict: keep units with `units/classification == 'good'`, with **no** thresholds applied to any of the 15 individual quality metrics, and no `unit_quality` ('good'/'multi') filtering. The method paper's additional >2 Hz firing-rate cut is deliberately *not* applied. A session with zero good units is skipped. This retains 69,453 of 272,227 units (25.5%), mean 401.5 per session (range 90–923), across 173 sessions. Per-trial neural validity (`is_good_trials`, all-zero windows) is handled as trial filtering (1-e) rather than unit filtering, which keeps a fixed neuron set per session as the target format requires.

ii.
```python
classifications = decode_array(nwb["units/classification"])
good_indices = np.flatnonzero(classifications == "good")
```

```python
good_units, annotations = get_good_units(nwb)
if len(good_units) == 0:
    print(f"SKIP {session_id}: no classifier-good units", flush=True)
    return None
```

```python
"unit_filter": "NWB units/classification == 'good'",
```

iii. Step 3: "Use only classifier-labeled `good` units. The QC classifiers are five region-specific logistic regressions trained from blinded manual labels using 15 jointly interpreted metrics. The white paper explicitly warns that simple per-metric thresholds cause unacceptable misses/false alarms." Step 4 resolution: "Use `classification == 'good'`; do not re-threshold the 15 metrics and do not apply the method-paper's later analysis-specific >2-Hz filter." Step 9 reconciles the count with the paper: "Converted/data good units are 69,453; the paper's 69,943 differs by exactly 490, attributable to the one NWB with no exported classifier/CCF labels."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset** (`go_start_times`). Because spike times and event timestamps share one absolute session clock, no resampling, interpolation or per-stream offset is required: the fixed relative edge vector is broadcast onto each trial's absolute go-cue time to give that trial's absolute bin edges, and spikes are binned against those edges directly. The identical `go_times[trial_indices, None] + …` construction is reused for the input bin centers and the tongue sampling, so all streams share one grid.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

```python
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
```

```python
"temporal_alignment_event": "go cue onset (NWB go_start_times)",
"off_start": -2.5,
"off_end": 1.5,
```

iii. Step 4: "NWB spikes and go events are absolute session timestamps … Subtract each trial's absolute go timestamp conceptually via absolute bin edges; no further synchronization or offset is required." Step 2 confirms "Native spike times and behavioral timestamps share the same absolute session clock; this differs from the paper's MATLAB export where trial spike arrays were already go-cue-relative." Step 10 Check 10: "Raw checks at first, middle, and last bins found no off-by-one issue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, fixed, for every stream. The grid is 81 edges from −2.5 s to +1.5 s relative to the go cue (`np.linspace(-2.5, 1.5, 81)`), giving exactly 80 non-overlapping half-open bins with centers at −2.475 … +1.475 s, identical for every trial and session. No rebinning or resampling of an already-binned product occurs: spikes are binned once at 50 ms straight from raw spike times, and the camera stream is sampled once onto the same 50 ms centers. `metadata['time_bin_size'] = 50.0` (ms), and the edges/centers are also stored in metadata.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
BIN_WIDTH_S = 0.05
```

```python
"time_bin_size": 50.0,
"time_bin_edges_seconds": TIME_EDGES.astype(np.float32),
"time_bin_centers_seconds": TIME_CENTERS.astype(np.float32),
"neural_bin_definition": (
    "Adjacent half-open 50-ms bins [edge_i, edge_i+1); no smoothing"
),
```

iii. Step 5 Key Decision 1: "80 bins with edges `np.linspace(-2.5, 1.5, 81)`; centers are edge midpoints. This prevents off-by-one ambiguity and uses half-open counting exactly like the reference histogram." Step 4 records the explicit override of the papers' binning: "Method code: 40-ms sliding window, 3.4-ms stride … Decoder specification overrides this: use 80 adjacent half-open 50-ms bins spanning exactly [-2.5, +1.5) s relative to go, report rates in Hz."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the instruction-tone / sample-epoch onsets), combined with the trial's go-cue time. Because an early lick replays the sample epoch, a trial can carry several sample onsets; the AI takes the **last sample onset strictly before the trial's go cue**. It additionally validates that the chosen tone falls at or after the trial's `start_time`, i.e. that it really belongs to that trial.

ii.
```python
def final_tone_onsets(nwb, trial_indices) -> np.ndarray:
    """Return the last sample/tone epoch onset before each selected go cue."""
    sample_starts = nwb[
        "acquisition/BehavioralEvents/sample_start_times/timestamps"
    ][:]
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    trial_starts = nwb["intervals/trials/start_time"][:]
    selected_go = go_times[trial_indices]
    positions = np.searchsorted(sample_starts, selected_go, side="left") - 1
    if np.any(positions < 0):
        raise ValueError("A selected trial has no sample/tone onset before go")
    tones = sample_starts[positions]
    if np.any(tones < trial_starts[trial_indices] - 1e-9):
        raise ValueError("A selected trial has no in-trial sample/tone onset")
    return tones
```

iii. Step 4: "Every trial has >=1 `sample_start_times`; 5,534 trials have repeats … Tone is the instruction sample; early lick triggers epoch replay → Select the last sample-start event before the final go cue in the trial, i.e. the tone epoch that led into the aligned delay/go. Most are go-1.85 s; protocol variants at -0.95/-2.45 s are preserved." Step 3 supplies the task structure that makes this the right event: "Sample epoch 0.65 s; 3 or 12 kHz tone … 1.2-s delay; 6-kHz go cue".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A signed continuous value per bin: the absolute time of each bin center minus that trial's tone-onset time, in seconds, stored as `float32` in input row 0. It is left continuous rather than binarised (the Decoder Task specifies "continuous, time-varying"), so it is a unit-slope ramp increasing by exactly 0.05 s per bin and crossing zero at the tone. Observed range across the dataset is [−1.525, 11.894] s; the lower bound is set by the shortest tone→go gap (0.95 s in some protocol variants) and the upper tail by trials whose sample epoch was replayed long before the go cue.

ii.
```python
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]

time_from_tone = absolute_centers - tone_onsets[:, None]
...
inputs = np.stack(
    [time_from_tone.astype(np.float32), photostim_on], axis=1
)
```

```python
"input_names": ["time from tone onset (s)", "photostimulation on"],
```

iii. Step 5 mapping: "For every bin center, compute absolute center timestamp minus that trial's final tone/sample onset; float32 seconds. Continuous signed time, as explicitly requested; preserves replayed sample epochs and protocol-dependent timing." Step 10 Check 6 verifies the construction globally: "every tone-time row advanced by 0.05 s/bin".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed **on the neural grid itself**: `absolute_centers` are the midpoints of exactly the same `TIME_EDGES` used to bin the spikes, offset by exactly the same `go_times[trial_indices]`. Bin *k* of the input therefore refers to the same physical interval as bin *k* of the firing rates by construction. No interpolation or offset correction is applied since all timestamps live on one absolute clock.

ii.
```python
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]   # inputs
absolute_edges   = go_times[trial_indices, None] + TIME_EDGES[None, :]     # neural
```

iii. Step 5 Key Decision 1 ("Fixed grid … centers are edge midpoints. This prevents off-by-one ambiguity"). The `--show-processing` plot panel 2 was reviewed for alignment in Step 7: "tone input is a unit-slope signed ramp crossing zero at the final tone … No temporal shift or discretization anomaly was seen."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The **paired photostimulation event timestamps** `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps` — not the trials-table `photostim_onset`/`photostim_duration` columns. The AI cross-validates the two sources: it asserts start and stop counts are equal, that every interval has positive duration, and (per session) that the number of trials-table rows with `photostim_onset != 'N/A'` equals the number of photostim events. Dataset-wide this is 18,588 intervals in 168 sessions.

ii.
```python
event_root = nwb["acquisition/BehavioralEvents"]
photo_starts = event_root["photostim_start_times/timestamps"][:]
photo_stops = event_root["photostim_stop_times/timestamps"][:]
if len(photo_starts) != len(photo_stops):
    raise ValueError("Photostimulation start/stop event counts differ")
if np.any(photo_stops <= photo_starts):
    raise ValueError("Non-positive photostimulation interval")
```

```python
photo_rows = decode_array(nwb["intervals/trials/photostim_onset"])
n_photo_rows = int(np.sum(photo_rows != "N/A"))
n_photo_events = len(
    nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"]
)
if n_photo_rows != n_photo_events:
    raise ValueError("Photostimulation trial rows and event counts differ")
```

iii. Step 4: "Absolute start/stop events; 18,588 paired intervals, exactly matching non-`N/A` trial rows … Evaluate the paired absolute event intervals on decoder time-bin centers; do not infer from power alone." The event stream was preferred because it is already on the same absolute clock as the bins (the trials-table onsets are strings relative to trial start and would need conversion), and because the count cross-check makes the two sources provably consistent.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series per trial, `float32`, in input row 1: a bin is 1 if its **center** lies inside any half-open stimulation interval `[start, stop)`. This is implemented without a Python loop by two `searchsorted` calls on the (sorted) start and stop event arrays — a center is inside an interval iff strictly more intervals have started than have stopped by that time. Trials with no stimulation come out all-zero automatically. Dataset range is [0, 1] with stimulation confined to pre-go bins.

ii.
```python
started = np.searchsorted(photo_starts, absolute_centers, side="right")
stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
photostim_on = (started > stopped).astype(np.float32)
```

```python
"trial_filter": (... "Behavioral categories otherwise retained."),
```

iii. Step 5 mapping: "Binary 1 where absolute bin center lies in any paired half-open laser interval `[start, stop)`, else 0; float32." The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", so a per-trial flag would not suffice. Step 3 records the expected structure to check against: "About 25% randomly interleaved trials in 17 mice; last 0.5 s of delay and ends before go", and Step 7's plot review confirms "laser state is confined to pre-go intervals".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation events are absolute session-clock times and are tested against `absolute_centers`, the same go-cue-anchored bin centers used for the neural binning and the tone input. Nothing is shifted or resampled; the comparison happens directly on the shared clock, so bin *k* of the photostim input covers the same interval as bin *k* of the firing rates.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
...
started = np.searchsorted(photo_starts, absolute_centers, side="right")
stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
```

iii. Step 4: "Native spike times and behavioral timestamps share the same absolute session clock", so evaluating the laser intervals at the absolute bin centers is exact. Step 7 processing-plot review: "laser state is confined to pre-go intervals", consistent with the paper's "ends before go".

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no stored choice column, so choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the side the tone instructed) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side; a miss means it licked the other side; `ignore` means it never licked. The AI validates that the two columns only take those values, and separately cross-checked the derivation against the raw lick-event timestamps.

ii.
```python
trials = nwb["intervals/trials"]
instruction = decode_array(trials["trial_instruction"])[trial_indices]
outcome_text = decode_array(trials["outcome"])[trial_indices]
...
if not set(np.unique(instruction)).issubset({"left", "right"}):
    raise ValueError("Unexpected trial instruction")
if not set(np.unique(outcome_text)).issubset(set(OUTCOME_VALUES)):
    raise ValueError("Unexpected outcome label")
```

iii. Step 4: "`trial_instruction` + `outcome`; event lick timestamps also available … Choice is actual lick direction, distinct from stimulus → Map hit to instructed side, miss to opposite side, ignore to no lick. This agrees with the first response-window lick in 94,669/94,990 native trials (99.66%); rare event conflicts/carryover licks make trial outcome the authoritative label." Step 5 Key Decision 8: "Use task semantics rather than raw first-lick timestamps; early/carryover sensor contacts explain the 0.34% disagreement and the paper defines choice jointly from instruction and correct/error outcome."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as `0 = left`, `1 = right`, `2 = no lick` (`output_values[0] = ['left','right','no lick']`), written to row 0 of the per-trial `(4, 80)` output array and broadcast across all 80 bins because it is a per-trial quantity. The array is initialised to 2 (no lick) so `ignore` trials need no explicit branch; hit trials take the instructed side and miss trials the opposite side. Resulting distribution: left 0.428 / right 0.422 / no lick 0.149.

ii.
```python
actual_choice = np.full(len(trial_indices), 2, dtype=np.int64)
hit = outcome_text == "hit"
miss = outcome_text == "miss"
actual_choice[hit & (instruction == "left")] = 0
actual_choice[hit & (instruction == "right")] = 1
actual_choice[miss & (instruction == "left")] = 1
actual_choice[miss & (instruction == "right")] = 0
...
outputs = np.empty((n_trials, 4, n_time), dtype=np.int64)
outputs[:, 0, :] = actual_choice[:, None]
```

```python
CHOICE_VALUES = ["left", "right", "no lick"]
```

iii. Step 5 Key Decision 2: "Every output trial is `(4, 80)`. Trial-level choice/outcome/early labels are broadcast through time so they coexist with time-varying tongue y in one dense categorical array." The left/right/no-lick coding follows the Decoder Task's listed order. Step 10 Check 4 verified the labels exhaustively: "A further full raw audit compared choice/outcome/early for every one of 90,363 retained trials; all passed."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of `intervals/trials`, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'` required by the Decoder Task. No derivation is needed; the AI only validates that no other value appears.

ii.
```python
outcome_text = decode_array(trials["outcome"])[trial_indices]
if not set(np.unique(outcome_text)).issubset(set(OUTCOME_VALUES)):
    raise ValueError("Unexpected outcome label")
```

iii. Step 2 enumerates the native totals directly from this column ("outcome hit/miss/ignore = 65,254/15,641/14,095"), and Step 5's mapping table lists `outcome → output[...,1,:]` with reference to the MATLAB pipeline's equivalent `correctness` field ("`1` correct, `0` error, `-1` no response", Step 1).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0 = ignore`, `1 = miss`, `2 = hit` via the index of `OUTCOME_VALUES`, written to row 1 of the output array and broadcast across all 80 bins. Resulting distribution: ignore 0.149 / miss 0.166 / hit 0.685.

ii.
```python
OUTCOME_VALUES = ["ignore", "miss", "hit"]
...
outcome_map = {name: idx for idx, name in enumerate(OUTCOME_VALUES)}
outcomes = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int64)
...
outputs[:, 1, :] = outcomes[:, None]
```

iii. Step 5 mapping: "Encode ignore=0, miss=1, hit=2 and broadcast. Retains all mandated classes." The ordering follows the Decoder Task's "(ignore, miss, hit)" listing. Step 9 checks the resulting distribution against the source: native [ignore .1484, miss .1647, hit .6870] vs converted [.1495, .1659, .6846], and explains why it differs from the paper's 84% figure ("computed on control trials excluding early licks", Step 3).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of `intervals/trials`, which holds `'no early'` / `'early'`. The AI validates that only those two values occur.

ii.
```python
early_text = decode_array(trials["early_lick"])[trial_indices]
if not set(np.unique(early_text)).issubset({"no early", "early"}):
    raise ValueError("Unexpected early-lick label")
```

iii. The trials table flags early licking explicitly (Step 2 lists `early_lick` among the per-trial columns, with native totals "early/no-early = 10,805/84,185"), so no derivation from lick events is needed. Step 3/4 note that the reference analyses *exclude* early-lick trials (`get_regular_trial_mask`), but that here "the required decoder outputs explicitly include early lick … so those categories must be retained."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0 = no`, `1 = yes` by a boolean comparison, written to row 2 of the output array and broadcast across all 80 bins. Resulting distribution: no 0.884 / yes 0.116.

ii.
```python
EARLY_VALUES = ["no", "yes"]
...
early = (early_text == "early").astype(np.int64)
...
outputs[:, 2, :] = early[:, None]
```

iii. Step 5 mapping: "Encode `no early`=0, `early`=1 and broadcast. Retained as required output." The coding order follows the Decoder Task's "(no, yes)". Step 12 re-verified these labels against the raw file for specific trials after the early-lick accuracy came in marginally below the 1.5× chance screen: "checked concrete converted/source trials `(session, converted trial, raw trial) = (0,1,1), (86,1,1), (172,1,1)`, obtaining `early`, `no early`, `no early` exactly."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is an `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` with matching absolute `timestamps` at ~294 Hz. Column 1 (y) is the measured value; column 2 (likelihood) determines visibility; column 0 (x) is used only to compute marker speed for outlier detection. The AI validates the array shape and that data and timestamps have equal length.

ii.
```python
tracking = nwb[
    "acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"
]
timestamps = tracking["timestamps"][:]
marker = tracking["data"][:]
if marker.ndim != 2 or marker.shape[1] < 3 or len(marker) != len(timestamps):
    raise ValueError("Unexpected tongue-tracking array")
x = marker[:, 0].astype(np.float64, copy=False)
y = marker[:, 1].astype(np.float64, copy=True)
likelihood = marker[:, 2].astype(np.float64, copy=False)
```

iii. Step 2: "`acquisition/BehavioralTimeSeries/Camera0_side_{Jaw,Nose,Tongue}Tracking` contains dense absolute timestamps and three float64 columns per marker (x, y, likelihood). All 174 sessions have tongue tracking. The median timestamp interval is 0.003400000544 s (~294.12 Hz)." Step 1 confirms the reference pipeline uses the same side camera markers ("side-camera `nose_x/y`, `tongue_x/y`, `jaw_x/y`, `whisker_x/y`").

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps, all at the session level before per-trial sampling:
1. **Visibility mask**: a frame counts as visible only if x, y and likelihood are finite and `likelihood >= 0.9`. (~10.5% of frames.) At least 10 visible frames are required.
2. **Velocity outlier repair**: marker speed `hypot(dx, dy)/dt` is computed over adjacent visible frame pairs; frames whose speed exceeds mean + 5 SD of that distribution have their y replaced by linear interpolation from the remaining visible, non-outlier frames. This is imported from the method paper's tracking preprocessing.
3. **Session percentiles**: the 40th and 60th percentiles of y over **all visible (corrected) frames in the session** become the two class edges (validated finite and ordered).
4. **Per-bin sampling and digitisation**: the frame nearest each bin center is taken and classified against those edges; bins with no visible/covered frame get class 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible = (
    np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
)
if np.sum(visible) < 10:
    raise ValueError("Too few visible tongue frames for session percentiles")

# Match the method paper's five-SD velocity outlier correction, restricting
# the reference distribution to adjacent high-confidence frames.
dt = np.diff(timestamps)
displacement = np.hypot(np.diff(x), np.diff(y))
valid_pairs = visible[:-1] & visible[1:] & np.isfinite(dt) & (dt > 0)
speeds = np.full(len(dt), np.nan, dtype=np.float64)
speeds[valid_pairs] = displacement[valid_pairs] / dt[valid_pairs]
reference_speeds = speeds[np.isfinite(speeds)]
outlier_frames = np.zeros(len(y), dtype=bool)
if len(reference_speeds) > 1:
    cutoff = np.mean(reference_speeds) + 5 * np.std(reference_speeds)
    outlier_frames[1:] = np.isfinite(speeds) & (speeds > cutoff)
interpolation_basis = visible & ~outlier_frames
if np.any(outlier_frames & visible) and np.sum(interpolation_basis) >= 2:
    target = np.flatnonzero(outlier_frames & visible)
    y[target] = np.interp(
        timestamps[target], timestamps[interpolation_basis], y[interpolation_basis]
    )

p40, p60 = np.percentile(y[visible], [40, 60])
if not np.isfinite(p40 + p60) or p40 > p60:
    raise ValueError("Invalid tongue percentile thresholds")
```

iii. Step 5 Key Decision 6: "A 0.9 likelihood cutoff is justified by the source's sharply bimodal likelihood distribution (only 11.87% of a 1/100-frame sample >=0.9; invisible mass is near 0). This also implements the requested class 3, unlike the paper's continuous-regression mean imputation." Key Decision 7: "Compute percentiles across all visible frames in the source session, before trial sampling, exactly matching 'over the session.'" Step 3 supplies the outlier rule: the method paper's pipeline "detected marker velocity outliers using a five-SD threshold, imputed outliers from nearby frames, and replaced occluded tongue positions with their mean for continuous-regression analyses"; the AI adopts the first two and rejects the third because "the present task … requires occluded tongue to be class 3, so mean-imputation is inappropriate for the output label."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four classes specified in the Decoder Task, with the per-session percentile edges from 8-b:
- `0` — y < p40
- `1` — p40 ≤ y ≤ p60 (equality at either edge falls in the middle class)
- `2` — y > p60
- `3` — not visible (low likelihood, non-finite, or no camera frame covering that bin center)

`output_values[3] = ['< 40th percentile', '40th to 60th percentile', '> 60th percentile', 'not visible']`. The array is initialised to 3 and only `sampled_visible` bins are overwritten. Dataset-wide distribution: [0.062, 0.032, 0.065, 0.841]; conditional on visible ≈ 39/20/41, i.e. approximately the intended 40/20/40.

ii.
```python
sampled_visible = covered & visible[nearest]
sampled_y = y[nearest]
classes = np.full(len(flat_centers), 3, dtype=np.int64)
classes[sampled_visible & (sampled_y < p40)] = 0
classes[
    sampled_visible & (sampled_y >= p40) & (sampled_y <= p60)
] = 1
classes[sampled_visible & (sampled_y > p60)] = 2
classes = classes.reshape(absolute_centers.shape)
```

```python
TONGUE_VALUES = [
    "< 40th percentile",
    "40th to 60th percentile",
    "> 60th percentile",
    "not visible",
]
"tongue_discretization": (
    "Per session visible-frame y percentiles: class 0 < p40; class 1 "
    "p40 through p60 inclusive; class 2 > p60; class 3 not visible"
),
```

iii. Step 5 Key Decision 7: "Boundaries follow the specification: equality at p40/p60 is class 1." The per-session scope and the 40/60 split are taken verbatim from the Decoder Task. Step 5's planned sanity check — "Confirm conditional tongue classes 0/1/2 approximate 40/20/40 over all visible source frames and class 3 occurs only for low-confidence/missing sampled frames" — is reported satisfied in Step 9 ("visible conditional split ~39/20/41").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. Camera timestamps are on the same absolute session clock as the spikes and go cues, so the AI evaluates the tongue at exactly the neural bin centers (`absolute_centers`): for each bin center it finds the **nearest camera frame** by `searchsorted` plus a left/right comparison, and uses that single frame's y and likelihood. A bin is only accepted if the center lies within the camera coverage *and* the nearest frame is within 1.5 median frame intervals (~5.1 ms) of the center; otherwise class 3. No interpolation or clock correction is applied. Because the video is trial-gated (off during the inter-trial interval), pre-trial bins on short trials legitimately fall outside coverage and become "not visible".

ii.
```python
flat_centers = absolute_centers.ravel()
right = np.searchsorted(timestamps, flat_centers, side="left")
right_clipped = np.clip(right, 0, len(timestamps) - 1)
left_clipped = np.clip(right - 1, 0, len(timestamps) - 1)
choose_left = np.abs(flat_centers - timestamps[left_clipped]) <= np.abs(
    timestamps[right_clipped] - flat_centers
)
nearest = np.where(choose_left, left_clipped, right_clipped)
median_dt = float(np.median(np.diff(timestamps)))
covered = (
    (flat_centers >= timestamps[0])
    & (flat_centers <= timestamps[-1])
    & (np.abs(timestamps[nearest] - flat_centers) <= 1.5 * median_dt)
)
sampled_visible = covered & visible[nearest]
```

```python
"tongue_visibility_rule": (
    "side-camera likelihood >= 0.9 and timestamp within 1.5 camera "
    "frame intervals; otherwise class 3"
),
```

iii. Step 4: the reference marker pipeline "uses side camera at 3.4-ms cadence … For requested categorical output, use nearest frame at each bin center, consider likelihood >=0.9 visible, compute session p40/p60 from visible y values, and label missing/low-confidence values as class 3." Step 1 records that the reference's own `align_markers_between_lims` resamples the marker stream by "retain[ing] the last frame in each time cell", i.e. one representative frame per bin rather than an average, which is the precedent for single-frame sampling. Step 2 quantifies the coverage gap: "Tongue timestamp coverage misses only 6 requested starts and 800 requested ends; uncovered output bins can be assigned the specified 'not visible' class." Step 7's plot review confirms no shift: "tongue bins use the plotted session percentile thresholds and low-confidence bins are class 3."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI distinguishes "nothing was recorded" (exclude) from "the measurement legitimately has no value" (explicit category), and otherwise fails loudly on anything unexpected:

- **Session never quality-controlled** (`classification`/`anno_name` unset for all 1,852 units in `sub-440958_ses-20190216T162508`): no unit matches `'good'`, the session is skipped with a printed message and returns `None`. 174 → 173 sessions.
- **Ragged/short `is_good_trials`** (9 sessions where the mask has fewer columns than the trial table): rather than padding or truncating, the columns are mapped onto trial rows through `obs_intervals`, with the column count asserted equal to the interval count.
- **Trials with no or invalid spike data**: excluded via the four-stage trial filter in 1-e (unobserved, any-unit-invalid, incomplete window, population-all-zero).
- **Untracked / occluded tongue and uncovered bins**: represented as the explicit class 3 rather than imputed (deliberately rejecting the method paper's mean-imputation, which is for continuous regression).
- **Anything else**: ~20 defensive `raise ValueError` checks (go-cue count vs trial count, duplicate/ambiguous interval→row matches, non-monotonic trial order, mismatched photostim start/stop counts, non-positive stim duration, unexpected categorical strings, non-finite or negative rates, non-integral spike counts, shape and class-range checks, subject-id disagreement).

ii.
```python
good_units, annotations = get_good_units(nwb)
if len(good_units) == 0:
    print(f"SKIP {session_id}: no classifier-good units", flush=True)
    return None
```

```python
if unit_trial_validity.shape[1] != len(common_obs):
    raise ValueError(
        "is_good_trials columns do not match the observed interval count: "
        f"{unit_trial_validity.shape[1]} vs {len(common_obs)}"
    )
```

```python
classes = np.full(len(flat_centers), 3, dtype=np.int64)   # default: not visible
```

```python
for row, n_values in enumerate([3, 3, 2, 4]):
    if outputs[:, row].min() < 0 or outputs[:, row].max() >= n_values:
        raise ValueError(f"Output row {row} is outside its class range")
```

iii. Step 4: "Exclude the zero-good session … Do not reconstruct labels from thresholds, because the white paper says thresholding is invalid and spatial labels are unavailable." Step 5 Key Decision 4 on the ragged masks: "applying it blindly would silently discard valid trials" (trajectory step 57). Step 10 "Issues Found and Resolved" documents both the ragged-mask investigation and the truncated-stream discovery: "Raw data proved all 375 unit streams ended before the flagged window. Added population-all-zero truncation filtering, regenerated sample/full outputs, and reran validation/checks successfully." Note that only the no-good-units case is a graceful skip; every other anomaly aborts the run by design, on the reasoning that unexplained data should be investigated rather than silently dropped.

## 10-a. What are the most time-consuming steps of the code?

i. Per the printed per-session timings in `conversion_full_out.txt`, the full conversion of 173 sessions took **194.5 s** (mean ~1.04 s/session, range 0.29 s to 12.80 s), plus **14.5 s** to pickle the 11.17 GiB result — 194.5 s total wall clock, well under the instructions' 15-minute budget. Within a session the cost is dominated by HDF5 I/O and by the per-unit binning loop:
1. `all_spikes = nwb["units/spike_times"][:]` — one sequential read of the entire concatenated spike buffer for **all** units in the session (up to ~3,191 sorted units, of which only ~400 are kept), which is why the slowest sessions are the ones with the largest spike buffers.
2. The per-unit `np.searchsorted` loop over the (n_trials × 81) edge matrix — O(n_good_units × n_trials × 81 × log n_spikes).
3. `tracking["data"][:]` — reading the full (n_frames, 3) tongue array (~680 k × 3).
4. Whole-array validation passes (`np.allclose(rates * BIN_WIDTH_S, np.rint(...))`, finiteness/negativity checks) which touch every element of the largest array in the session.
5. Pickling and the memory pressure of accumulating 11.17 GiB of `float32` neural data before the single write.

ii.
```python
# One sequential read is substantially faster than thousands of HDF5 reads.
all_spikes = nwb["units/spike_times"][:]
...
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
```

```python
elapsed = time.perf_counter() - started
print(
    f"DONE {session_id}: {len(good_units)} units, "
    f"{selection.n_original}->{len(trial_indices)} trials, {elapsed:.2f} s",
    flush=True,
)
```

iii. Step 6: "Per-unit HDF5 reads and Python loops over every trial/bin would cause excessive I/O and interpreter overhead. The target format itself requires ~11.2 GiB of float32 neural payload." Step 7's timing table projects "~6 minutes including heterogeneous unit/trial counts and I/O … Total full conversion estimate <10 minutes, below the 15-minute optimization threshold", and Step 9 reports the realised 194.50 s. Because the measured time was already far inside budget, no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain:
1. **Per-unit binning loop** in `bin_spikes`. The per-trial and per-bin dimensions are already vectorised (one `searchsorted` per unit against the full edge matrix). This loop is hard to remove because `spike_times` is ragged — each unit has a different number of spikes, so there is no single sorted array to search. It could in principle be collapsed with a global "unit-offset" trick (adding a large per-unit constant to both spikes and edges so one global `searchsorted` works), at a substantial cost in clarity.
2. **Observation-interval → trial-row matching loop** in `map_observed_trials`. This is the one clearly vectorizable loop the AI left in: it runs `np.flatnonzero` over the *whole* trial-start array once per observed interval, i.e. O(n_obs × n_trials) ≈ 640 × 640 comparisons per session. It could be a single `np.searchsorted` on the sorted `start_time` array (or `np.isin`) with an equality check, as the reference solution does with `np.isin(np.round(start_time, 4), np.round(obs[:, 0], 4))`. The AI did not flag this.
3. **Per-trial list comprehensions** at the end of `convert_session` (`[rates[i] for i in range(len(rates))]`, etc.), which materialise per-trial views; these are required by the target list-of-lists format rather than avoidable work.

Everything else — event sampling, tongue visibility/velocity/percentile/classification, categorical encoding, validity masks — is already fully vectorised.

ii.
```python
for j, (obs_start, obs_stop) in enumerate(common_obs):
    matches = np.flatnonzero(
        np.isclose(starts, obs_start, atol=1e-8, rtol=0)
        & np.isclose(stops, obs_stop, atol=1e-8, rtol=0)
    )
    if len(matches) != 1:
        raise ValueError(...)
    observed_trial_indices[j] = matches[0]
```

```python
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. Step 6: "Code speedups added: Read each session's concatenated spike vector once; use NumPy `searchsorted` against all trial edges for each unit; vectorize event sampling, tracking sampling, categorical construction, and validity masks; process one session at a time." The AI explicitly measured the outcome (Step 7 timing table) and stopped optimising once the projected runtime was under the 15-minute threshold, which is why the O(n_obs × n_trials) matching loop was never revisited.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and each session is processed in a single pass, so there is no second pass over the data and no recomputation of expensive quantities. What *is* repeated is a set of small redundant HDF5 reads and re-derivations:
- `acquisition/BehavioralEvents/go_start_times/timestamps` is read from disk **six** times per session — in `map_observed_trials`, `final_tone_onsets`, `construct_inputs`, `bin_spikes`, `plot_processing`, and twice more inline when building the `tone_minus_go_s_range` metadata entry — instead of being read once and passed down.
- `intervals/trials/start_time` is read in both `map_observed_trials` and `final_tone_onsets`.
- `units/spike_times_index` is read in both `bin_spikes` and `plot_processing`.
- `absolute_centers` is recomputed inside `construct_inputs` from the same expression that `bin_spikes` uses for the edges, and `np.diff(timestamps)` is computed twice in `process_tongue` (once for `dt`, once for `median_dt`).
- The `decode_array` pass over `intervals/trials/photostim_onset` is a pure cross-check duplicating information already read from the event streams.

These are all small relative to the spike buffer read, so they do not materially affect the 194 s runtime, but they are genuine repeated work that a single read-once-and-pass structure would remove.

ii.
```python
def map_observed_trials(nwb, good_units) -> TrialSelection:
    ...
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]

def final_tone_onsets(nwb, trial_indices) -> np.ndarray:
    ...
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]

def construct_inputs(nwb, trial_indices, tone_onsets):
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]

def bin_spikes(nwb, good_units, trial_indices) -> np.ndarray:
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

```python
"tone_minus_go_s_range": [
    float(np.min(tone_onsets - nwb[
        "acquisition/BehavioralEvents/go_start_times/timestamps"
    ][trial_indices])),
    float(np.max(tone_onsets - nwb[
        "acquisition/BehavioralEvents/go_start_times/timestamps"
    ][trial_indices])),
],
```

iii. The AI's stated design (Step 6) is "process one session at a time" with each session's heavy arrays read once — "Read each session's concatenated spike vector once" — which it does achieve for the expensive datasets. The repeated `go_start_times` reads are a by-product of passing the open `h5py.File` handle into each helper function rather than a pre-extracted per-session record; the AI never identifies them as repeated work in CONVERSION_NOTES. Because the tongue discretisation edges are per-session, they too are computed inside the single pass, so no second pass over the dataset is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Almost everything computed reaches the output, but several items are computed and then either discarded or never consumed by the decoder:
- **Validation-only work**: the full-array `np.allclose(rates * BIN_WIDTH_S, np.rint(rates * BIN_WIDTH_S))` integrality check and the finite/negative scan touch the entire 11 GiB of rates; the `decode_array` pass over `photostim_onset` and the `np.unique` membership checks on the three categorical columns; the redundant full-sequence `obs_intervals` comparison for three units. These were explicitly requested by the instructions ("Include sanity checks", "Validate data shapes and types at each step") but contribute nothing to the saved arrays.
- **Plot-support fields retained unconditionally**: `TongueProcessing` carries `timestamps`, `corrected_y` and `likelihood` (three full-session frame arrays) back to the caller on every session even when `--show-processing` is off, where they are dropped.
- **Unused derived quantities**: `selection.observed_indices`, `TrialSelection.recording_start/stop` beyond the window test, `tracking` column 0 (x) which exists only to compute the velocity-outlier mask, and the velocity-outlier interpolation itself, which touches a handful of frames per session and is highly unlikely to move a 40th/60th percentile.
- **Rich per-session provenance** (`metadata['session_info']`, 25 fields × 173 sessions including `source_trial_indices`) — valuable for auditing, unused by `train_decoder.py`.
- **Oversized dtypes**: outputs are built as `int64` when the four variables have at most 4 classes (`int8` would do), costing ~4× on the ~230 MB output payload; `subject_idx`/`brain_region_idx` are also `int64`.
- **Decoder-unused fields**: `brain_regions`/`brain_region_idx` (293 leaf CCF annotations) are required by the target format but not used by the decoder.

ii.
```python
# Rates from integer counts must be integer after multiplying by bin width.
if not np.allclose(rates * BIN_WIDTH_S, np.rint(rates * BIN_WIDTH_S)):
    raise ValueError("Firing rates are inconsistent with 50-ms spike counts")
```

```python
return TongueProcessing(
    classes=classes,
    ...
    timestamps=timestamps,
    corrected_y=y,
    likelihood=likelihood,
)
```

```python
outputs = np.empty((n_trials, 4, n_time), dtype=np.int64)
```

```python
"source_trial_indices": trial_indices.astype(np.int32),
```

iii. Step 5 Key Decision 9 covers the dtype choice: "float32 neural/input arrays and int64 categorical outputs. Dense neural payload is projected at about 11.2 GiB; available disk/RAM are sufficient" — i.e. the output dtype was judged immaterial next to the neural payload, which is true (the neural array is ~50× larger). The validation work is justified by the instructions' Step 6/Step 10 requirements and by the AI's stated policy of failing loudly ("Validate data shapes and types at each step"; Step 10 Check 5: "Every trial was finite, nonnegative, not population-all-zero, shaped neuron x 80, and satisfied `rate * 0.05 == integer spike count`"). The velocity-outlier correction is justified in Step 3 as matching "the method paper's five-SD velocity outlier correction", and the per-session provenance by Step 13's documentation goals.
