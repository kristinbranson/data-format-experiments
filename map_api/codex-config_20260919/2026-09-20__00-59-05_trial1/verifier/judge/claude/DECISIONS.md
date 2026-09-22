# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB file per session laid out as `/app/data/sub-<id>/<file>.nwb`. It discovers every file with a single sorted glob, and opens each one exactly once with `pynwb.NWBHDF5IO` inside a `with` block (no `h5py` anywhere). Within a file it pulls everything it needs from the standard NWB containers: `nwb.units` (spike times, `classification`, `anno_name`), `nwb.trials.to_dataframe()` (per-trial labels), `nwb.acquisition['BehavioralEvents']` (go / sample / photostim event timestamps), `nwb.acquisition['BehavioralTimeSeries']` (side-camera tongue tracking), and `nwb.subject.subject_id` / `nwb.identifier` for identity. Sessions are processed serially; the per-session dicts are then stitched together in `build_dataset`. `--sample` takes the first 2 files, `--full` (default) takes all 174.

ii.
```python
DATA_DIR = Path("/app/data")
...
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
if not files:
    raise FileNotFoundError(f"No NWB files under {DATA_DIR}")
if args.sample:
    files = files[:2]
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    units = nwb.units
    ...
    trials_all = nwb.trials.to_dataframe()
```
```python
def _events(nwb, name: str) -> np.ndarray:
    """Return an event timestamp vector without bypassing PyNWB."""
    series = nwb.acquisition["BehavioralEvents"].time_series[name]
    return np.asarray(series.timestamps[:], dtype=np.float64)
```
```python
def build_dataset(files: list[Path], show_processing: bool) -> dict:
    converted = []
    for path in files:
        session = convert_session(path, show_processing=show_processing and len(converted) < 2)
        if session is not None:
            converted.append(session)
```

iii. From CONVERSION_NOTES Step 2: `dandiset.yaml` declares DANDI 000363 v0.230822.0128 with **174 NWB assets and 28 subjects**, and the AI confirmed by scan that all 174 files open, that the `trials` table has the same 14 columns in every file, and that the required streams (`units`, `BehavioralEvents`, `Camera0_side_TongueTracking`) exist in every session. It therefore concluded the directory listing *is* the complete dataset and a sorted glob is sufficient and deterministic. It explicitly states all native-data inspection went through `pynwb.NWBHDF5IO` and "no HDF5 parser was used", satisfying the hard instruction constraint.

## 1-b. How are the data split into subjects?

i. Subject identity is read per session from `nwb.subject.subject_id` (a numeric string such as `'440956'`) and stored on the session record. At assembly the unique ids are sorted into `subjects`, and `subject_idx` holds each session's index into that list, as `int16`, in the same order as `neural`/`input`/`output`. No grouping by directory name is done; the folder name is treated as redundant. Result: 28 subjects with 3–10 sessions each.

ii.
```python
"subject": str(nwb.subject.subject_id),
```
```python
subjects = sorted({s["subject"] for s in converted})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_lookup[s["subject"]] for s in converted], dtype=np.int16),
```

iii. CONVERSION_NOTES Step 5 lists the mapping `subject.subject_id -> subjects, subject_idx` with the note "28 subjects expected", which is the count asserted in both `dandiset.yaml` and the papers' Methods. The AI used the canonical NWB subject field rather than parsing the path, and verified the resulting count (28) against that expectation in Step 9's consistency table.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is performed. Each session keeps `nwb.identifier` as `session_id` plus its source `path`, and both go into `metadata['session_info']` together with per-session trial/unit counts and derived statistics. Session order in all top-level lists follows the sorted file list (which, because the filename embeds the acquisition timestamp, is chronological within a subject). A session is dropped only if it yields no classifier-good units or fewer than 2 usable trials.

ii.
```python
"session_id": nwb.identifier,
...
"info": {
    "n_trials_nwb": len(trials_all),
    "n_trials": len(trials),
    "trials_excluded_no_population_spikes": n_excluded_no_neural,
    "n_neurons": len(good_indices),
    ...
}
```
```python
session_info = []
for s in converted:
    info = dict(s["info"])
    info.update(path=s["path"], session_id=s["session_id"], subject=s["subject"])
    session_info.append(info)
```

iii. Step 2 of CONVERSION_NOTES establishes the one-file-per-session layout, so the file boundary is the session boundary and nothing has to be inferred. Step 4 records the reconciliation that matters here: the archive has 174 assets but both papers report **173 behavioral sessions**, and the AI found exactly one file (`sub-440958_ses-20190216T162508`) whose `classification` column is `nan` for all 1,852 units. It decided this session "cannot yield a neural decoder session" and is dropped naturally by the unit-QC rule, landing on exactly the published 173.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `trials` table, one row per behavioural trial, with `start_time`/`stop_time` defining a half-open interval used to associate event streams to trials. Rather than assuming positional correspondence between event arrays and trial rows, the AI associates each event stream to trials by `searchsorted` on the trial interval, then *asserts* that every trial contains exactly one `go_start_times` event and at least one `sample_start_times` event, raising if not. The go time of a trial is that single go event.

ii.
```python
def _events_by_trial(timestamps, starts, stops) -> list[np.ndarray]:
    """Associate absolute event timestamps with half-open trial intervals."""
    left = np.searchsorted(timestamps, starts, side="left")
    right = np.searchsorted(timestamps, stops, side="right")
    return [timestamps[a:b] for a, b in zip(left, right)]
```
```python
go_by_trial = _events_by_trial(_events(nwb, "go_start_times"), starts, stops)
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
if not all(len(x) == 1 for x in go_by_trial):
    bad = [i for i, x in enumerate(go_by_trial) if len(x) != 1]
    raise ValueError(f"{path.name}: trials without exactly one go event: {bad[:10]}")
if not all(len(x) >= 1 for x in sample_by_trial):
    bad = [i for i, x in enumerate(sample_by_trial) if len(x) < 1]
    raise ValueError(f"{path.name}: trials without a sample/tone onset: {bad[:10]}")
go_times = np.asarray([x[0] for x in go_by_trial])
```

iii. CONVERSION_NOTES Step 2 warns that "Event-list counts can exceed trial counts for malformed/repeated state-machine events, so trial association must be based on trial intervals rather than positional indexing except for the one-go-per-trial stream after verification." This is the early-lick replay behaviour: a lick during sample/delay replays that epoch, so `sample_start_times` and `delay_start_times` can have several entries per trial while `go_start_times` has exactly one. Step 5's planned sanity check "Verify every trial has one go and at least one sample onset" was implemented as a hard assertion and passed on all 174 files (94,990 raw trials).

## 1-e. How are trials filtered based on quality controls?

i. Only one substantive trial filter is applied, and it is **data-derived rather than metadata-derived**: after binning spikes for every trial, the AI drops any trial whose entire classifier-good population has zero spikes across the whole 4 s window. A session is dropped if fewer than 2 trials survive. No behavioural quality filter is applied — early-lick, `ignore`/no-response, photostimulated, and auto/free-water trials are all deliberately retained. This removed 3,511 trials across 100 sessions (max 376 in one session), leaving 90,859 of 94,370 trials in the 173 retained sessions.

Notably, the AI *first implemented* the metadata route (intersecting `units/obs_intervals`), found it over-restrictive, and replaced it. The abandoned helper `_any_observation_mask` is still present in the file but is never called.

ii.
```python
# NWBs can include behavioral trials after every ephys probe stopped. The
# reference bins all classifier-good units without applying per-unit manual
# observation masks; preserve that logic, but discard windows with no spike
# from any retained unit because they contain no neural recording at all.
neural_present = np.any(rates != 0, axis=(1, 2))
n_excluded_no_neural = int((~neural_present).sum())
if n_excluded_no_neural:
    trials = trials.loc[neural_present].copy()
    go_times = go_times[neural_present]
    tone_times = tone_times[neural_present]
    inputs = inputs[neural_present]
    outputs = outputs[neural_present]
    rates = rates[neural_present]
if len(trials) < 2:
    print(f"SKIP {path.name}: fewer than two trials with neural data", flush=True)
    return None
```
Dead code left behind from the rejected approach:
```python
def _any_observation_mask(units, good_indices, starts, stops) -> np.ndarray:
    """Trials whose requested interval is observed for at least one kept unit. ..."""
```

iii. Step 7 documents the discovery and the reasoning: "its NWB contains 480 behavioral trials but population spike recording ends after trial 159. An attempted all-unit observation-interval intersection was too strict and removed valid trials/entire miss classes because per-unit accepted intervals differ. Final resolution matches reference binning: bin all classifier-good units, then exclude only trials with zero spikes across the complete retained population/window." Step 4/5 justify keeping the behaviourally "irregular" trials: the reference `get_regular_trial_mask` excludes early-lick, auto/free water, no-response and stimulation trials, but "Requested categories would be destroyed by the reference 'regular trial' mask" — early lick, ignore, and photostim are explicitly required decoder outputs/inputs here, so the analysis-specific mask is the one place a documented divergence from the reference is unavoidable. The ≥2-trial rule comes from the target-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units['spike_times']` (session-absolute seconds) for the subset of units with `units['classification'] == 'good'`, combined with the per-trial go-cue times from `BehavioralEvents/go_start_times`, which place the bin edges. Per-unit region labels come from `units['anno_name']` (kept at full Allen CCF resolution, 293 distinct labels). No other neural representation is read.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
...
annotations_all = np.asarray(units["anno_name"][:]).astype(str)
annotations = annotations_all[good_indices]
```
```python
def _unit_spike_times(units, unit_index: int) -> np.ndarray:
    """Read one ragged unit spike vector via the PyNWB DynamicTable API."""
    return np.asarray(units["spike_times"][unit_index], dtype=np.float64)
```

iii. Step 1 of CONVERSION_NOTES notes this is "extracellular electrophysiology, not imaging; delta-F/F is not applicable", and Step 5 maps `units.spike_times` for `classification == 'good'` directly onto `neural`. `spike_times` is the only neural stream in the file. The AI deliberately kept detailed `anno_name` rather than the probe target or a coarse area label: "Preserve detailed Allen annotation as neuron region, which is the anatomically measured location and supports later aggregation; do not substitute probe target for histology."

## 2-b. How is the `neural` data processed?

i. Spike counts in 80 non-overlapping half-open 50 ms bins, divided by the bin width to give firing rate in Hz, stored `float32`. For each good unit, all 81 bin edges for all trials are concatenated into one flat, monotonically increasing array and a *single* `np.searchsorted` gives the running spike total at every edge; differencing adjacent positions yields the per-bin counts for all trials at once. No smoothing, normalisation, baseline subtraction, or minimum-rate cutoff is applied. Two post-hoc assertions check the rates are finite, non-negative, and exact multiples of 20 Hz.

ii.
```python
def _bin_spikes(units, good_indices, go_times) -> np.ndarray:
    """Vectorized per-unit spike binning; result is trials x neurons x time."""
    absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
    # Flattened edges are sorted for this task (successive go cues are >4 s apart),
    # allowing one searchsorted call per neuron rather than one per trial.
    flattened = absolute_edges.ravel()
    monotonic = np.all(flattened[1:] >= flattened[:-1])
    rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
    for out_idx, unit_idx in enumerate(good_indices):
        spikes = _unit_spike_times(units, int(unit_idx))
        if monotonic:
            positions = np.searchsorted(spikes, flattened, side="left")
            counts = np.diff(positions.reshape(len(go_times), N_TIME + 1), axis=1)
        else:
            counts = np.stack([
                np.diff(np.searchsorted(spikes, row, side="left"))
                for row in absolute_edges
            ])
        rates[:, out_idx, :] = counts.astype(np.float32) / BIN_WIDTH
    return rates
```
```python
if not np.isfinite(rates).all() or np.any(rates < 0):
    raise ValueError(f"{path.name}: invalid firing rates")
if not np.allclose(rates / (1 / BIN_WIDTH), np.round(rates / (1 / BIN_WIDTH))):
    raise ValueError(f"{path.name}: rates are not integer-count multiples")
```

iii. Step 1 identified the reference's `sliding_histogram`, whose "reusable binning definition is `[center-width/2, center+width/2)` and firing rate is spike count / width". Step 4's discrepancy table records that the method paper used 40 ms width / 3.4 ms stride, and resolves it: "Keep reference half-open counting/rate logic but use 80 non-overlapping 50-ms bins as the explicit decoder override." Step 5 decision 1 explicitly declines to apply the method paper's analysis-specific <2 Hz neuron cutoff, on the grounds that it "is analysis-specific and not a global dataset curation rule".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered on exactly one criterion: `units['classification'] == 'good'`, the verdict of the region-specific spike-sorting QC classifier. No individual quality-metric thresholds are re-derived, no firing-rate floor, and no hemisphere or broad-region restriction. Sessions with zero good units are skipped. There is an additional (ineffective — see 9) guard intended to reject good units lacking a CCF annotation. Result: 69,453 of 272,227 units (25.5%), mean 401.46 per session, range 90–923.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    print(f"SKIP {path.name}: no classifier-good units", flush=True)
    return None
annotations_all = np.asarray(units["anno_name"][:]).astype(str)
annotations = annotations_all[good_indices]
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: classifier-good unit lacks CCF annotation")
```

iii. Step 3 summarises the QC white paper: "region-specific logistic-regression QC classifiers trained from manually curated Kilosort2 clusters using 15 quality metrics. 'Good' classifier units were used in paper analyses… QC ROC AUC averaged >0.9." Step 4 resolves the count discrepancy explicitly: the papers report 69,943 good units across 173 sessions, the archive labels 69,453, and the AI chose to "Use the release's explicit classifier labels, not reverse-engineered metric thresholds or the paper aggregate. The 490-unit (0.7%) difference is attributed to data-release/code-version differences and documented." It also rejected the alternative of using the 71,305 units with a non-empty annotation, because that would re-admit the 1,852 units of the never-QC'd session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams share one session-absolute clock, so alignment is simply adding the go-cue-relative bin grid to each trial's go-cue timestamp and binning spikes against the resulting absolute edges — no resampling, interpolation, or per-stream offset. The go time used is the unique `go_start_times` event falling inside that trial's `[start_time, stop_time]`.

ii.
```python
go_times = np.asarray([x[0] for x in go_by_trial])
```
```python
absolute_edges = go_times[:, None] + BIN_EDGES[None, :]
flattened = absolute_edges.ravel()
```
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
```

iii. Step 4's alignment row: "Raw reference export spikes already go-aligned; lick/laser shifted to go… NWB stores all streams in absolute session time… Use each NWB trial's absolute go event, then subtract it consistently from spike, event, and video timestamps. Associate events by trial intervals to avoid state-machine replay count mismatches." Step 10 check 12 confirms "81 edges/80 centers, final center +1.475 (not +1.5), half-open bins", and the `--show-processing` plots draw the go line at 0 and the tone line at `tone − go` over the population rate to demonstrate visually that nothing is shifted.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, fixed for every trial and every session: 81 edges from −2.5 s to +1.5 s relative to the go cue give 80 bins, with centers from −2.475 s to +1.475 s. The grid is defined once at module level and reused everywhere — for spikes, for the tone-time input, for the photostim input, and for the tongue output — so every stream shares an identical time base. There is no rebinning of an already-binned product: spikes are binned once from raw times, and the tongue is sampled once onto the same grid. `metadata['time_bin_size'] = 50.0` (ms), with `off_start`/`off_end` = −2.5/+1.5 and the edge and center vectors also stored in metadata.

ii.
```python
BIN_WIDTH = 0.050
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH / 2, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)
```
```python
"time_bin_size": 50.0,
"time_bin_units": "ms",
"temporal_alignment_event": "auditory Go cue onset",
"off_start": OFF_START,
"off_end": OFF_END,
"n_timepoints": N_TIME,
"bin_edges_relative_to_go_s": BIN_EDGES.tolist(),
"bin_centers_relative_to_go_s": BIN_CENTERS.tolist(),
"bin_interval_convention": "half-open [left, right)",
```

iii. The 50 ms width and the [−2.5, +1.5] s window are mandated by the Decoder Task section; Step 4 records this as an explicit override of the reference papers' 40 ms/3.4 ms and 200 ms/10 ms sliding windows, while retaining the reference's half-open counting convention. Step 5 decision 6: "Bin edges are exactly `np.linspace(-2.5,1.5,81)` and centers end at +1.475 s, avoiding an off-by-one endpoint."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the auditory sample/tone onsets) together with the trial's go-cue time. Because an early lick replays the sample epoch, a trial can contain several sample onsets; the AI takes the **last one at or before the go cue** within that trial's interval.

ii.
```python
sample_by_trial = _events_by_trial(_events(nwb, "sample_start_times"), starts, stops)
...
# Last onset is the successfully completed replay on early-lick trials.
tone_times = np.asarray([x[x <= g][-1] for x, g in zip(sample_by_trial, go_times)])
```

iii. Step 5's mapping table: "Last onset handles early-lick state-machine replays and yields the final completed sample epoch; normal onset is ~1.85 s before go." Step 3 describes the task structure that motivates this — "three 150-ms presentations of either 3-kHz or 12-kHz tone… then 1.2-s delay; 100-ms go cue… Early licking replays the sample/delay epoch." The last tone before the go cue is the one the animal actually used to make its decision.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of each of the 80 bin centers minus that trial's tone onset — i.e. a continuous, monotonically increasing ramp with a fixed 0.05 s step, offset per trial by the tone-to-go gap. Stored as `float32` in row 0 of the `(2, 80)` input array. No clipping, normalisation, or binarisation. Observed range across the full dataset: [−1.525, 11.894] s.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
inputs = np.empty((len(trials), 2, N_TIME), dtype=np.float32)
inputs[:, 0, :] = (absolute_centers - tone_times[:, None]).astype(np.float32)
```
Checked in `validate_converted`:
```python
assert np.all(np.diff(inp[0]) > 0)
```

iii. Step 5 decision 7 is explicit about the representation choice: "Follow the explicit Decoder Task ('time from tone onset in seconds, continuous, time-varying') rather than the generic key-format suggestion that event times be binary." Step 10 check 9 verifies the increments: "Tone increments are 0.05 s with maximum float32 deviation `7.64e-7`." The long upper tail (up to 11.9 s) is the expected consequence of early-lick replays lengthening the tone→go interval.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid: the same `BIN_CENTERS` array, added to the same `go_times`, so bin *k* of the input covers precisely the interval of bin *k* of the firing rates. No separate alignment step exists or is needed.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]   # used for inputs
absolute_edges   = go_times[:, None] + BIN_EDGES[None, :]     # used for spikes
```

iii. Single shared clock and single shared grid — Step 4: "Use each NWB trial's absolute go event, then subtract it consistently from spike, event, and video timestamps." The `--show-processing` plot panel overlays the tone-time ramp and the population firing rate on the same axis, with the tone marker drawn at `tone − go`, as a visual check that the ramp crosses zero at the tone.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The `BehavioralEvents/photostim_start_times` and `photostim_stop_times` timestamp series, associated to trials by the trial interval. The AI did **not** use the `trials` table's `photostim_onset` / `photostim_duration` columns. It asserts that within each trial the number of start and stop events is equal. In the 6 non-ogen sessions these series exist but are empty, so those sessions simply get an all-zero photostim input.

ii.
```python
stim_starts_by_trial = _events_by_trial(
    _events(nwb, "photostim_start_times"), starts, stops
)
stim_stops_by_trial = _events_by_trial(
    _events(nwb, "photostim_stop_times"), starts, stops
)
for i, (on, off) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    if len(on) != len(off):
        raise ValueError(f"{path.name}: unpaired laser events in trial {i}")
```

iii. Step 5's mapping table: "`photostim_start_times` / `photostim_stop_times` → `input[1]`… Pair events within each trial interval; expected during final 0.5 s before go and off by go." The choice of the event series over the trials table is consistent with the AI's general policy (Step 2) of associating absolute timestamps by trial interval rather than trusting positional/string-encoded fields; the event timestamps are already on the global clock and need no `'N/A'` string handling or trial-start arithmetic.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: bin *k* is 1 if its center falls in `[laser_on, laser_off)` for any stimulation epoch in that trial, else 0. Trials with no stimulation events stay 0 everywhere. Stored as `float32` in row 1 of the input array; `validate_converted` asserts the values are strictly 0 or 1.

ii.
```python
inputs[:, 1, :] = 0
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = ((absolute_centers[i] >= onset) &
                  (absolute_centers[i] < offset))
        inputs[i, 1, active] = 1.0
```
```python
assert np.all((inp[1] == 0) | (inp[1] == 1))
```

iii. The Decoder Task requires "Whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary state rather than a per-trial flag. The half-open `[on, off)` membership test matches the half-open convention used for the spike bins (`metadata['photostim_definition'] = "bin center in [photostim_start, photostim_stop)"`). Step 3 gives the timing sanity check the AI relied on: "Photoinhibition occurs in the last 0.5 s of delay and ends before go cue (including 100-ms ramp-down). This provides a strong timing sanity check for the binary photostimulation input" — and Step 10 check 3 independently re-derived the binary membership from the raw timestamps and compared with `np.allclose` (PASS).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The laser timestamps are already session-absolute, and they are compared directly against `absolute_centers` — the same go-cue-anchored bin centers used for the neural data — so no offset correction or interpolation is applied.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
...
active = ((absolute_centers[i] >= onset) & (absolute_centers[i] < offset))
```

iii. Same reasoning as 3-c: one global clock, one shared grid. The `--show-processing` plot steps the laser input against time-from-go, so its placement in the final 0.5 s of the delay (roughly −1.2 to −0.7 s relative to go) can be read off directly and compared against the paper's description.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the NWB file, so it is derived from two `trials` columns: `trial_instruction` (`'left'`/`'right'`) crossed with `outcome` (`'hit'`/`'miss'`/`'ignore'`). Hit ⇒ the animal licked the instructed side; miss ⇒ it licked the opposite side; ignore ⇒ no lick. An unexpected outcome string raises.

ii.
```python
def _classify_choice(instruction: str, outcome: str) -> int:
    """Map task-defined response to left=0, right=1, no lick=2."""
    if outcome == "ignore":
        return 2
    if outcome == "hit":
        return 0 if instruction == "left" else 1
    if outcome == "miss":
        return 1 if instruction == "left" else 0
    raise ValueError(f"Unexpected outcome {outcome!r}")
```
```python
instructions = trials["trial_instruction"].astype(str).to_numpy()
outcomes_text = trials["outcome"].astype(str).to_numpy()
choice = np.asarray([
    _classify_choice(inst, outcome)
    for inst, outcome in zip(instructions, outcomes_text)
], dtype=np.int8)
```

iii. Step 5 decision 3 explains the choice of derivation *and* the alternative it rejected: "Use outcome crossed with instruction. Across all raw trials this agrees with the first response-period lick for >99.6%; discrepancies are predominantly auto/free-water cases or sparse event-log glitches. It also guarantees the semantically required no-lick label for ignore trials." In other words, the AI audited the task-derived label against the raw `left_lick_times`/`right_lick_times` event streams and preferred the task-defined label as the more reliable of the two.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded left = 0, right = 1, no lick = 2 (`int8`), written into row 0 of the per-trial `(4, 80)` output array and repeated identically across all 80 bins. `output_values[0] = ['left', 'right', 'no lick']`. Range is asserted in `validate_converted`.

ii.
```python
OUTPUT_NAMES = ["lick direction choice", "outcome", "early lick", "tongue y-position"]
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ...
]
```
```python
outputs = np.empty((len(trials), 4, N_TIME), dtype=np.int8)
outputs[:, 0, :] = choice[:, None]
```
```python
assert np.all((out[0] >= 0) & (out[0] <= 2))
```

iii. Step 5 decision 4 covers the tiling: "Store all four outputs as `(4,80)` int arrays. Repeat per-trial choice/outcome/early labels across time and vary tongue class by bin. A single 2-D representation is required to combine per-trial and time-varying outputs in the validator/trainer." Step 10 check 12 verified the invariant "choice no-lick iff outcome ignore" holds across the full dataset, and the resulting distribution [0.429, 0.422, 0.149] is close to the raw left/right instruction balance as expected.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the `trials` table, which already holds exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes_text = trials["outcome"].astype(str).to_numpy()
```

iii. Step 5's mapping table: "`trials.outcome` → `output[1]` outcome | Direct categorical map ignore=0, miss=1, hit=2 | `correctness` mapping in reference loader | Retain ignore trials because explicitly requested." Step 1 had identified the reference's equivalent field as `correctness` (1 hit, 0 error, −1 no response), so the NWB column is the direct analogue and no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps ignore→0, miss→1, hit→2 (`int8`); the value is written to row 1 of the output array and repeated across all 80 bins. `output_values[1] = ['ignore', 'miss', 'hit']`. Full-dataset distribution: [0.149, 0.166, 0.684].

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcomes_text], dtype=np.int8)
outputs[:, 1, :] = outcome[:, None]
```

iii. The 0/1/2 ordering follows the Decoder Outputs spec literally ("Outcome (ignore, miss, hit, per-trial)"). Tiling across bins for the same single-array reason as 5-b. Step 9 notes that the 68.4% hit rate is "Not directly comparable" to the paper's 84% correct figure because the paper's number is computed on selected non-early control trials whereas this dataset deliberately retains early, ignore, and stimulated trials — a difference the AI documented rather than tried to eliminate.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `early_lick` column of the `trials` table, holding `'no early'` / `'early'`.

ii.
```python
early_text = trials["early_lick"].astype(str).to_numpy()
```

iii. Step 5's mapping table: "`trials.early_lick` → `output[2]` early lick | `no early`=0, `early`=1 | `early_lick_trials` in reference loader | Retain early trials because explicitly requested." The flag is stored explicitly by the acquisition software, so no derivation from lick times is needed; and because the lick that sets the flag occurs during sample or delay, the underlying event falls inside the −2.5 s pre-go window, making the per-trial flag decodable from the extracted neural window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Fixed dictionary `'no early'`→0, `'early'`→1 (`int8`), written to row 2 and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Full-dataset distribution [0.885, 0.115].

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_text], dtype=np.int8)
outputs[:, 2, :] = early[:, None]
```
```python
assert np.all((out[2] >= 0) & (out[2] <= 1))
```

iii. Coding follows the Decoder Outputs spec ("Early lick (no, yes, per-trial)"). Same tiling rationale as 5-b/6-b. Step 12 specifically re-scrutinised this output because its decoder accuracy (0.756 vs 0.5 chance = 1.51×) was closest to the protocol's 1.5× investigation trigger, and confirmed "its raw labels, variation (11.5% yes), alignment, and small generalization gap all pass".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = (tongue_x, tongue_y, tongue_likelihood) with matching `timestamps` at ~294 Hz. Column 1 is the y value; column 2 is the DeepLabCut likelihood that decides visibility.

ii.
```python
behavior = nwb.acquisition["BehavioralTimeSeries"]
tongue = behavior.time_series["Camera0_side_TongueTracking"]
tongue_times = np.asarray(tongue.timestamps[:], dtype=np.float64)
tongue_data = np.asarray(tongue.data[:], dtype=np.float64)
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Step 2: "`acquisition/BehavioralTimeSeries` has side-camera Jaw, Nose, and Tongue tracking in every session; each is `(samples,3)` = x, y, likelihood with explicit timestamps. Some sessions add whisker/lick-port or Camera3 duplicates; Camera0 side tongue is consistent across all sessions." The AI verified presence in all 174 files and chose Camera0 side view for consistency and because Step 3 records that the papers' "Video/marker analysis uses side view, 300-Hz tracking". `metadata['tongue_tracking_series']` records the choice.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame is "visible" iff its y and likelihood are finite **and** likelihood ≥ 0.9. (2) The 40th and 60th percentiles of y are computed once per session over **all visible raw frames** in the session (not over binned values). (3) Each bin center is matched to the temporally **nearest single camera frame** (`_nearest_indices`), and that frame's y and likelihood are used for that bin. There is no averaging over the ~15 frames that fall inside each 50 ms bin, no velocity-outlier cleaning, and no mean-imputation of occluded frames. Thresholds are recorded per session in `session_info`.

ii.
```python
DLC_VISIBLE_THRESHOLD = 0.9
```
```python
session_visible = np.isfinite(tongue_y) & np.isfinite(tongue_likelihood) & (
    tongue_likelihood >= DLC_VISIBLE_THRESHOLD
)
if session_visible.sum() < 2:
    raise ValueError(f"{path.name}: insufficient visible tongue samples")
q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    len(trials), N_TIME
)
matched_y = tongue_y[nearest]
matched_likelihood = tongue_likelihood[nearest]
```
```python
def _nearest_indices(source_times, query_times) -> np.ndarray:
    """Indices of temporally nearest source samples for sorted query times."""
    idx = np.searchsorted(source_times, query_times, side="left")
    idx = np.clip(idx, 1, len(source_times) - 1)
    prev = idx - 1
    use_prev = query_times - source_times[prev] <= source_times[idx] - query_times
    return np.where(use_prev, prev, idx)
```

iii. Step 5 decision 5: "Use DLC likelihood >=0.9, finite y, and nearest frame. The paper's invisible-tongue mean imputation is deliberately not used because class 3 must preserve missing visibility. Percentiles are computed once per session over visible y samples. Five-SD velocity outliers will be inspected in sample plots; because the requested target is discretized and percentile-robust, any cleaning will be added only if it changes sampled-bin classes materially." The threshold value is justified by the likelihood distribution being sharply bimodal ("Confidence is sharply bimodal near 0/1, making 0.9 a stable visibility threshold" — verified: ~89% of frames below 0.01, ~10.5% at or above 0.99, <0.1% in between). Step 4 records the deliberate divergence from the method paper, which "mean-imputes occluded tongue after five-SD velocity outlier cleanup", on the grounds that the Decoder Task requires a distinct `not visible` class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as specified: 0 = y below the session's 40th percentile, 1 = between the 40th and 60th (inclusive on both edges), 2 = above the 60th, 3 = not visible. The array is initialised to 3 and only bins whose matched frame is visible are overwritten, so invisibility is the default. `output_values[3]` names the four codes; `validate_converted` asserts the 0–3 range. Full-dataset distribution over timepoints: [0.062, 0.032, 0.065, 0.841].

ii.
```python
visible = np.isfinite(matched_y) & np.isfinite(matched_likelihood) & (
    matched_likelihood >= DLC_VISIBLE_THRESHOLD
)
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
tongue_class[visible & (matched_y < q40)] = 0
tongue_class[visible & (matched_y >= q40) & (matched_y <= q60)] = 1
tongue_class[visible & (matched_y > q60)] = 2
outputs[:, 3, :] = tongue_class
```
```python
OUTPUT_VALUES[3] = [
    "below 40th percentile", "40th to 60th percentile",
    "above 60th percentile", "not visible",
]
```

iii. The cut points and the per-session scope are dictated verbatim by the Decoder Task ("0: < 40th percentile of y-position over the session; 1: 40th to 60th percentile; 2: > 60th percentile; 3: not visible"). Step 5's planned check — "visible tongue classes approach 40/20/40 session fractions before time-window sampling" — is borne out: restricted to visible bins the realised split is 0.39 / 0.20 / 0.41. Step 10 check 12 verified `q40 <= q60` in every session, and check 4 independently recomputed the session percentiles, nearest frame, visibility, and class from the raw NWB and compared with `np.allclose` (PASS).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events, so each of the 80 go-cue-anchored bin centers is mapped to the nearest camera frame by a single vectorised nearest-neighbour search over the whole session's timestamp vector. Because the camera runs at ~294 Hz and the bin is 50 ms, the matched frame is normally within ~1.7 ms of the bin center. There is no interpolation and no offset correction. Note that `_nearest_indices` clips to the valid index range and imposes no maximum distance, so a bin center falling inside a camera gap still receives the nearest frame outside the bin (empirically ~0.2–0.3% of bins, and in the sessions checked almost none of those are labelled visible, so the practical impact is negligible).

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
...
nearest = _nearest_indices(tongue_times, absolute_centers.ravel()).reshape(
    len(trials), N_TIME
)
```
```python
"tongue_frame_matching": "nearest timestamp to each neural bin center",
```

iii. Step 4's alignment resolution again: all streams are in absolute session time and are anchored to the trial's go event, so the tongue class for bin *k* covers the same interval as the firing rates for bin *k*. Step 5's planned sanity check "all extraction centers within available tongue timestamp range" was intended to cover the coverage question, and Step 10 check 4 independently re-derived the nearest-frame class from the raw file and matched with `np.allclose`. The `--show-processing` plots overlay the raw tongue y trace, the visible frames, the likelihood trace with the 0.9 line, and the resulting class steps all on a common time-from-go axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four distinct cases, handled by three different strategies:

- **Session never quality-controlled** (`classification` is NaN for all 1,852 units in `sub-440958_ses-20190216T162508`): `.astype(str)` turns NaN into `'nan'`, so no unit matches `'good'`, and the session is skipped with a printed message. This is how the AI lands on 173 sessions.
- **Trials with no ephys recording** (behaviour continues after the probes stop; up to 376 leading/trailing trials in some sessions, plus free-water trials): detected as all-zero population activity and dropped (see 1-e).
- **Occluded / retracted tongue**: non-finite y or likelihood < 0.9 ⇒ the bin is assigned the explicit `not visible` class rather than imputed.
- **Sessions with no photostimulation**: the `photostim_*` event series exist but are empty, so the binary input is simply all zeros with no special-casing.

Elsewhere the AI deliberately chose to **fail fast rather than paper over** anomalies: unexpected `outcome` strings, trials without exactly one go event, trials with no sample onset, unpaired laser start/stop events, non-finite or negative firing rates, rates that are not integer multiples of 20 Hz, and sessions with <2 visible tongue frames all raise. None of these fired on the full dataset.

ii.
```python
classifications = np.asarray(units["classification"][:]).astype(str)
good_indices = np.flatnonzero(classifications == "good")
if len(good_indices) == 0:
    print(f"SKIP {path.name}: no classifier-good units", flush=True)
    return None
```
```python
neural_present = np.any(rates != 0, axis=(1, 2))
...
if len(trials) < 2:
    print(f"SKIP {path.name}: fewer than two trials with neural data", flush=True)
    return None
```
```python
tongue_class = np.full((len(trials), N_TIME), 3, dtype=np.int8)
tongue_class[visible & (matched_y < q40)] = 0
```
```python
if not all(len(x) == 1 for x in go_by_trial):
    raise ValueError(f"{path.name}: trials without exactly one go event: {bad[:10]}")
...
if not np.isfinite(rates).all() or np.any(rates < 0):
    raise ValueError(f"{path.name}: invalid firing rates")
```

iii. Step 4 resolves the unlabelled session as "Exclude the zero-good-unit session naturally because it cannot yield a neural decoder session; remaining session count is exactly 173", which is exactly the published number. Step 7 documents the discovery and fix of the no-ephys trials. Step 5 decision 5 and Step 4's "Tongue missingness" row justify representing occlusion as a category instead of imputing: "Preserve invisibility as class 3; do not mean-impute invisible values." The underlying principle is consistent — where *nothing was recorded*, the session or trial is dropped (emitting it would fabricate 4 s of 0 Hz across every unit); where the measurement *legitimately has no value*, it becomes an explicit category.

One caveat: the guard intended to catch good units lacking a CCF annotation (`if np.any(annotations == "")`) cannot actually fire, because `.astype(str)` converts a missing annotation to the string `'nan'`, not `''`. Such a unit would silently receive the brain-region label `"nan"`. It has no effect on this dataset, since the only session with NaN annotations is already dropped by the unit-QC rule and all 69,453 retained units have real annotations, but the check does not do what its error message claims.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion of 174 files took **223.4 s** (~1.3 s/session, range 0.47–6.16 s), including pickling the 11.042 GiB result — comfortably inside the instructions' 15-minute budget, so no further optimisation was pursued. The dominant costs are NWB I/O and the per-unit binning: (a) reading each unit's ragged `spike_times` vector one unit at a time through the PyNWB `DynamicTable` API (hundreds of reads per session, and the single largest cost, scaling with unit count — the per-session times track unit count closely); (b) reading the full `(n_frames, 3)` tongue array (~680 k × 3 per session); (c) one `np.searchsorted` over 81 × n_trials edges per unit. Secondary costs the AI added itself: the `np.allclose(rates / 20, round(rates / 20))` validation, which allocates two temporaries the size of the whole rates array per session, and binning trials that are subsequently discarded.

ii.
```python
started = time.perf_counter()
...
elapsed = time.perf_counter() - started
print(
    f"DONE {path.name}: {result['info']['n_trials']} trials, "
    f"{result['info']['n_neurons']} neurons, {elapsed:.2f} s", flush=True
)
```
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = _unit_spike_times(units, int(unit_idx))   # one HDF5 read per unit
```

iii. Step 6: "Naive nested unit x trial x bin spike counting would invoke millions of Python operations. Retaining full raw unit/video tables beyond a session would also inflate memory." Step 7's timing table extrapolated 1.11 s/session to "~3.2 min for 174 files by linear session count" and concluded "full run safely below 15 min"; the realised 223 s confirmed that estimate, so the AI (like the reference) judged no bottleneck worth optimising further. The AI printed per-session timings as instructed, but did not profile sub-steps within a session, so its bottleneck attribution is inferred from the unit-count scaling rather than measured directly.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The dominant loop — over trials × bins — was already vectorised away: all trial edges are flattened into one array so each unit needs a single `searchsorted`. What remains:

- **The per-unit loop** in `_bin_spikes`. This is irreducible: spike times are stored ragged, so there is no single sorted array to search across units. (It could still be made cheaper by reading the whole `spike_times` buffer once and slicing it with the `VectorIndex` offsets, instead of one PyNWB read per unit, as the reference does.)
- **The nested photostim loop** over trials × stimulation epochs. Since there is at most one epoch per trial, this could have been fully vectorised into a single `(n_trials, 80)` comparison against per-trial on/off vectors.
- **Three Python list comprehensions over trials**: `tone_times` (`[x[x <= g][-1] for ...]`), `choice` (per-trial `_classify_choice` calls), and `outcome`/`early` dictionary lookups. All are O(n_trials) Python-level and could be done with `np.where` on vectorised string comparisons.
- **`_events_by_trial`**, which materialises one Python list slice per trial for four different event streams.

ii. Vectorised (the part that mattered):
```python
flattened = absolute_edges.ravel()
positions = np.searchsorted(spikes, flattened, side="left")
counts = np.diff(positions.reshape(len(go_times), N_TIME + 1), axis=1)
```
Not vectorised:
```python
for i, (onsets, offsets) in enumerate(zip(stim_starts_by_trial, stim_stops_by_trial)):
    for onset, offset in zip(onsets, offsets):
        active = ((absolute_centers[i] >= onset) & (absolute_centers[i] < offset))
        inputs[i, 1, active] = 1.0
```
```python
tone_times = np.asarray([x[x <= g][-1] for x, g in zip(sample_by_trial, go_times)])
choice = np.asarray([_classify_choice(inst, outcome)
                     for inst, outcome in zip(instructions, outcomes_text)], dtype=np.int8)
```

iii. Step 6: "For each unit, all absolute bin edges across trials are flattened and passed to one `np.searchsorted`, then differenced into all trial/bin counts. Video frames are matched in one vectorized nearest-neighbor search." The AI vectorised precisely the two loops that dominate (unit × trial × bin spike counting, and frame matching) and left the O(n_trials) Python loops, which run at most ~800 iterations per session and are a negligible share of the 1.3 s/session. It did not document these remaining loops as a known residual cost.

## 10-c. What processing does the code repeat multiple times?

i. Nothing substantive is recomputed: each NWB file is opened exactly once inside a single `with` block, the bin grid is built once at module level and reused for every trial, session, and data stream, and the per-session tongue percentiles are computed in the same pass (no second pass over the data is needed). The redundancies that do exist are trivial:

- `trials_all` is copied to `trials` and `start_time`/`stop_time` are extracted twice (`starts_all`/`stops_all`, then `starts`/`stops`), with the first pair never used.
- `rates / (1 / BIN_WIDTH)` is evaluated twice in the same assertion, materialising the full array twice.
- `np.isfinite`/`np.any` scan the whole rates array in addition to `np.any(rates != 0, ...)` used for filtering.
- The `--show-processing` path re-derives `rel_video` and visibility masks that were already computed during conversion.

ii.
```python
trials_all = nwb.trials.to_dataframe()
starts_all = trials_all["start_time"].to_numpy(np.float64)   # never used
stops_all = trials_all["stop_time"].to_numpy(np.float64)     # never used
trials = trials_all.copy()
starts = trials["start_time"].to_numpy(np.float64)
stops = trials["stop_time"].to_numpy(np.float64)
```
```python
if not np.allclose(rates / (1 / BIN_WIDTH), np.round(rates / (1 / BIN_WIDTH))):
```

iii. The AI did not flag repeated processing as an issue; Step 6 only claims "Only one NWB/session is open at a time; compact float32/int8 arrays are retained". Structurally the conversion is a genuine single pass — the design choice that makes this possible is computing the per-session tongue percentiles inside the same session pass, since the discretisation is per-session rather than global. The listed redundancies are leftovers rather than design decisions and cost a small constant factor.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none affecting correctness:

- **Dead code**: `_any_observation_mask` (20 lines, containing a nested per-unit × per-interval loop) is defined but never called — a leftover from the rejected `obs_intervals` filtering approach. `starts_all`/`stops_all` are computed and never used. The `monotonic` fallback branch in `_bin_spikes` is unreachable in practice (go cues are always >4 s apart) but is still evaluated once per session via `np.all(flattened[1:] >= flattened[:-1])`.
- **Work on trials that are then discarded**: because the trial filter is derived *from* the binned rates, the AI bins spikes and builds inputs/outputs for all 94,370 trials and then throws away 3,511 of them. In the worst session this is ~70% of the binning work (376 of 535 trials).
- **Per-session `info` statistics** (`choice_counts`, `outcome_counts`, `early_lick_counts`, `tongue_class_counts`, `photostim_trial_count`, `auto_water_trial_count`, `free_water_trial_count`, `tongue_visible_fraction_at_bins`) are computed for every session and stored in `metadata['session_info']`; they are documentation only and unused by the decoder. Some are also computed on the pre-filter arrays while `n_trials` is post-filter, so they are mildly inconsistent with each other.
- **The 20 Hz-multiple assertion**, which costs two full-array temporaries per session.
- `tongue_x` (column 0) is read as part of the `(n, 3)` array but never used.

ii.
```python
def _any_observation_mask(units, good_indices, starts, stops) -> np.ndarray:
    """..."""
    valid = np.zeros(len(starts), dtype=bool)
    for unit_idx in good_indices:
        intervals = np.asarray(units["obs_intervals"][int(unit_idx)], dtype=np.float64)
        ...
    return valid
# never called anywhere in the file
```
```python
monotonic = np.all(flattened[1:] >= flattened[:-1])
...
else:
    counts = np.stack([...])   # unreachable in practice
```
```python
"choice_counts": np.bincount(choice, minlength=3).tolist(),
"outcome_counts": np.bincount(outcome, minlength=3).tolist(),
"early_lick_counts": np.bincount(early, minlength=2).tolist(),
"tongue_class_counts": np.bincount(tongue_class.ravel(), minlength=4).tolist(),
```

iii. The AI does not acknowledge any of this in CONVERSION_NOTES; Step 13 claims the directory was cleaned and "Investigation scripts/output and bytecode moved to cache", but the dead helper inside `convert_data.py` was not removed. The per-session statistics and the extra assertions are defensible as deliberate self-validation and provenance (the instructions ask for sanity checks at each step and for informative metadata), but the stale `_any_observation_mask` and the unused `starts_all`/`stops_all` are simply residue from the abandoned first attempt at trial filtering, and their presence is misleading to a reader trying to work out how trials are actually filtered.
