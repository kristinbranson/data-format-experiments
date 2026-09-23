# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats `/app/data` as a DANDI layout of one NWB file per session under `sub-<id>/`. It enumerates every file with a single sorted glob, then opens each file exactly once with `pynwb.NWBHDF5IO` inside a `with` block and reads everything it needs from that one handle (`nwb.trials`, `nwb.units`, `nwb.acquisition['BehavioralEvents']`, `nwb.acquisition['BehavioralTimeSeries']`, `nwb.subject`, `nwb.identifier`). `h5py` is never imported. `--sample` truncates the file list to the first 2 files; `--full` (default) processes all 174. Results are accumulated in a list of per-session dicts and stitched together in `main()`.

ii.
```python
DATA_DIR = Path("/app/data")
...
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
if args.sample:
    files = files[:2]
print(f"Found {len(files)} NWB sessions; processing with pynwb", flush=True)
```

```python
def convert_session(path: Path, show_processing: bool = False) -> tuple[dict | None, Counter]:
    """Load and convert one NWB session."""
    audit = Counter()
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        classifications = np.asarray(nwb.units["classification"][:]).astype(str)
        ...
        events = nwb.acquisition["BehavioralEvents"].time_series
        ...
        tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` is DANDI:000363 v0.230822.0128 (Mesoscale Activity Map Dataset): one `dandiset.yaml` plus 174 NWB 2.x files under 28 `sub-<id>/` directories. Each file is one behavior+ecephys session... All content inspection used `pynwb.NWBHDF5IO(..., load_namespaces=True)`; `h5py` was not used." Step 10 Check 3 adds that the reference repository loads DataJoint MATLAB exports rather than NWB, so its loader cannot be reused; the converter uses "the equivalent NWB objects through mandatory `pynwb`" and "Difference is source API only." Sorting the glob makes session order deterministic.

## 1-b. How are the data split into subjects (mice)?

i. Each session's animal is read from `nwb.subject.subject_id` (a numeric string such as `'440956'`). After all sessions are converted, the unique subject ids are sorted to form `subjects`, and `subject_idx` holds each session's index into that list, in the same order as `neural`/`input`/`output`. The AI verified 28 subjects, matching the data paper.

ii.
```python
subject = str(nwb.subject.subject_id)
session = {..., "subject": subject, ...}
```

```python
subjects = sorted({x["subject"] for x in converted})
subject_lookup = {x: i for i, x in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_lookup[x["subject"]] for x in converted], dtype=np.int32),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`subject.subject_id` → `subjects`, `subject_idx`; Stable sorted subject list; session lookup indices; NWB metadata; 28 subjects expected." Step 9 consistency table records "Subjects | 28 | all dataset | 28 | 28 | Yes". The subject id is the canonical per-file animal identifier and matches the containing `sub-<id>/` directory name, so no separate grouping step is needed.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no splitting or grouping is performed. The session label is `nwb.identifier` (e.g. `SC015_20190207_120657_s1`), stored per session and carried into `metadata['session_info']` along with `source_file`, `native_trials`, `included_trials`, `good_units`, the tongue quantiles, and auto/free-water counts. Session order in the output follows the sorted file list. One of the 174 files is dropped (zero classifier-good units), leaving 173 sessions.

ii.
```python
session_id = str(nwb.identifier)
...
"info": {
    "session_id": session_id,
    "source_file": str(path),
    "native_trials": int(len(trials)),
    "included_trials": int(len(trial_inds)),
    "good_units": int(len(unit_inds)),
    "tongue_y_q40": float(q40),
    "tongue_y_q60": float(q60),
    "auto_water_trials_included": int(tr.auto_water.sum()),
    "free_water_trials_included": int(tr.free_water.sum()),
},
```

```python
"metadata": { ..., "session_info": [x["info"] for x in converted], ... }
```

iii. CONVERSION_NOTES Step 2: "174 NWB 2.x files ... Each file is one behavior+ecephys session". Step 5 Key Decision 1: "Session inclusion: Process all 174 NWBs, then omit the single session with zero classifier-good neurons; expected 173 sessions, matching the paper." Step 4 records the corresponding consistency check: "174 files; 173 with >=1 good unit | Paper: 173 behavioral sessions | Drop sole zero-good-unit session; exact match."

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table via `nwb.trials.to_dataframe()`, one row per behavioural trial, with `start_time`/`stop_time` defining trial bounds. The AI asserts that the number of `go_start_times` events equals the number of trial rows, so the go cue used for alignment maps one-to-one onto trial rows. Output `neural`/`input`/`output` lists are then built with one entry per retained trial row.

ii.
```python
trials = nwb.trials.to_dataframe()
...
trial_starts = trials.start_time.to_numpy(dtype=np.float64)
trial_stops = trials.stop_time.to_numpy(dtype=np.float64)
...
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
if len(go_all) != len(trials):
    raise ValueError(f"{path.name}: go cue/trial count mismatch")
```

```python
for j in range(len(trial_inds)):
    neural.append(np.ascontiguousarray(rates[:, j, :], dtype=np.float32))
    inputs.append(np.ascontiguousarray(np.vstack((tone_time[j], laser[j])), dtype=np.float32))
    out = np.empty((4, len(CENTERS)), dtype=np.int8)
    ...
```

iii. CONVERSION_NOTES Step 2 documents the trials table columns and notes that `go_start_times` has exactly one event per trial, whereas `sample_start_times` can have "1–16 sample-start events per trial because early licking can trigger replay" (Step 4). The trials table is therefore used as the authoritative trial boundary definition and the go-cue count is used as a hard cross-check.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at "is there valid neural data for this trial", plus a deliberate decision to apply **no** behavioural filter.

1. **Per-unit valid-trial masks.** Each classifier-good unit carries a Boolean `is_good_trials` vector. The AI discovered this vector is indexed by the unit's `obs_intervals` (insertion-local), not by the session trial table, so it maps each observation interval to a trial row by exact `start_time`/`stop_time` matching and expands the mask to full session length (trials outside `obs_intervals` become `False`). A trial is kept only if it is valid for **every** retained unit (`np.all(..., axis=0)`). This removed 1,569 trials.
2. **All-population-zero trials.** After binning, any trial where no retained neuron fired a single spike across the whole 4 s window is dropped as an unrecorded period. This removed 2,423 trials (essentially the free-water/no-spike trials).

Early-lick, miss, ignore, photostimulation, auto-water and free-water trials are all deliberately **retained**. A session with <2 surviving trials raises an error (never triggered). Net: 94,370 native trials in the 173 retained sessions → 90,378 kept (95.8%).

ii.
```python
def full_unit_trial_mask(units, unit_i, trial_starts, trial_stops):
    """Expand a unit's insertion-local good-trial vector to session trials."""
    local_good = np.asarray(units["is_good_trials"][unit_i], dtype=bool)
    intervals = np.asarray(units["obs_intervals"][unit_i], dtype=np.float64)
    ...
    trial_i = np.searchsorted(trial_starts, intervals[:, 0], side="left")
    ...
    if not (np.allclose(trial_starts[trial_i], intervals[:, 0], atol=1e-5) and
            np.allclose(trial_stops[trial_i], intervals[:, 1], atol=1e-5)):
        raise ValueError(f"unit {unit_i}: observation intervals do not map to trials")
    full = np.zeros(len(trial_starts), dtype=bool)
    full[trial_i] = local_good
    return full
```

```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
trial_mask = np.all(unit_valid, axis=0)
trial_inds = np.flatnonzero(trial_mask)
if len(trial_inds) < 2:
    raise ValueError(f"{path.name}: fewer than two jointly valid trials")
audit["trials_native"] += len(trials)
audit["trials_invalid_period"] += int(np.sum(~trial_mask))
```

```python
# A small number of NWB observation intervals extend past the actual
# final spike timestamp (an export edge case). A completely silent
# hundreds-neuron population over four seconds marks an unrecorded trial,
# not physiological silence, so exclude it as an invalid data period.
recorded_trial = np.any(rates != 0, axis=(0, 2))
if np.any(~recorded_trial):
    audit["trials_all_neural_zero_excluded"] += int(np.sum(~recorded_trial))
    trial_inds = trial_inds[recorded_trial]
    ...
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "Trial validity: Keep trials for which every retained unit's `is_good_trials` is true. This excludes invalid recording periods while preserving a constant neuron set. Keep early, ignore, miss, stimulation, auto-water and free-water trials because requested targets/inputs require broad behavioral coverage." Step 4 quantifies it: "68,888/69,453 good units valid on all trials; 565 have partial masks (minimum 69.6% valid) ... Retain otherwise-good units but exclude any trial not valid for every retained unit. This honors explicit valid-period metadata and avoids fabricating neural values; all sessions retain >=2 trials." Step 1/Step 4 explain the refusal to copy the reference's `get_regular_trial_mask`: "the requested decoder explicitly requires early-lick, miss/no-lick, and photostimulation labels, so excluding those trials would destroy required output/input classes." Step 7/Step 10 document the second filter as an edge case found during the sample run: "one nominally observed terminal trial occurred after all units' final spikes and produced an all-zero population, so it is now excluded as an invalid recording period."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, session-absolute seconds) restricted to units with `units/classification == 'good'`, together with `BehavioralEvents/go_start_times` which supplies the per-trial bin edges. No other neural representation is used.

ii.
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
...
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
...
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. CONVERSION_NOTES Step 5 mapping table: "`units.spike_times` for `classification == \"good\"` → `neural`". Step 1 notes the data are electrophysiology spike times so "delta-F/F is not applicable", and the reference's `sliding_histogram`/`process_one_area` implement the "same spike-count/rate principle".

## 2-b. How is the `neural` data processed?

i. Spike counts per bin are computed with one `np.searchsorted` per unit over the flattened `(n_trials, 81)` array of absolute bin edges; differencing adjacent edge positions gives per-bin counts; dividing by 0.05 s converts to Hz. Bins are half-open `[left, right)`. Results are stored `float32`. No smoothing, no normalisation, no baseline subtraction, no z-scoring.

ii.
```python
BIN_SIZE_S = 0.050
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
...
# Vectorized binning: one searchsorted call per unit returns all trial edges.
rates = np.empty((len(unit_inds), len(trial_inds), len(CENTERS)), dtype=np.float32)
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. CONVERSION_NOTES Step 5: "histogram absolute spikes into 80 half-open 50-ms bins from -2.5 to +1.5 s; divide counts by 0.05 s to Hz; float32 `(neurons,80)`" with reference functions "`sliding_histogram`, `process_one_area`" — "Same spike-count/rate principle; task-mandated bins." Step 10 Check 6: "Both count spikes in half-open windows and divide by width. Reference movement processing uses overlapping 40-ms/3.4-ms bins; required decoder bins are non-overlapping 50-ms, exactly 80 over [-2.5,+1.5)." Step 10 Check 2 records an independent `np.histogram` spot check on raw NWB passing `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if and only if `units/classification == 'good'` — the released output of the region-specific 15-metric spike-sorting QC classifier. No individual QC metric is re-thresholded. Units whose `classification` is NaN become the string `'nan'` under `.astype(str)` and are therefore excluded. A session with zero good units returns `None` and is dropped (exactly one such session). Retained: 69,453 of 272,227 units across 173 sessions, median 390/session. The per-unit `is_good_trials` mask is used to filter trials rather than units (see 1-e), so the neuron set is constant within a session.

ii.
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
if len(unit_inds) == 0:
    audit["sessions_zero_good_units"] += 1
    return None, audit
```

```python
"unit_filter": "NWB units classification == 'good' (paper's region-specific 15-metric classifier)",
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "Neuron curation: Use supplied multimetric classifier label exactly (`classification == good`); no individual QC thresholds because the QC white paper explicitly rejects that strategy." Step 3 elaborates: "Fifteen jointly informative QC metrics fed five region-specific logistic-regression classifiers trained on blinded manual labels; individual metric thresholding is explicitly discouraged because distributions overlap." Step 2 records the category counts ("69,453 good, 200,922 unlabelled, 1,852 NaN") and identifies `sub-440958_ses-20190216T162508` as the sole session with zero good units. Step 4/Step 9 flag the residual 0.70% discrepancy versus the paper's 69,943 and decline to repair it: "Use release-native labels; 490-unit (0.70%) discrepancy is attributable to released-version/export differences and cannot be repaired without inventing labels."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. All NWB streams share one session-absolute clock, so the AI takes each trial's `go_start_times` timestamp and adds the fixed relative edge grid to produce the absolute bin edges for that trial; spikes are binned against those absolute edges directly. No resampling, interpolation, or per-stream offset correction is applied. The same `absolute_edges` / `go[:, None] + CENTERS[None, :]` construction is reused for the photostimulation input, the tone-time input, and the tongue output, guaranteeing that bin *k* means the same interval in all four streams.

ii.
```python
go_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
if len(go_all) != len(trials):
    raise ValueError(f"{path.name}: go cue/trial count mismatch")
...
go = go_all[trial_inds]
absolute_edges = go[:, None] + EDGES[None, :]
```

```python
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. CONVERSION_NOTES Step 10 Check 5: "Reference spikes are go-relative and explicitly shifts lick/laser times by go. Converter subtracts the absolute NWB go timestamp from every stream, which is equivalent." Step 4 final understanding: "align raw spikes and timestamped behavior directly to each NWB go onset". `metadata['temporal_alignment_event'] = 'go cue onset'`, `off_start = -2.5`, `off_end = 1.5`. Step 10 Check 2 verified "First/last centers are exactly -2.475/+1.475 and 80-bin endpoint checks pass."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 non-overlapping half-open bins spanning `[-2.5, +1.5)` s relative to the go cue, i.e. centres from −2.475 s to +1.475 s. The grid is defined once at module level as 81 edges and reused for every trial and every session, so `n_timepoints` is exactly 80 everywhere. Spikes are binned once, at that resolution, from raw spike times — there is no intermediate representation and therefore no rebinning of the neural data. The 300 Hz video is resampled down to the same 50 ms grid (see 8-b/8-d). The reference paper's 40 ms / 3.4 ms sliding window is explicitly not used.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```

```python
"time_bin_size": 50.0,
"n_timepoints": len(CENTERS),
"time_bin_units": "ms",
"bin_interval_convention": "left-closed, right-open; [-2.5, 1.5) s relative to go cue",
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "Alignment and endpoints: edges are `np.arange(-2.5,1.5+0.05,0.05)` and bins are `[left,right)`, yielding exactly 80 centers from -2.475 to +1.475 s. This removes endpoint ambiguity." Step 3/Step 4: "Movement paper bins spikes with 40-ms width/3.4-ms stride, but the decoder task mandates 50-ms width; this is a justified task override" and "Required non-overlapping 50-ms bins supersede paper analysis bins."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the auditory sample/tone onsets) and the trial's `go_start_times`. Because an early lick replays the sample epoch, a trial can contain 1–16 sample events; the AI takes the **last** sample onset at or before the go cue, and validates that this onset does not fall before the trial's `start_time`.

ii.
```python
def last_sample_before_go(sample_times, trial_starts, go_times):
    """Last sample/tone onset in each trial at or before its go cue."""
    inds = np.searchsorted(sample_times, go_times, side="right") - 1
    if np.any(inds < 0):
        raise ValueError("A trial has no sample onset before go cue")
    tone = sample_times[inds]
    if np.any(tone < trial_starts):
        bad = np.flatnonzero(tone < trial_starts)[:5]
        raise ValueError(f"Sample onset falls before trial start for trials {bad.tolist()}")
    return tone
```

```python
sample_times = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_all = last_sample_before_go(sample_times, trial_starts, go_all)
...
tone = tone_all[trial_inds]
```

iii. CONVERSION_NOTES Step 4: "1–16 sample-start events per trial because early licking can trigger replay ... Use the last sample onset before go as onset of the completed instruction epoch; preserves nominal relationship to go and avoids using abandoned/replayed epochs." Step 5 Key Decision 5 repeats this: "Select the last sample start at or before go within the trial—the completed instruction epoch that causally precedes that go cue." `metadata['tone_onset_definition']` states the same. Step 5 sanity checks include "normal completed sample onset clusters at -1.85 s", which follows from the task structure (0.65 s sample + 1.2 s delay).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute time of each bin centre (`go + centre`) minus the tone onset time. This produces a continuous, monotonically increasing, time-varying value per bin, stored `float32` as row 0 of the `(2, 80)` input array. No discretisation, clipping, or normalisation is applied. Observed range over the full dataset is [−1.5, 11.9] s (large positive values on early-lick-replay trials with long sample/delay repetitions).

ii.
```python
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]
```

```python
inputs.append(np.ascontiguousarray(np.vstack((tone_time[j], laser[j])), dtype=np.float32))
```

iii. CONVERSION_NOTES Step 5 mapping table: "Bin centers and final pre-go `sample_start_times` event → `input[0]`; Continuous seconds since completed tone/sample onset: `(go + bin_center) - tone_onset`; Time-varying float32." Step 9 checks the resulting range: "[-1.5, 11.9] at bin centers | Yes; long positive times are early-lick replay delays." Step 10 Check 2 verified it against an independent reconstruction from raw NWB event timestamps with `np.allclose`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same bin grid used for the firing rates: `go[:, None] + CENTERS[None, :]` where `CENTERS` are the midpoints of the same `EDGES` array passed to `np.searchsorted` for the spikes. Both quantities are computed from the same `go` vector after trial selection, so element *k* of the input row corresponds to the same absolute interval as column *k* of the neural matrix. No interpolation is needed because event and spike timestamps are already on the same session-absolute clock.

ii.
```python
EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
...
absolute_edges = go[:, None] + EDGES[None, :]      # neural
tone_time = go[:, None] + CENTERS[None, :] - tone[:, None]   # input[0]
```

iii. CONVERSION_NOTES Step 10 Check 5: "Converter subtracts the absolute NWB go timestamp from every stream, which is equivalent [to the reference shifting lick/laser times by go]." Step 5 Key Decision 4 fixes the shared edge/centre convention. Step 7 plot review: "Both plots show the go line at 0 ... linear tone-time input ... No alignment artifact was seen."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` — the explicit laser on/off timestamp pairs, in session-absolute seconds. The AI checks that the two series have equal length. It does **not** use the trials-table `photostim_onset` / `photostim_duration` columns (which are `'N/A'`-padded strings relative to trial start); the trials-table `auto_water`/`free_water`/`photostim_*` columns are only read for metadata bookkeeping. I verified numerically that the two sources are identical: in `sub-440956_ses-20190207T120657`, all 78 table onsets equal the 78 event start times to the printed precision, and `stop − start` equals `photostim_duration` (0.5 s).

ii.
```python
laser_starts = np.asarray(events["photostim_start_times"].timestamps[:], dtype=np.float64)
laser_stops = np.asarray(events["photostim_stop_times"].timestamps[:], dtype=np.float64)
if len(laser_starts) != len(laser_stops):
    raise ValueError(f"{path.name}: photostim start/stop mismatch")
laser = photostim_bins(laser_starts, laser_stops, absolute_edges)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`photostim_start_times/stop_times` → `input[1]`; Binary 1 where a 50-ms bin overlaps laser-on interval, otherwise 0; Reference shifts laser events by go cue; Time-varying float32." Step 1 records that the reference code's stimulation fields are "power, type, onset, offset, with onset/offset aligned to go cue", so the AI chose the timestamped event series as the direct NWB equivalent. Step 10 Check 7: "Tone and laser use NWB event timestamps ... binary laser is marked by positive-duration bin overlap."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary `(n_trials, 80)` time series. For each laser interval in the session, every bin whose half-open interval has positive-duration overlap with `[onset, offset)` is set to 1 (`left < offset and right > onset`); all other bins stay 0. Trials with no laser event are all-zero with no special-casing. This is an "any overlap" convention rather than "bin centre inside the interval": for the standard 0.5 s stimulation it marks 10 or 11 bins depending on phase (I measured 849 vs 780 bins marked in one session compared with a centre-in-interval rule), but the *set of trials* marked is identical (78 vs 78, and 149 vs 149 in a second session). Stored `float32`, range [0, 1].

ii.
```python
def photostim_bins(starts, stops, absolute_edges):
    """Mark bins with positive-duration overlap with any photostimulation."""
    n_trials = absolute_edges.shape[0]
    result = np.zeros((n_trials, len(CENTERS)), dtype=np.float32)
    left, right = absolute_edges[:, :-1], absolute_edges[:, 1:]
    # There are comparatively few laser intervals; each operation is vectorized
    # over every trial and time bin.
    for onset, offset in zip(starts, stops):
        result[np.logical_and(left < offset, right > onset)] = 1.0
    return result
```

iii. CONVERSION_NOTES Step 5: "Binary 1 where a 50-ms bin overlaps laser-on interval, otherwise 0". Step 3 supplies the expected prevalence and timing: "~25% randomly interleaved subset; late final 0.5 s of delay". Step 5 sanity checks include "laser input agrees with late-delay timing and is off after go in standard photoinhibition trials"; Step 9 confirms the observed range is [0, 1]. This also satisfies the task instruction that a time-type input be represented as a binary time series.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The laser timestamps are absolute and are compared directly against `absolute_edges` — the very same `(n_trials, 81)` edge array that is passed to `np.searchsorted` for the spike binning. `absolute_edges` is also subset by the all-zero-trial mask at the same time as `rates`, so trial indexing stays in lock-step. No interpolation or offset is applied.

ii.
```python
absolute_edges = go[:, None] + EDGES[None, :]
...
laser = photostim_bins(laser_starts, laser_stops, absolute_edges)
...
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
```

```python
    absolute_edges = absolute_edges[recorded_trial]
    laser = laser[recorded_trial]
    tone_time = tone_time[recorded_trial]
    rates = rates[:, recorded_trial, :]
```

iii. CONVERSION_NOTES Step 10 Check 5, as for 3-c: every stream is placed on the absolute go-cue-referenced grid, which is "equivalent" to the reference code's practice of shifting laser and lick times by the go cue. Step 7's processing plots show "correctly bounded stimulation" against the go line at 0.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no explicit choice column, so choice is derived from the trials-table pair `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`): a hit means the animal licked the instructed side, a miss the opposite side, an ignore means no lick. As an independent check (not used to build the output), the AI also computes the choice implied by the first `left_lick_times`/`right_lick_times` event in `[go, go + 1.5)` and audits agreement — 90,096/90,378 (99.69%) match.

ii.
```python
def derive_choice(instruction: str, outcome: str) -> int:
    """0 left, 1 right, 2 no lick from task instruction and outcome."""
    if outcome == "ignore":
        return 2
    instructed = 0 if instruction == "left" else 1
    return instructed if outcome == "hit" else 1 - instructed
```

```python
left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
audit["choice_lick_matches"] += int(np.sum(choices == lick_choices))
audit["choice_lick_mismatches"] += int(np.sum(choices != lick_choices))
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Choice derivation: Trial table contains instruction/outcome, not explicit choice. In a two-port task, outcome deterministically maps instruction to actual choice (same side for hit, opposite for miss, none for ignore); raw response lick events provide an independent check." Step 10 Check 10: "Choice matched first response-window lick on 90,096/90,378 trials (99.69%). The 282 mismatches are expected event ambiguity (multiple/carry-over lick events near response boundaries); authoritative task outcome+instruction is retained because it defines scored behavioral choice."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, as `int8`, and broadcast across all 80 time bins into row 0 of the `(4, 80)` per-trial output array. `output_values[0] = ['left', 'right', 'no lick']`. Full-dataset distribution: left 0.428, right 0.422, no lick 0.149.

ii.
```python
instructions = tr.trial_instruction.astype(str).to_numpy()
outcomes_str = tr.outcome.astype(str).to_numpy()
choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)], dtype=np.int8)
```

```python
out = np.empty((4, len(CENTERS)), dtype=np.int8)
out[0] = choices[j]
```

```python
"output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
"output_values": [
    ["left", "right", "no lick"],
    ...
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "Outputs as time-varying matrices: Broadcast trial-level choice/outcome/early labels across time and retain tongue as truly time-varying, producing uniform integer `(4,80)` arrays compatible with joint decoder training." Code order (`left, right, no lick`) follows the Decoder Task specification. Step 9 records the distribution as "[0.428,0.422,0.149] | Plausible/balanced sides".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`. No derivation.

ii.
```python
outcomes_str = tr.outcome.astype(str).to_numpy()
```

iii. CONVERSION_NOTES Step 2 enumerates the column and its native totals ("outcome hit 65,254, miss 15,641, ignore 14,095"). Step 5 mapping table: "`trials.outcome` → `output[1]` outcome; Map ignore/miss/hit to 0/1/2; `correctness` conventions in preprocessing". Step 1 notes the reference code's equivalent encoding ("correctness 1=correct/free-water, 0=error, -1=no response").

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore→0`, `miss→1`, `hit→2` (the order given in the Decoder Task), stored `int8` and broadcast across all 80 bins into row 1. `output_values[1] = ['ignore', 'miss', 'hit']`. Full-dataset distribution: ignore 0.149, miss 0.166, hit 0.685.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcomes = np.asarray([outcome_map[x] for x in outcomes_str], dtype=np.int8)
```

```python
out[1] = outcomes[j]
```

iii. CONVERSION_NOTES Step 9 cross-checks the retained distribution against the native release ("all-native [ignore .148, miss .165, hit .687] | [.149,.166,.685] | Yes"). Step 10 Check 9 goes further and reconstructs the paper's performance statistic: "Converted control/no-early response accuracy is 81.67% (43,680 hit/9,801 miss), close to paper mean 84%; including ignore in the denominator would incorrectly give 68.65%."

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column, which holds `'early'` / `'no early'`. No derivation.

ii.
```python
early_str = tr.early_lick.astype(str).to_numpy()
```

iii. CONVERSION_NOTES Step 2 records the column and native counts ("early 10,805, no-early 84,185"). Step 5 mapping table: "`trials.early_lick` → `output[2]` early lick; Map no early/early to 0/1; `early_lick_trials`". The AI notes in Step 1/Step 4 that the reference analyses *exclude* early-lick trials, but that they must be retained here because early lick is a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early'→0`, `'early'→1`, stored `int8` and broadcast across all 80 bins into row 2. `output_values[2] = ['no', 'yes']`. Full-dataset distribution: no 0.884, yes 0.116.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.asarray([early_map[x] for x in early_str], dtype=np.int8)
```

```python
out[2] = early[j]
```

iii. CONVERSION_NOTES Step 5 Key Decision 7 (broadcast per-trial labels across time). Step 9: "Early distribution | excluded in many paper analyses | early flag | all-native [.886,.114] | [.884,.116] | Yes". Step 12 explicitly revisits this output because its accuracy ratio (1.505× chance) was closest to the investigation threshold and concludes "raw-label checks, class support, small generalization gap, and 0.7527 validation accuracy support retaining the specified representation."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with explicit `timestamps` (~294 Hz). Column 1 (`y`) supplies the value; column 2 (the DeepLabCut likelihood) decides visibility. The side camera is used, matching the movement paper.

ii.
```python
tracking = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
track_t = np.asarray(tracking.timestamps[:], dtype=np.float64)
track_data = np.asarray(tracking.data[:], dtype=np.float64)
raw_y, likelihood = track_data[:, 1], track_data[:, 2]
```

iii. CONVERSION_NOTES Step 2: "`acquisition/BehavioralTimeSeries`: side-view jaw, nose, and tongue `(x, y, likelihood)` at explicit timestamps. Tongue series exists in every file." Step 5 mapping table cites the reference basis: "Movement paper side-view DeepLabCut markers". Step 3: "Video/keypoints are synchronized at 300 Hz. The movement paper used side view".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. **Nearest-frame sampling, not averaging.** For every (trial, bin) the AI computes the absolute bin-centre time `go + centre` and finds the single nearest camera frame by `searchsorted` + left/right comparison. That frame's `y` is the bin's value and that frame's `likelihood` decides visibility, with threshold 0.9. Frames with non-finite `y` or `likelihood` are excluded from the percentile pool. No smoothing, interpolation, or averaging over the ~15 frames that fall inside each 50 ms bin; no imputation of occluded positions.

Consequence I measured on `sub-440956_ses-20190207T120657`: the threshold itself is almost irrelevant (10.586% of frames visible at 0.5 vs 10.517% at 0.9 — the likelihood is strongly bimodal), but the nearest-frame rule marks only 12.6% of bins visible, whereas a rule that counts a bin as visible if *any* of its frames is visible marks 20.6%. Across the full dataset the AI's "not visible" class is 84.1% of all bins.

ii.
```python
def nearest_indices(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Indices of nearest sorted timestamps for each query time."""
    right = np.searchsorted(timestamps, query, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query - timestamps[left]) <= np.abs(timestamps[right] - query)
    return np.where(choose_left, left, right)
```

```python
TONGUE_VISIBLE_LIKELIHOOD = 0.9
...
visible = np.isfinite(raw_y) & np.isfinite(likelihood) & (likelihood >= TONGUE_VISIBLE_LIKELIHOOD)
if not np.any(visible):
    raise ValueError(f"{path.name}: no visible tongue frames")
q40, q60 = np.quantile(raw_y[visible], [0.4, 0.6])
query = go[:, None] + CENTERS[None, :]
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
y = raw_y[track_idx]
tongue_visible = likelihood[track_idx] >= TONGUE_VISIBLE_LIKELIHOOD
```

iii. CONVERSION_NOTES Step 5 mapping table: "Nearest video sample at each bin center; likelihood <0.9 -> 3 not visible ... DLC likelihood is strongly bimodal near 0/1; 0.9 is a conservative standard visibility threshold." Step 3 explains the departure from the movement paper's handling: "The movement paper ... set occluded tongue positions to their mean for its regression. The present categorical target instead explicitly reserves class 3 for not visible, so occlusion must remain distinguishable rather than imputed." Step 7 reports the plot review: "bimodal DLC likelihood with a clear 0.9 threshold". No justification is given anywhere in the notes or trajectory for preferring nearest-frame sampling over averaging the frames within each bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles (`np.quantile(..., [0.4, 0.6])`) of `y` over **all visible frames of the whole session** (not restricted to trial windows or to retained trials) form two edges. Classes: `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, `3` if the sampled frame is not visible. The array is initialised to 3 and each visible class is written by a masked assignment, so any unhandled case remains 3. The per-session `q40`/`q60` are recorded in `session_info`. Resulting full-dataset distribution: 0.062 / 0.032 / 0.065 / 0.841 — i.e. 39% / 20% / 41% *within the visible bins*, matching the requested 40/20/40 split by construction.

ii.
```python
q40, q60 = np.quantile(raw_y[visible], [0.4, 0.6])
...
tongue_class = np.full(query.shape, 3, dtype=np.int8)
tongue_class[tongue_visible & (y < q40)] = 0
tongue_class[tongue_visible & (y >= q40) & (y <= q60)] = 1
tongue_class[tongue_visible & (y > q60)] = 2
```

```python
"output_values": [ ..., ["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"] ],
"tongue_visibility_likelihood_threshold": TONGUE_VISIBLE_LIKELIHOOD,
"tongue_percentile_scope": "all visible side-camera tongue frames within each session",
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "Tongue visibility/discretization: Use DLC likelihood >=0.9. Compute q40/q60 only from visible session frames (otherwise hidden placeholder coordinates would dominate), with exact boundary convention requested: class 0 `<q40`, class 1 `q40<=y<=q60`, class 2 `>q60`, class 3 hidden." The mapping table adds "Percentiles computed before trial selection over all visible session frames, as requested." Step 5 sanity check: "visible class distribution is approximately 40/20/40 by construction over session frames"; Step 9 confirms "[.062,.032,.065,.841] | Yes; visible subclasses close 40/20/40 conditional proportions."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and events, so the AI builds the query grid as `go[:, None] + CENTERS[None, :]` — the midpoints of the identical edge array used for the spike binning — and picks the nearest camera frame to each query time. Bin *k* of the tongue output is therefore centred on the same instant as bin *k* of the firing rates. No interpolation or offset correction.

I measured the nearest-frame offset: median 0.9 ms, and >25 ms (half a bin) for only 0.0–0.23% of bins depending on session. The residual cases arise because the video is trial-gated and has gaps of up to ~1.7 s in the inter-trial interval; in those bins the nearest-frame rule silently imports a frame from up to ~0.36 s away instead of declaring the bin unobserved.

ii.
```python
query = go[:, None] + CENTERS[None, :]
track_idx = nearest_indices(track_t, query.ravel()).reshape(query.shape)
y = raw_y[track_idx]
tongue_visible = likelihood[track_idx] >= TONGUE_VISIBLE_LIKELIHOOD
```

iii. CONVERSION_NOTES Step 10 Check 5 (all streams referenced to the absolute go timestamp). Step 7 plot review: "Trial-level outputs are constant and tongue is time-varying when visible. No alignment artifact was seen." Step 10 Check 2: an independently reconstructed "nearest-frame tongue category" from raw NWB passes `np.allclose` against the pickle. Step 5 sanity check: "class 3 dominates pre-go and visible tongue increases after go."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five classes of defect, handled in three different ways.

**Excluded:**
- Session never quality-controlled (`classification` is NaN for all 1,852 units in `sub-440958_ses-20190216T162508`): `.astype(str)` renders NaN as `'nan'`, so no unit matches `'good'`, and `convert_session` returns `None`. Counted in `audit['sessions_zero_good_units']`.
- Trials outside a unit's `obs_intervals`, or flagged `is_good_trials == False` (1,569 trials).
- Trials where the entire retained population has zero spikes over the 4 s window (2,423 trials), diagnosed as observation intervals extending past the last exported spike.

**Represented explicitly:** frames with non-finite `y`/`likelihood` or likelihood below threshold are excluded from the percentile pool, and any bin sampling such a frame gets the dedicated `3 = not visible` class rather than an imputed value.

**Repaired:** `is_good_trials` turned out to be indexed by `obs_intervals` rather than by the session trial table; instead of assuming, the AI maps observation intervals back onto trial rows by exact start/stop matching and validates the mapping with `np.allclose(..., atol=1e-5)`, raising if it fails.

**Fail-loud rather than degrade:** several conditions raise `ValueError` instead of dropping the session — trial/go-cue count mismatch, laser start/stop length mismatch, no sample onset before a go cue, a tone before trial start, fewer than 2 surviving trials, no visible tongue frames. None fired on the full dataset.

ii.
```python
classifications = np.asarray(nwb.units["classification"][:]).astype(str)
unit_inds = np.flatnonzero(classifications == "good")
if len(unit_inds) == 0:
    audit["sessions_zero_good_units"] += 1
    return None, audit
```

```python
    if len(local_good) != len(intervals):
        raise ValueError(f"unit {unit_i}: is_good_trials/obs_intervals mismatch")
    trial_i = np.searchsorted(trial_starts, intervals[:, 0], side="left")
    if np.any(trial_i >= len(trial_starts)):
        raise ValueError(f"unit {unit_i}: observation interval outside trial table")
    if not (np.allclose(trial_starts[trial_i], intervals[:, 0], atol=1e-5) and
            np.allclose(trial_stops[trial_i], intervals[:, 1], atol=1e-5)):
        raise ValueError(f"unit {unit_i}: observation intervals do not map to trials")
```

```python
recorded_trial = np.any(rates != 0, axis=(0, 2))
if np.any(~recorded_trial):
    audit["trials_all_neural_zero_excluded"] += int(np.sum(~recorded_trial))
    ...
if len(trial_inds) < 2:
    raise ValueError(f"{path.name}: fewer than two recorded trials")
```

```python
visible = np.isfinite(raw_y) & np.isfinite(likelihood) & (likelihood >= TONGUE_VISIBLE_LIKELIHOOD)
if not np.any(visible):
    raise ValueError(f"{path.name}: no visible tongue frames")
```

iii. CONVERSION_NOTES Step 10 "Issues Found and Resolved": "`is_good_trials` initially assumed session length: raw inspection showed it is indexed by unit `obs_intervals`; expanded to the full trial table by exact start/stop matching and reran sample/full validation." / "One sample terminal trial and analogous full-data gaps had no spikes from any unit: added a population-zero acquisition-period exclusion; final validation has no warning and zero all-zero trials." / "Paper/native unit count differs by 0.70%: preserved release-native classifier labels rather than fabricating 490 labels." Step 3 supplies the rationale for not imputing occlusion: "The present categorical target instead explicitly reserves class 3 for not visible, so occlusion must remain distinguishable rather than imputed." Every exclusion is counted in a `Counter` audit that is printed at the end of the run and stored in `metadata['conversion_audit']`.

## 10-a. What are the most time-consuming steps of the code?

i. Measured from `/app/conversion_full_out.txt`: 174 sessions in 3.19 min wall clock including serialisation, roughly 0.6–2.3 s per session, scaling with unit count (e.g. 787 neurons → 2.27 s; 182 neurons → 0.64 s). The dominant costs are all I/O and per-unit work inside `convert_session`:

1. **Per-unit `spike_times` reads** — one indexed HDF5 read per good unit (`nwb.units["spike_times"][unit_i]`), i.e. ~400 reads per session, 69,453 across the dataset. This is the single largest component, and it is also where the per-unit `np.searchsorted` over the 81×n_trials edge array runs.
2. **Per-unit `is_good_trials` + `obs_intervals` reads** in `full_unit_trial_mask` — two more ragged reads per unit plus the `np.stack` of an `(n_units, n_trials)` bool matrix.
3. **Loading the whole tongue-tracking array** — `tracking.data[:]` is up to ~1.4 M × 3 float64 per session (~34 MB), read in full even though only ~0.2% of frames are ultimately sampled.
4. **Pickling the 11.79 GB result** at the end.

The script prints per-session elapsed time and a rolling ETA so the bottleneck is visible during the run.

ii.
```python
for i, path in enumerate(files):
    t0 = time.perf_counter()
    session, session_audit = convert_session(...)
    ...
    elapsed = time.perf_counter() - t0
    total_elapsed = time.perf_counter() - started
    eta = total_elapsed / (i + 1) * (len(files) - i - 1)
    print(f"[{i+1}/{len(files)}] {path.name}: {label}; {elapsed:.2f}s; ETA {eta/60:.1f} min", flush=True)
```

```python
for out_i, unit_i in enumerate(unit_inds):
    spikes = np.asarray(nwb.units["spike_times"][unit_i], dtype=np.float64)
    edge_indices = np.searchsorted(spikes, absolute_edges, side="left")
    rates[out_i] = np.diff(edge_indices, axis=1) / BIN_SIZE_S
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Naively looping over every unit × trial would require ~35 million Python-level histograms. Retaining all converted sessions in the required nested-list pickle also creates an unavoidable multi-GB in-memory result. Code speedups added: For each unit, a single vectorized `np.searchsorted` call evaluates all trial-bin edges, then `np.diff` yields counts. Laser overlap and tracking lookup are vectorized over trials/time. Arrays are float32/int8, and tracking/source objects are scoped to one NWB at a time." Step 7 estimated "~1.25 s/session ... ~3.6 min conservatively for 174 sessions", which the actual 3.19 min confirmed — comfortably inside the 15-minute budget, so no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The dominant loop was already vectorised: binning runs one `searchsorted` per *unit* over all trials and edges at once, rather than per unit × trial. What remains:

1. **`full_unit_trial_mask` called once per good unit** (list comprehension inside `np.stack`). Each call does its own HDF5 reads, `searchsorted`, and two `np.allclose` validations. Since `obs_intervals` is in practice shared by all units on a probe, this could read the intervals once and stack `is_good_trials` in a single vectorised operation.
2. **`first_response_lick_choice` called once per trial** — a pure-Python function with two scalar `searchsorted` calls, executed 90,378 times. Fully vectorisable with two array `searchsorted` calls; it produces only an audit statistic.
3. **`derive_choice` in a per-trial list comprehension**, plus the `outcome_map` / `early_map` comprehensions — all three are `np.where`-able over string arrays.
4. **`photostim_bins` loop over laser intervals**, each iteration touching the entire `(n_trials, 80)` array. Cost is O(n_events × n_trials × 80); restricting each event to the trials whose window can contain it (a `searchsorted` on `go`) would make it near-free.
5. **The final per-trial assembly loop** building `neural`/`inputs`/`outputs` lists, which does a `np.vstack` and an `np.empty` per trial. This one is largely forced by the target format's list-of-trials structure.

None of these are on the critical path relative to HDF5 reads, which is why the run still finished in 3.19 min.

ii.
```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
```

```python
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
```

```python
choices = np.asarray([derive_choice(a, b) for a, b in zip(instructions, outcomes_str)], dtype=np.int8)
outcomes = np.asarray([outcome_map[x] for x in outcomes_str], dtype=np.int8)
early = np.asarray([early_map[x] for x in early_str], dtype=np.int8)
```

```python
    for onset, offset in zip(starts, stops):
        result[np.logical_and(left < offset, right > onset)] = 1.0
```

iii. The AI documents only the vectorisation it *performed*, not the loops it left behind. CONVERSION_NOTES Step 6: "For each unit, a single vectorized `np.searchsorted` call evaluates all trial-bin edges, then `np.diff` yields counts. Laser overlap and tracking lookup are vectorized over trials/time." The inline comment in `photostim_bins` argues the remaining loop is cheap: "There are comparatively few laser intervals; each operation is vectorized over every trial and time bin." Step 7's timing table concluded the projected ~3.6 min was well under the 15-minute threshold, so the instruction's optimisation trigger never fired.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once and each session is processed in a single pass; the bin grid (`EDGES`, `CENTERS`) is a module-level constant computed once. The genuine repetitions are:

1. **`obs_intervals` is read and re-validated once per good unit** (69,453 times dataset-wide) even though it is shared across units from the same probe insertion — the reference reads it once per session. Two `np.allclose` calls over the full interval array are repeated with it.
2. **The likelihood threshold is applied twice** to two different views of the same data: once over all frames to build the percentile pool (`visible`), and once over the sampled frames (`tongue_visible`).
3. **`nwb.units[...]` column access** (`classification`, `anno_name`, `spike_times`, `is_good_trials`, `obs_intervals`) goes through the `DynamicTable` accessor separately each time rather than pulling the underlying buffers once.
4. **Choice is computed twice by two independent routes** — from `instruction × outcome` and from the lick event streams — but this is intentional (see 10-d).

Nothing is recomputed across sessions, and the tongue percentiles are per-session so they fit inside the single pass with no second read of the file.

ii.
```python
def full_unit_trial_mask(units, unit_i, trial_starts, trial_stops):
    local_good = np.asarray(units["is_good_trials"][unit_i], dtype=bool)
    intervals = np.asarray(units["obs_intervals"][unit_i], dtype=np.float64)
    ...
    if not (np.allclose(trial_starts[trial_i], intervals[:, 0], atol=1e-5) and
            np.allclose(trial_stops[trial_i], intervals[:, 1], atol=1e-5)):
```

```python
visible = np.isfinite(raw_y) & np.isfinite(likelihood) & (likelihood >= TONGUE_VISIBLE_LIKELIHOOD)
...
tongue_visible = likelihood[track_idx] >= TONGUE_VISIBLE_LIKELIHOOD
```

iii. CONVERSION_NOTES Step 6 frames the design as a single pass with per-session scoping: "Arrays are float32/int8, and tracking/source objects are scoped to one NWB at a time." The per-unit repetition is a deliberate consequence of the trial-validity policy documented in Step 4 — the AI chose not to assume `obs_intervals` is uniform across units and instead verifies the mapping for each one ("NWB `is_good_trials` is indexed by the unit's `obs_intervals`, not always by the full session trial table", docstring of `full_unit_trial_mask`). The notes do not otherwise call out repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items, all small:

1. **The lick-based choice cross-check.** `left_lick_times` and `right_lick_times` are read and `first_response_lick_choice` is called for all 90,378 trials, but `lick_choices` feeds only two audit counters — it never enters `output`. This is the largest discarded computation, and it is the one the instructions arguably asked for ("invent SANITY CHECKS"); it would be cheaper run once outside the conversion path.
2. **`trial_stops`** is read and used only inside `full_unit_trial_mask`'s `np.allclose` validation; it appears nowhere in the output.
3. **The full `unit_valid` matrix** — an `(n_units, n_trials)` bool array, up to ~900 × 800 per session — is materialised and then immediately collapsed by `np.all(..., axis=0)`. A running AND would avoid the allocation.
4. **`tracking.data[:]`** loads all three columns (`x`, `y`, `likelihood`) as float64 for up to 1.4 M frames; `tongue_x` is never used, and only ~0.2% of the frames are ever sampled.

Minor bookkeeping that is computed but not used by the decoder (though it is legitimately stored as metadata): `auto_water_trials_included`, `free_water_trials_included`, `tongue_y_q40`, `tongue_y_q60`, and the audit counters. `brain_region_idx` uses the fine Allen `anno_name` strings, producing 293 regions, and is not consumed by the decoder either — but the target format requires the field.

ii.
```python
left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
lick_choices = np.asarray([
    first_response_lick_choice(left_licks, right_licks, g) for g in go
], dtype=np.int8)
audit["choice_lick_matches"] += int(np.sum(choices == lick_choices))
audit["choice_lick_mismatches"] += int(np.sum(choices != lick_choices))
```

```python
unit_valid = np.stack([
    full_unit_trial_mask(nwb.units, i, trial_starts, trial_stops) for i in unit_inds
])
trial_mask = np.all(unit_valid, axis=0)
```

```python
track_data = np.asarray(tracking.data[:], dtype=np.float64)
raw_y, likelihood = track_data[:, 1], track_data[:, 2]
```

iii. The AI does not describe any of this as unnecessary; it presents the lick audit as a required validation. CONVERSION_NOTES Step 5 mapping table: choice is derived from instruction+outcome and then "confirm against first response-period lick events". Step 5 Key Decision 6: "raw response lick events provide an independent check." Step 5's planned checks list "Choice/lick audit: compare instruction+outcome-derived choice with first left/right lick during `[go, go+1.5)`; investigate mismatches", and Step 10 Check 10 reports the result (99.69% agreement, 282 mismatches attributed to "multiple/carry-over lick events near response boundaries"). The fine-grained region labels are justified in Step 5 as "Fine labels preserve maximum anatomical information."
