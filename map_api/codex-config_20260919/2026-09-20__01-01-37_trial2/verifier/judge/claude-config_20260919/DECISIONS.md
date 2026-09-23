# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the DANDI:000363 layout (`/app/data/sub-<id>/<session>.nwb`, one NWB file per session) as the complete inventory. It globs every `sub-*/*.nwb` path, sorts it for deterministic subject/session ordering, and opens each file exactly once with `pynwb.NWBHDF5IO` inside a `with` block. Within a session it reads `nwb.units` (spike times, `classification`, `anno_name`, `is_good_trials`, electrodes), `nwb.trials` (as a dataframe), `nwb.acquisition['BehavioralEvents']` (go/sample/lick/photostim event timestamps), `nwb.acquisition['BehavioralTimeSeries']` (side-camera tracking) and `nwb.subject`. No `h5py` access is used anywhere. All 174 files are opened; 173 are kept.

ii.
```python
DATA_DIR = Path("/app/data")

def nwb_files() -> list[Path]:
    """Return source sessions in deterministic subject/session order."""
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```
```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    classifications = as_string_array(nwb.units["classification"])
    ...
    trials_df_all = nwb.trials[:]
    events = nwb.acquisition["BehavioralEvents"].time_series
```
```python
for path in files:
    if max_sessions is not None and len(neural) >= max_sessions:
        break
    print(f"Processing {path.name}", flush=True)
    session, diagnostics = process_session(path, region_to_idx)
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` is DANDI:000363 version 0.230822.0128 (53.6 GB): 174 NWB files arranged as one directory per subject ... plus `dandiset.yaml`. All NWB inspection used `pynwb.NWBHDF5IO`; no direct HDF5 loading was used." The AI cross-checked the glob against the dandiset manifest: "The dataset manifest independently reports 174 files and 28 subjects, matching the PyNWB inventory." Step 10 check 5 justifies using the released NWB rather than the reference pipeline's per-probe MATLAB exports: "Both access Kilosort2 spikes, task events, video markers, and CCF anatomy; using released NWB avoids re-parsing old MATLAB containers."

## 1-b. How are the data split into subjects?

i. The subject of a session is read from `nwb.subject.subject_id` (a numeric string such as `'440956'`). The AI maintains a `subject_to_idx` dictionary populated in first-encounter order while iterating the sorted file list, appending each new id to `subjects` and recording the index in `subject_idx` (one entry per kept session). Because the file glob is sorted and the directory name is `sub-<subject_id>`, first-encounter order is the same as sorted order. Result: 28 subjects, 3-10 sessions each.

ii.
```python
"subject": str(nwb.subject.subject_id),
```
```python
subject = session["subject"]
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
...
subject_idx.append(subject_to_idx[subject])
```
```python
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int32),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`subject.subject_id` → `subjects`, `subject_idx` | Stable sorted unique strings and per-session indices | 28 subjects expected." Step 2 confirms 28 subjects with 3-10 sessions each from the data, and Step 3/4 confirm the papers also report 28 mice, so the NWB subject field is taken as authoritative without any extra grouping step.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no splitting or grouping is performed. Session order in the output follows the sorted file list. Each session is labelled by `nwb.identifier` (e.g. `SC015_20190207_120657_s1`) and recorded in `metadata['session_info']` together with the source path, trial count, neuron count and per-session diagnostics. One session is dropped (zero classifier-good units), giving 173 output sessions.

ii.
```python
return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```
```python
"info": {
    "session_id": str(nwb.identifier),
    "source_file": str(path),
    "n_trials": n_trials,
    "n_behavioral_table_trials": len(trials_df_all),
    "n_recorded_trial_entries": n_recorded_trials,
    "excluded_population_all_zero_trials": excluded_all_zero,
    "n_neurons": len(good),
    ...
}
```
```python
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```

iii. CONVERSION_NOTES Step 2: "Each NWB file is one recording session." Step 4: "one NWB equals one session; use all 173 sessions containing classifier-good, anatomically annotated units". The AI justified the 174→173 drop by consistency with the data paper: "Exclude only `sub-440958_ses-20190216T162508...`, yielding the published 173 sessions."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials[:]`), one row per behavioural trial, paired positionally with the go-cue timestamps in `BehavioralEvents/go_start_times`. The AI first establishes the number of *recorded* (ephys-covered) trials as the common length of each good unit's `is_good_trials` vector, checks that this length is identical across units and does not exceed the behavioural table, then takes the leading `n_recorded_trials` rows of the table and the leading `n_recorded_trials` go cues. It checks `len(go_times_all) >= n_recorded_trials` (a weaker check than one-go-cue-per-trial equality; the notes state this equality was verified separately for all 174 files).

ii.
```python
trials_df_all = nwb.trials[:]
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
if np.any(recorded_lengths != recorded_lengths[0]):
    raise ValueError(f"{path.name}: inconsistent recorded-trial counts across units")
n_recorded_trials = int(recorded_lengths[0])
if n_recorded_trials > len(trials_df_all):
    raise ValueError(f"{path.name}: unit trial count exceeds behavioral table")
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
events = nwb.acquisition["BehavioralEvents"].time_series
go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
if len(go_times_all) < n_recorded_trials:
    raise ValueError(f"{path.name}: {len(go_times_all)} go cues for {n_recorded_trials} recorded trials")
go_times = go_times_all[:n_recorded_trials]
```

iii. CONVERSION_NOTES Step 4: "All 174 NWBs have one go event per trial." Step 7: "Direct PyNWB inspection showed that eight NWBs contain more behavioral-table rows than each unit's `is_good_trials` vector/recording interval (for example, 480 behavior rows but 160 recorded entries); spikes stopped with the recording. The converter now uses the common per-unit validity-vector length as the source ephys trial count." The trials table is therefore used directly as the trial definition, with a length correction for sessions where behaviour outlasted the recording.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at *absence of neural data* rather than behaviour:

1. **Positional truncation to the recorded block.** The behavioural table and the go-cue array are truncated to the first `n_recorded_trials` rows, where `n_recorded_trials` is the length of the per-unit `is_good_trials` vector. This assumes the ephys-observed trials are always the *leading* contiguous block of the behavioural table. That drops 1,060 rows across 7 sessions.
2. **Population-all-zero rejection.** After binning, any trial in which *no* curated unit fired a single spike anywhere in the 4-s window is discarded as "no neural data". This removes 2,576 trials across 94 sessions.

A session is required to have >= 2 surviving trials (enforced with `raise`, not a skip). No behavioural quality filter is applied: early-lick, ignore/no-response, photostimulated, and free/auto-water trials are all retained. Final counts: 94,370 behavioural rows in retained sessions → 93,310 after truncation → 90,734 trials.

The AI did **not** use `units/obs_intervals`. Verified against the raw data: in 172 of the 173 retained sessions the observed block does start at row 0, so truncation is equivalent to an `obs_intervals` filter. In `sub-455219_ses-20190807T134913` the observed block is rows **125-629** of 630, not 0-504. There the AI keeps rows 0-504, of which only 380 are actually observed; the 125 unobserved leading rows are then removed by the all-zero filter, but **125 genuinely recorded trials (rows 505-629) are silently discarded**. This exactly accounts for the 126-trial gap between the AI (90,734) and the reference (90,860). Trial rows and go cues are truncated together, so there is no temporal misalignment — the loss is purely dropped data.

The all-zero rule is an empirical proxy for the reference's explicit `free_water` exclusion: of the 2,576 removed trials, 2,450 are free-water trials, 125 are the unobserved leading rows above, and 1 is a trailing empty window.

ii.
```python
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
...
go_times = go_times_all[:n_recorded_trials]
```
```python
rates = bin_spikes(nwb.units, good, go_times)
# A rare trailing interval can be present in `obs_intervals` and the
# validity-vector length despite containing no spikes from any unit.
# Treat population-all-zero windows as absent neural data, not trials.
neural_valid = np.any(rates != 0, axis=(1, 2))
excluded_all_zero = int((~neural_valid).sum())
if excluded_all_zero:
    rates = rates[neural_valid]
    trials_df = trials_df.iloc[np.flatnonzero(neural_valid)].copy()
    go_times = go_times[neural_valid]
    tone_onsets = tone_onsets[neural_valid]
    centers_abs = centers_abs[neural_valid]
n_trials = len(trials_df)
if n_trials < 2:
    raise ValueError(f"{path.name}: fewer than two trials with neural data")
```

iii. CONVERSION_NOTES Step 3/4/5 argue that the papers' behavioural exclusions cannot be applied here: "The papers' specific analyses usually exclude photoinhibition, free/auto-water, early-lick, and ignore trials ... Those exclusions conflict with required decoder variables/classes, so all native trials with valid go-cue alignment are retained." Key decision 1: "Exclude only the one zero-curated-unit session; retain every trial because filtering requested classes would make the decoder targets impossible. Require >= 2 trials." Key decision 5 explicitly rejects using the validity mask as a *filter*: "Do not apply `units.is_good_trials` as an additional undocumented filter" — it is used only for its length. Step 10: "2,576 entries across 94 sessions had no spikes from any curated unit (typically trailing after acquisition stopped); excluded as invalid neural periods ... removes all validator warnings."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (absolute, session-clock spike times), gated by `units/classification == 'good'`, together with `BehavioralEvents/go_start_times` which positions the bin edges. `units/anno_name` supplies the region label for each kept unit but does not affect the rates.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
```
```python
spike_column = units["spike_times"]
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
```
```python
go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`units.spike_times`, `units.classification`; `BehavioralEvents/go_start_times` → `neural`". Step 1 notes that the data are electrophysiology so "dF/F is not applicable", and Step 2 records that `nwb.units` contains "absolute spike times" — so spike times are the only neural representation available.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit the whole `(n_trials, 81)` matrix of absolute bin edges is passed to a single `np.searchsorted` call, adjacent positions are differenced to give per-bin spike counts, and counts are divided by the 0.05-s bin width. Bins are half-open `[edge_i, edge_{i+1})`. No smoothing, normalisation, baseline subtraction, or firing-rate threshold is applied. Output is `float32`, reshaped per trial to `(n_neurons, 80)`.

ii.
```python
def bin_spikes(units, good_indices, go_times) -> np.ndarray:
    """Return trial x neuron x time firing rates in Hz."""
    absolute_edges = go_times[:, None] + EDGES_REL[None, :]
    rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
    spike_column = units["spike_times"]
    for out_idx, unit_idx in enumerate(good_indices):
        spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
        edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
        rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
    return rates
```
```python
"neural": [rates[i] for i in range(n_trials)],
```

iii. CONVERSION_NOTES Step 1 identifies the reference function `sliding_histogram`, which "counts spikes in half-open bins and optionally divides by bin width". Step 4: "Use required nonoverlapping 50-ms half-open bins; divide counts by 0.05 s." Step 10 check 8: "Both use half-open spike intervals and convert counts to Hz ... Every converted value is exactly a nonnegative multiple of 20 Hz." Step 3 records that the movement paper's own 40-ms/3.4-ms sliding windows and the data paper's 200-ms/10-ms decoder bins are overridden by the explicit 50-ms requirement in the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept — the released region-specific logistic-regression QC classifier verdict. No thresholds are applied to any individual sorting metric, and no firing-rate floor is applied. A session with zero good units is skipped entirely (`return None`). The AI additionally *requires* every retained unit to carry a non-empty `anno_name` CCF annotation, raising an error otherwise (never triggered). Result: 69,453 of 272,227 units (25.5%), median 390 / mean 401.5 per session, range 90-923; one session (`sub-440958_ses-20190216T162508`) dropped.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```
```python
annotations = as_string_array(nwb.units["anno_name"])[good]
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: curated unit without CCF annotation")
```
```python
"unit_filter": "NWB units.classification == 'good' (released region-specific classifier QC)",
```

iii. CONVERSION_NOTES Step 3 neuron curation: "Use the released region-specific logistic-regression classifier result (`units.classification == 'good'`). The classifiers use 15 quality metrics and were trained on manual good/unlabelled labels by major region. The newer movement paper additionally excludes mean firing rates below 2 Hz for video-to-neural prediction, but this is analysis-specific and would discard valid classifier-QC neural inputs; it is not adopted for this general decoder conversion." Step 4 reconciles the count with the paper: archive gives 69,453 vs published 69,943, "(likely version/reporting transposition); the archive exactly reproduces 173 nonempty sessions and ~70k/25.5% of 272,227 clusters." Step 10 check 10: "Converted median neurons/session is 390 versus paper 393."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, taken from `BehavioralEvents/go_start_times`. Spike times and event timestamps share one session-absolute clock, so no resampling, interpolation, or per-stream offset correction is needed: the fixed relative edge grid is simply added to each trial's go-cue time to form that trial's absolute edges, and spikes are binned against those.

ii.
```python
go_times = go_times_all[:n_recorded_trials]
```
```python
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
...
edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```
```python
"temporal_alignment_event": "go cue onset",
"off_start": OFF_START,   # -2.5
"off_end": OFF_END,       # +1.5
```

iii. CONVERSION_NOTES Step 10 check 7: "Both use go onset as zero. Raw reference exports already store go-relative spikes; NWB stores absolute spikes and absolute go events, so subtracting/adding go produces the same alignment." Step 4 notes the AI deliberately does *not* clip the window to the NWB behavioural trial interval: "17,467 trials have go-2.5 earlier than NWB `trials.start_time`, especially replay/early trials ... Do not treat the behavioral interval boundary as missing neural/video data; use continuous acquisition/spikes and exact requested go window."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, i.e. 80 non-overlapping half-open bins spanning -2.5 s to +1.5 s about the go cue. A single relative edge vector (81 edges) and its 80 centres are computed once at module scope and reused for every trial and session, so every trial has exactly 80 timepoints. No rebinning occurs: spike *times* are binned directly at the target resolution rather than being re-aggregated from some other bin width. `metadata['time_bin_size'] = 50.0` (ms) and the bin centres are also stored in metadata.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
N_TIME = len(CENTERS_REL)
```
```python
"time_bin_size": 50.0,
"neural_measure": "firing rate (Hz), half-open nonoverlapping 50-ms spike bins",
"n_timepoints": N_TIME,
"time_bin_centers_seconds_from_go": CENTERS_REL.astype(np.float32),
```

iii. CONVERSION_NOTES Step 5 key decision 7: "bins are `[edge_i, edge_{i+1})`, centers are edge+25 ms, exactly 80 points; metadata bin size is 50 ms, offsets -2.5/+1.5 s." Step 1: "For this task, the required nonoverlapping 50-ms bins override the reference bin width/stride while preserving its half-open counting convention." Step 10 check 11 verified "exact 81 edges/80 bins, final edge +1.5 exclusion convention".

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the instruction-tone/sample-epoch onsets) plus `trials.start_time` and the go-cue times. For each trial the AI selects the **first** sample event at or after that trial's `start_time`, and validates that it falls at or before the trial's go cue. (The reference instead takes the **last** sample onset before the go cue.) Measured on the raw data, the two definitions differ on ~2.5-3% of trials — those where an early lick replayed the sample epoch and produced more than one tone.

ii.
```python
def first_event_per_trial(events, starts, upper_bounds, name) -> np.ndarray:
    """Select the first event in each [trial start, upper bound] interval."""
    indices = np.searchsorted(events, starts, side="left")
    if np.any(indices >= len(events)):
        raise ValueError(f"Missing {name} event after a trial start")
    selected = events[indices]
    bad = selected > upper_bounds + 1e-8
    if np.any(bad):
        raise ValueError(f"Missing {name} event in {int(bad.sum())} trial intervals")
    return selected
```
```python
trial_starts = np.asarray(trials_df.start_time, dtype=np.float64)
sample_events = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_onsets = first_event_per_trial(sample_events, trial_starts, go_times, "tone")
```

iii. CONVERSION_NOTES Step 5 mapping: "`BehavioralEvents/sample_start_times`; trial boundaries → `input[0]` | Select first tone/sample onset within each native trial". `metadata['tone_onset_definition'] = "first sample_start event within the native trial"`. Step 9 acknowledges the consequence of the "first" choice and accepts it: converted range is `[-1.5, 11.9]` s, and "Long values arise from early-lick replay; retained."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for every bin, the absolute bin-centre time minus that trial's tone onset, in seconds, stored as `float32` in row 0 of the input array. Equivalently `bin_centre_relative_to_go + (go - tone)`. No discretisation into a binary onset series is performed.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
...
tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
...
input_cube = np.stack([tone_time, photostim], axis=1)
```
```python
"input_names": ["time from tone onset (s)", "photostimulation on"],
```

iii. CONVERSION_NOTES Step 5 key decision 3: "Although generic format guidance suggests binary onset series, the Decoder Task explicitly requests continuous 'time from tone onset in seconds'; the explicit task controls." Step 5 mapping adds: "Continuous/time-varying as explicitly requested."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same grid as the neural bins. `centers_abs` is the go-cue-relative centre vector `CENTERS_REL` added to each trial's go-cue time — the same `go_times` and the same module-level grid used to build the spike edges — so bin *k* of the input covers the identical interval as bin *k* of the firing rates. Trials removed by the all-zero filter are dropped from `tone_onsets` and `centers_abs` at the same time as from `rates`, so indexing stays consistent.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
```
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
rates = bin_spikes(nwb.units, good, go_times)
```
```python
    go_times = go_times[neural_valid]
    tone_onsets = tone_onsets[neural_valid]
    centers_abs = centers_abs[neural_valid]
```

iii. Implicit in CONVERSION_NOTES Step 5 key decision 7 (one shared 80-point grid) and confirmed by Step 10 check 3: an independent PyNWB re-derivation of "the first sample event in the native trial and calculated center-relative seconds" passed `np.allclose` against the pickle. Step 7 processing plots showed "go-aligned firing rates, linear tone-relative time ... No temporal offset or bin-count anomaly remains."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `photostim_onset` and `photostim_duration` (stored as strings, with `'N/A'` on unstimulated trials), combined with `trials.start_time` to convert the trial-relative onset to the session-absolute clock. Bin centres (already absolute) are then tested against the interval. `BehavioralEvents` photostim event timestamps are not used.

ii.
```python
onset_raw = np.asarray(trials_df.photostim_onset).astype(str)
duration_raw = np.asarray(trials_df.photostim_duration).astype(str)
trial_start = np.asarray(trials_df.start_time, dtype=np.float64)
onset = np.full(len(trials_df), np.nan)
duration = np.full(len(trials_df), np.nan)
active = onset_raw != "N/A"
onset[active] = np.asarray(onset_raw[active], dtype=float) + trial_start[active]
duration[active] = np.asarray(duration_raw[active], dtype=float)
```

iii. CONVERSION_NOTES Step 2 records that `nwb.trials` contains "photostimulation onset/power/duration"; Step 5 mapping: "trial photostimulation onset/duration → `input[1]`". Step 3 notes the experimental design the values should match: "Photoinhibition occupies the final 0.5 s of delay and ends at go cue (including ramp-down)" and "~25% randomly interleaved trials in 17 mice".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series rather than a per-trial flag: 1 where a bin centre lies in the half-open interval `[onset, onset + duration)`, 0 elsewhere, cast to `float32` and placed in row 1 of the input array. Unstimulated trials are masked out explicitly with the `active` flag (so the NaN bounds can never produce a 1). Verified range `[0.0, 1.0]`.

ii.
```python
return (
    active[:, None]
    & (centers_abs >= onset[:, None])
    & (centers_abs < onset[:, None] + duration[:, None])
).astype(np.float32)
```
```python
photostim = trial_photostimulation(trials_df, centers_abs)
input_cube = np.stack([tone_time, photostim], axis=1)
```

iii. CONVERSION_NOTES Step 5 mapping: "Binary 1 when a 50-ms bin center lies in `[stim_on, stim_off)`, otherwise 0 ... Preserves stimulated trials rather than filtering." Step 4 records the deliberate decision not to drop photostimulated trials (the papers' `get_regular_trial_mask` does) because photostimulation is a required decoder input. Step 7 plot review confirms "a late-delay photostimulation interval ending before go where applicable".

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. By converting the trial-relative onset into absolute session time (`start_time + photostim_onset`) and comparing it against `centers_abs`, the very same go-cue-relative bin-centre array used for the neural binning and the tone input. Half-open comparison matches the half-open spike bins.

ii.
```python
onset[active] = np.asarray(onset_raw[active], dtype=float) + trial_start[active]
```
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
...
photostim = trial_photostimulation(trials_df, centers_abs)
```

iii. CONVERSION_NOTES Step 5 mapping note: "reference aligns stimulation by subtracting go time". Step 10 check 3 independently re-derived "the trial-relative photostimulation interval" from raw PyNWB and confirmed it with `np.allclose`.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table columns: `trial_instruction` (`'left'` / `'right'`) and `outcome` (`'hit'` / `'miss'` / `'ignore'`). Hit ⇒ the animal licked the instructed side; miss ⇒ it licked the other side; ignore ⇒ no lick. The AI additionally cross-validates the derived label against the first left/right lick event in the response window using `BehavioralEvents/left_lick_times` and `right_lick_times`, but only as a diagnostic — it never overrides the derived label.

ii.
```python
def derive_choice(instruction, outcome) -> np.ndarray:
    """Map task result to actual lick choice: left=0, right=1, no lick=2."""
    instructed = np.where(instruction == "left", 0, 1)
    choice = instructed.copy()
    choice[outcome == "miss"] = 1 - choice[outcome == "miss"]
    choice[outcome == "ignore"] = 2
    return choice.astype(np.uint8)
```
```python
def event_choice_check(nwb, trials_df, go_times, choice) -> dict:
    """Compare derived choice against the first left/right response lick."""
    events = nwb.acquisition["BehavioralEvents"].time_series
    left = np.asarray(events["left_lick_times"].timestamps[:], dtype=float)
    right = np.asarray(events["right_lick_times"].timestamps[:], dtype=float)
    ...
```

iii. CONVERSION_NOTES Step 5 key decision 2: "Outcome plus instruction is deterministic under this two-alternative task and handles no-response cleanly; spot-check against lick-event direction." Step 9 reports the spot-check result: "Choice inferred from instruction/outcome agrees with first response-epoch lick events on 90,449/90,734 trials (99.69%)". Step 10: "Instruction/outcome is the authoritative trial result; 285 complex/missing event sequences are not used to overwrite native outcome-derived choice."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, stored as `uint8` in row 0 of the `(4, 80)` per-trial output array and broadcast across all 80 bins (choice is a per-trial quantity but the format is kept rectangular and time-varying). `output_values[0] = ['left', 'right', 'no lick']`. Observed distribution: left 0.428, right 0.422, no lick 0.149.

ii.
```python
choice = derive_choice(instruction, outcome_text)
...
output_cube = np.empty((n_trials, 4, N_TIME), dtype=np.uint8)
output_cube[:, 0, :] = choice[:, None]
```
```python
"output_names": ["lick direction choice", "outcome", "early lick", "tongue y-position"],
"output_values": [
    ["left", "right", "no lick"],
    ...
]
```

iii. CONVERSION_NOTES Step 5 mapping: "hit→instructed side; miss→opposite side; ignore→no lick; broadcast across 80 bins ... Values: left=0, right=1, no lick=2". Key decision 4: "Store all outputs as `(4,80)`; repeat per-trial labels over time so tongue y can remain time-varying and dimensions stay rectangular."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'` requested by the instructions. No derivation.

ii.
```python
outcome_text = np.asarray(trials_df.outcome).astype(str)
```

iii. CONVERSION_NOTES Step 2 lists `outcome` as a native trials-table field, and Step 5 mapping: "`trials.outcome` → `output[1]` outcome | Map ignore=0, miss=1, hit=2 | reference `correctness` | Required order." Step 10 check 4 verified three concrete native trials map to the right (choice, outcome) pairs: "(right,hit), (right,miss), and (no lick,ignore)."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore→0`, `miss→1`, `hit→2` (the order given in the instructions); the result is `uint8` in row 1 of the output array, broadcast across all 80 bins. Observed distribution: ignore 0.149, miss 0.166, hit 0.684.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.uint8)
...
output_cube[:, 1, :] = outcome[:, None]
```
```python
["ignore", "miss", "hit"],
```

iii. CONVERSION_NOTES Step 9 explains why the hit rate is below the paper's figure: "Outcome distribution | selected control correct 84% | ... | [ignore .149, miss .166, hit .684] | Lower than selected 84% as required early/stim/all sessions retained" — i.e. the difference is fully explained by the deliberate decision not to apply the paper's trial exclusions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the trials-table `early_lick` column, which holds `'no early'` / `'early'`.

ii.
```python
early_text = np.asarray(trials_df.early_lick).astype(str)
```

iii. CONVERSION_NOTES Step 2 lists `early_lick` as a native field; Step 5 mapping: "`trials.early_lick` → `output[2]` early lick | Map no early=0, early=1 | reference `early_lick_trials` | Retained as requested." Step 3 notes the behavioural meaning that makes the label recoverable from the -2.5 s window: "Early licking triggers replay of the sample/delay epoch."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean test `early_text == "early"` cast to `uint8` (0 = no, 1 = yes), written into row 2 and broadcast across all 80 bins. Observed distribution: no 0.884, yes 0.116. Note this is a permissive mapping — any string other than the exact `'early'` becomes 0 — rather than an explicit two-key dictionary lookup that would fail loudly on an unexpected value.

ii.
```python
early = (early_text == "early").astype(np.uint8)
...
output_cube[:, 2, :] = early[:, None]
```
```python
["no", "yes"],
```

iii. CONVERSION_NOTES Step 4/5 justify retaining early-lick trials at all: the papers' `get_regular_trial_mask` "removes early, ignore, stimulation, free/auto water" but "Retain them because they are requested decoder inputs/outputs; this is a necessary task-specific difference." Step 12 additionally re-checked the label because early lick was the output closest to the 1.5×-chance threshold: "native labels were checked on three concrete trials, class prevalence is 11.6% yes (not degenerate)".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` and `timestamps` is the matching ~294 Hz frame clock. Column 1 is taken as tongue y and column 2 as the DeepLabCut tracking likelihood. The column indices are hard-coded rather than read from the series' `description` attribute.

ii.
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series[
    "Camera0_side_TongueTracking"
]
timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
tracking = np.asarray(ts.data[:], dtype=np.float64)
y_all = tracking[:, 1]
likelihood_all = tracking[:, 2]
```

iii. CONVERSION_NOTES Step 2: "`BehavioralTimeSeries` contains side-camera jaw, nose, and tongue x/y/likelihood with explicit timestamps. Tongue tracking exists in all 174 sessions." Step 3 notes the reference papers "use side view" for markers, and Step 1 confirms "Reference marker processing includes `tongue_y`, uses side-camera trial numbers".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Four steps:
1. **Visibility mask over the whole session**: a frame counts as visible only if y and likelihood are finite and `likelihood >= 0.9`.
2. **Session thresholds**: the 40th and 60th percentiles of `y` over *all visible raw frames* in that session (not over binned means, and not including invisible frames).
3. **Per-bin sampling**: for each bin centre, take the last camera frame at or before that centre (`searchsorted(..., 'right') - 1`); the bin's y and likelihood are that frame's. No averaging within the bin.
4. **Discretisation**: visible frames are cut at the two thresholds; anything not visible becomes class 3.

The AI raises an error if a session has fewer than two visible frames (never triggered). Resulting distribution: 0.062 / 0.032 / 0.065 / 0.841 — i.e. among the 15.9% visible bins the split is ≈39/20/41%, close to the intended 40/20/40.

ii.
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.9
```
```python
visible_all = (
    np.isfinite(y_all)
    & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
if visible_all.sum() < 2:
    raise ValueError("Fewer than two visible tongue samples in session")
q40, q60 = np.percentile(y_all[visible_all], [40, 60])

frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
valid_idx = (frame_idx >= 0) & (frame_idx < len(timestamps))
safe_idx = np.clip(frame_idx, 0, len(timestamps) - 1)
y = y_all[safe_idx]
likelihood = likelihood_all[safe_idx]
visible = (
    valid_idx
    & np.isfinite(y)
    & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
)
```

iii. CONVERSION_NOTES Step 4/5 explain the departure from the papers' marker handling: "Marker outliers were defined by five-sigma frame velocity and imputed from nearby frames; when tongue was occluded/in the mouth, position was set to its mean. Here the explicit fourth tongue class ('not visible') requires preserving missing/low-confidence visibility instead of mean imputation." Key decision 6: "The reference's continuous-regression imputation would erase the requested not-visible class. Use native likelihood for visibility and native y only when visible." On the threshold: "Likelihood is bimodal, making 0.9 robust; percentiles exclude invisible frames." On the sampling rule: "At each 50-ms bin center use the last tracking frame at/before center (reference convention)", referring to `align_markers_between_lims`, which selects "the last sample in each interval".

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, using the 40th/60th percentiles of visible raw y computed above: `y < q40 → 0`, `q40 <= y <= q60 → 1`, `y > q60 → 2`, and any bin whose sampled frame is missing or below the likelihood cutoff → `3` ("not visible"). The array is initialised to 3, so class 3 is the default for everything not positively classified. The per-session thresholds are recorded in `session_info['tongue_percentiles_y']` and plotted by `--show-processing`.

ii.
```python
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
category[visible & (y < q40)] = 0
category[visible & (y >= q40) & (y <= q60)] = 1
category[visible & (y > q60)] = 2
diagnostics = {
    "visible_native_fraction": float(visible_all.mean()),
    "visible_aligned_fraction": float(visible.mean()),
    "class_counts": np.bincount(category.ravel(), minlength=4).tolist(),
}
return category, (float(q40), float(q60)), diagnostics
```
```python
["below 40th percentile", "40th to 60th percentile", "above 60th percentile", "not visible"],
```

iii. This follows the instructions' discretisation literally (per-session percentiles, 0/1/2/3 including "not visible"). CONVERSION_NOTES Step 5 mapping: "Compute session q40/q60 on all native visible y values; visible y <q40→0, q40≤y≤q60→1, y>q60→2." Step 10 check 11 lists the boundary/edge cases checked, including "low-confidence/NaN tongue samples". Step 7: "Confirm per-session tongue thresholds and class occupancy; visually overlay raw y/likelihood, aligned samples, thresholds, and categorical output."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Via the shared `centers_abs` array — the same go-cue-relative bin centres used for the firing rates — so bin *k* of the tongue output corresponds to bin *k* of the neural data. The camera timestamps are on the same session clock, so alignment is a plain `searchsorted` lookup; no interpolation or offset correction. The rule is "last frame at or before the bin centre", with the only guard being array bounds: there is **no check that the selected frame actually falls inside the bin**. Because the side video is trial-gated (one ~1 s gap per inter-trial interval), a bin centre landing in a gap silently inherits the last frame of the previous trial. Measured on the raw data this affects ~0.17% of bins, and only ~0.003% of bins are both stale and labelled visible — so the effect is negligible in practice, but the guard is absent by construction.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
...
tongue, tongue_thresholds, tongue_diag = tongue_categories(nwb, centers_abs)
...
output_cube[:, 3, :] = tongue
```
```python
frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
valid_idx = (frame_idx >= 0) & (frame_idx < len(timestamps))
safe_idx = np.clip(frame_idx, 0, len(timestamps) - 1)
```
```python
"tongue_alignment": "last video frame at or before each neural-bin center",
```

iii. CONVERSION_NOTES Step 5 mapping cites the reference convention: "At each 50-ms bin center use the last tracking frame at/before center (reference convention)" — i.e. `align_markers_between_lims`, which takes the last marker sample in each alignment interval. Step 4 justifies not clipping to the behavioural trial interval: "Do not treat the behavioral interval boundary as missing neural/video data; use continuous acquisition/spikes and exact requested go window." Step 10 check 4: an independent PyNWB re-derivation of "timestamp-aligned tongue y/likelihood with session percentiles" passed `np.allclose`.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five distinct cases, handled by three different strategies — exclude, categorise, or abort:

- **Session never quality-controlled** (`classification` / `anno_name` are NaN): `as_string_array` turns NaN into the string `'nan'`, so no unit matches `'good'`, and the session is skipped with a recorded `skip_reason` (1 session).
- **Behavioural rows beyond the ephys recording**: the trials table and go-cue array are truncated to the per-unit `is_good_trials` length (1,060 rows across 7 sessions).
- **Trials with no spikes from any curated unit** (free-water trials, unrecorded windows): detected empirically as population-all-zero and excluded (2,576 trials); the count is recorded per session.
- **Frames with no tracked tongue** (low likelihood or non-finite): assigned the explicit `'not visible'` category rather than imputed.
- **Structural anomalies**: inconsistent `is_good_trials` lengths across units, unit trial count exceeding the behavioural table, too few go cues, a missing tone event in a trial interval, a curated unit with no CCF annotation, a session with < 2 visible tongue frames, and a session left with < 2 trials all raise `ValueError` — i.e. they abort the whole conversion rather than skipping the session. None of these fired on this dataset, but the `< 2 trials` case in particular is a format requirement the reference handles by dropping the session.

ii.
```python
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```
```python
neural_valid = np.any(rates != 0, axis=(1, 2))
excluded_all_zero = int((~neural_valid).sum())
if excluded_all_zero:
    rates = rates[neural_valid]
    ...
n_trials = len(trials_df)
if n_trials < 2:
    raise ValueError(f"{path.name}: fewer than two trials with neural data")
```
```python
visible = (
    valid_idx
    & np.isfinite(y)
    & np.isfinite(likelihood)
    & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
)
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
```
```python
if np.any(annotations == ""):
    raise ValueError(f"{path.name}: curated unit without CCF annotation")
```

iii. CONVERSION_NOTES Step 7: "Initial validation revealed hundreds of trailing all-zero trials in session 2 ... spikes stopped with the recording. The converter now uses the common per-unit validity-vector length as the source ephys trial count, and removes rare population-all-zero windows." Step 10 "Issues Found and Resolved" lists the trailing-row and all-zero fixes and states the result: "This yields 90,734 usable trials and removes all validator warnings." Key decision 6 justifies the categorical treatment of invisible tongue rather than imputation. Step 10 check 11 lists the edge cases explicitly checked: "sessions with behavioral rows after ephys ends, the zero-good session, all-zero neural windows, empty anatomy, low-confidence/NaN tongue samples, and minimum trial counts."

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion took 178.7 s for 174 files (0.35-1.5 s per session, scaling with unit count), plus pickling of the 11.84 GB output. The dominant costs are (1) NWB/HDF5 I/O — one ragged `spike_times` read per good unit (up to 923 per session), the `(n_frames, 3)` tongue array (~0.7-1 M rows), and one ragged `is_good_trials` read per good unit; (2) the per-unit `np.searchsorted` over the `(n_trials, 81)` edge matrix and materialising the `(n_trials, n_neurons, 80)` `float32` rate cube; (3) serialisation of the 11.84 GB pickle. Timing is printed per session and in total.

ii.
```python
started = time.perf_counter()
...
session["info"]["processing_seconds"] = time.perf_counter() - started
```
```python
print(
    f"Finished in {time.perf_counter() - total_started:.2f} s; "
    f"pickle size {output_path.stat().st_size / 1e9:.3f} GB",
    flush=True,
)
```
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
```

iii. CONVERSION_NOTES Step 6: "The dominant unavoidable cost is materializing neuron × trial × 80 firing rates. Per-unit Python spike counting and repeated event scans would be unnecessarily slow." Step 7 estimated "0.48 s/session (sample mean) → ~83 s for 173 sessions" and Step 9 reports "Full conversion completed in 178.7 s, below the estimate and optimization threshold", so no further optimisation was pursued against the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five Python-level loops remain:
- **`bin_spikes` per-unit loop** — the trial dimension is already vectorised (all `(n_trials, 81)` edges in one `searchsorted`), so only the per-unit iteration remains. This is inherent to the ragged spike storage and cannot be collapsed into a single `searchsorted`; it could however be made cheaper by reading the whole `spike_times` buffer once and slicing it, instead of one HDF5 read per unit.
- **`recorded_lengths` comprehension** — one ragged HDF5 read per good unit purely to obtain a length that is asserted identical for all of them.
- **`event_choice_check` per-trial loop** — a pure-Python loop over every trial doing two `searchsorted` calls each; fully vectorisable (and only used for a diagnostic).
- **`region_idx` per-unit loop** — building the region index one label at a time; replaceable with `np.unique(..., return_inverse=True)` plus a dict merge.
- **The three list comprehensions** slicing the session cubes into per-trial arrays; these are required by the target format and are cheap.

ii.
```python
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
```
```python
for i, go in enumerate(go_times):
    stop = min(float(trials_df.stop_time.iloc[i]), go + 1.5)
    li = np.searchsorted(left, go, side="left")
    ri = np.searchsorted(right, go, side="left")
```
```python
for i, label in enumerate(annotations):
    if label not in region_to_idx:
        region_to_idx[label] = len(region_to_idx)
    region_idx[i] = region_to_idx[label]
```

iii. CONVERSION_NOTES Step 6 documents only the spike-binning vectorisation: "For each unit, `searchsorted` evaluates all trial-bin edges at once; behavioral alignment is vectorized; one NWB is opened once; session cubes are computed once and exposed as trial matrices." The remaining loops (`recorded_lengths`, `event_choice_check`, region mapping) are not discussed in the notes; the AI's justification for stopping there is that the measured runtime was already "below the estimate and optimization threshold" (Step 9).

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened exactly once, and every derived quantity (rates, inputs, outputs, tongue thresholds) is computed once per session, so there is no second pass over the dataset. Two genuine redundancies remain inside a session:
- `len(nwb.units["is_good_trials"][int(i)])` is read for **every** good unit (up to 923 ragged HDF5 reads per session) to obtain a single number that the code then asserts is identical across units. Two reads would suffice for the consistency check.
- `event_choice_check` re-reads the `BehavioralEvents` container and re-derives per-trial choice from the lick streams, duplicating work already done by `derive_choice` from the trials table.

Additionally, `nwb.trials[:]` is materialised in full before being truncated, and `centers_abs` / `rates` are recomputed nowhere but are re-sliced by the `neural_valid` mask rather than being built only for valid trials.

ii.
```python
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
if np.any(recorded_lengths != recorded_lengths[0]):
    raise ValueError(f"{path.name}: inconsistent recorded-trial counts across units")
```
```python
choice_check = event_choice_check(nwb, trials_df, go_times, choice)
```

iii. The AI's stated position (CONVERSION_NOTES Step 6) is that repetition was avoided: "one NWB is opened once; session cubes are computed once and exposed as trial matrices", and that "repeated event scans would be unnecessarily slow". The `recorded_lengths` and `event_choice_check` redundancies are not acknowledged; they are justified implicitly by the consistency checks they support (Step 5 key decision 5 requires proving the validity vectors agree across units; Step 5 key decision 2 requires the independent choice spot-check).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations run on every session in the production path but never reach the decoder:
- **`event_choice_check`** — a full per-trial pass over the left/right lick event streams. Its output is only a three-integer diagnostic in `session_info`; the decoder ignores it. This is validation code left inside the conversion loop rather than in a separate script.
- **Tongue diagnostics** — `visible_native_fraction`, `visible_aligned_fraction` and a `np.bincount` over the whole category array, per session, stored only in metadata.
- **`tongue_x` (column 0)** of the tracking array is read from disk as part of the `(n_frames, 3)` slab and never used.
- **`duration_raw` / `onset_raw` string casts** are applied to the full column before masking to the active trials.
- **Metadata payloads** such as `time_bin_centers_seconds_from_go`, `n_behavioral_table_trials`, `n_recorded_trial_entries`, `source_file`, and `processing_seconds` are stored for every session and unused downstream.
- **`--show-processing` plotting** (opt-in only, so not part of the default run).

None of these is expensive relative to the 178.7 s total, and all are defensible as provenance/QC records; but they are strictly discarded by `train_decoder.py`.

ii.
```python
choice_check = event_choice_check(nwb, trials_df, go_times, choice)
session = {
    ...
    "info": {
        ...
        "tongue_diagnostics": tongue_diag,
        "choice_event_check": choice_check,
    },
}
```
```python
diagnostics = {
    "visible_native_fraction": float(visible_all.mean()),
    "visible_aligned_fraction": float(visible.mean()),
    "class_counts": np.bincount(category.ravel(), minlength=4).tolist(),
}
```
```python
"time_bin_centers_seconds_from_go": CENTERS_REL.astype(np.float32),
```

iii. The AI does not frame these as waste; it frames them as required verification. CONVERSION_NOTES Step 5 key decision 2 mandates the lick-event cross-check ("spot-check against lick-event direction"), and Step 9/10 report its result (99.69% agreement) as evidence that the choice derivation is right. Step 7 similarly uses the tongue class counts per session ("tongue=[...]" is printed for every session in `conversion_full_out.txt`) to confirm "per-session tongue thresholds and class occupancy". The overall justification is that the runtime budget was comfortably met, so retaining the diagnostics inline cost nothing.
