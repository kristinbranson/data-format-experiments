# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files matching `sub-*/*.nwb` under the data directory, sorts them, and processes each file sequentially with `pynwb.NWBHDF5IO`. Within each file, it reads `nwb.units`, `nwb.trials`, `nwb.acquisition['BehavioralEvents']`, and `nwb.acquisition['BehavioralTimeSeries']`. It finds 174 NWB files across 28 subjects.

ii.
```python
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
    ...
    events = nwb.acquisition["BehavioralEvents"].time_series
```

iii. The AI noted in CONVERSION_NOTES.md that the dataset is DANDI:000363 with 174 NWB files, one per session, and that `pynwb` is the mandated API. It verified the count against `dandiset.yaml`.

## 1-b. How are the data split into subjects?

i. Each NWB file's `nwb.subject.subject_id` (a numeric string like `'440956'`) identifies the subject. Subjects are collected in encounter order and indexed per session.

ii.
```python
subject = session["subject"]
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

```python
"subject": str(nwb.subject.subject_id),
```

iii. The AI documented 28 subjects in CONVERSION_NOTES.md, consistent with both papers.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The AI identifies each session by `nwb.identifier`. Sessions are processed in sorted file order. 173 sessions are retained after excluding one with zero curated units.

ii.
```python
files = nwb_files()  # sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
"session_id": str(nwb.identifier),
```

iii. The AI documented that the file boundary is the session boundary and that one session (`sub-440958_ses-20190216T162508`) is dropped for having no good units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials[:]`). However, the AI truncates the behavioral trials table to the number of recorded trial entries as determined by the length of `is_good_trials` vectors per curated unit. It then further removes trials where all curated neurons fire zero spikes across the entire trial window.

ii.
```python
trials_df_all = nwb.trials[:]
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
...
n_recorded_trials = int(recorded_lengths[0])
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
```

iii. The AI found that 8 NWBs contain more behavioral-table rows than the `is_good_trials` validity vector length, and used this to truncate. It documented this in Step 7 and Step 10 of CONVERSION_NOTES.md.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two filters: (1) truncation of the trials table to the `is_good_trials` vector length (removing trailing behavioral rows beyond the recording), and (2) removal of trials where all curated neurons have zero spikes across the entire window. The AI does NOT explicitly filter `free_water` trials and does NOT use `obs_intervals`. This yields 90,734 trials from 173 sessions.

ii.
```python
n_recorded_trials = int(recorded_lengths[0])
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
...
neural_valid = np.any(rates != 0, axis=(1, 2))
excluded_all_zero = int((~neural_valid).sum())
if excluded_all_zero:
    rates = rates[neural_valid]
    trials_df = trials_df.iloc[np.flatnonzero(neural_valid)].copy()
```

iii. The AI justified this by noting that trailing behavioral rows have no spikes and that population-all-zero windows indicate absent neural data. It reported 2,576 all-zero entries excluded across 94 sessions. However, it does not mention `free_water` filtering or `obs_intervals` usage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for each unit classified as `'good'`, and `BehavioralEvents/go_start_times` for alignment.

ii.
```python
spike_column = units["spike_times"]
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
```

iii. The AI noted that spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.searchsorted` on absolute bin edges, then divided by bin width (0.05s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
rates = np.empty((len(go_times), len(good_indices), N_TIME), dtype=np.float32)
...
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```

iii. The AI documented that the half-open bin convention matches the reference code's `sliding_histogram` function and that the 50ms width is mandated by the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. No additional metric thresholds are applied. Sessions with zero good units are skipped. This retains 69,453 units across 173 sessions.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```

iii. The AI justified this by referencing the spike sorting QC paper's classifier labels and documented that the 2-Hz firing rate cutoff from the movement paper was analysis-specific and not adopted.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as offsets from each trial's go-cue time. Spikes and events share the same absolute clock, so no additional alignment is needed.

ii.
```python
go_times_all = np.asarray(events["go_start_times"].timestamps[:], dtype=np.float64)
...
absolute_edges = go_times[:, None] + EDGES_REL[None, :]
```

iii. The AI documented that all NWB times are on one global clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins spanning -2.5s to +1.5s relative to the go cue. Bin edges are defined once and reused for all trials and sessions.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START = -2.5
OFF_END = 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_SIZE_S / 2, BIN_SIZE_S)
CENTERS_REL = EDGES_REL[:-1] + BIN_SIZE_S / 2
N_TIME = len(CENTERS_REL)
```

iii. The AI documented that the 50ms bin width and -2.5/+1.5s window are mandated by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` timestamps and `trials.start_time`. The AI selects the **first** sample_start event within each native trial interval (between trial start and go cue).

ii.
```python
def first_event_per_trial(
    events: np.ndarray, starts: np.ndarray, upper_bounds: np.ndarray, name: str
) -> np.ndarray:
    """Select the first event in each [trial start, upper bound] interval."""
    indices = np.searchsorted(events, starts, side="left")
    ...
    return selected

trial_starts = np.asarray(trials_df.start_time, dtype=np.float64)
sample_events = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
tone_onsets = first_event_per_trial(sample_events, trial_starts, go_times, "tone")
```

iii. The AI's CONVERSION_NOTES.md Step 5 states: "Select first tone/sample onset within each native trial." The AI chose the first tone event after trial start, while the reference takes the last tone before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center (in absolute time), subtract the tone onset time to get seconds since tone onset.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
tone_time = (centers_abs - tone_onsets[:, None]).astype(np.float32)
```

iii. The AI documented this as "continuous time-varying" per the decoder task specification.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers (go_times + CENTERS_REL), so tone time and neural data share the same temporal grid.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `trials.photostim_onset` (onset relative to trial start), `trials.photostim_duration`, and `trials.start_time`.

ii.
```python
onset_raw = np.asarray(trials_df.photostim_onset).astype(str)
duration_raw = np.asarray(trials_df.photostim_duration).astype(str)
trial_start = np.asarray(trials_df.start_time, dtype=np.float64)
```

iii. The AI documented these as the source fields in its mapping plan.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The onset is converted from a string to float and added to trial start to get absolute onset time. A binary time series is created where bin center falls within [onset, onset + duration).

ii.
```python
active = onset_raw != "N/A"
onset[active] = np.asarray(onset_raw[active], dtype=float) + trial_start[active]
duration[active] = np.asarray(duration_raw[active], dtype=float)
return (
    active[:, None]
    & (centers_abs >= onset[:, None])
    & (centers_abs < onset[:, None] + duration[:, None])
).astype(np.float32)
```

iii. The AI notes that `'N/A'` marks non-stimulated trials and the output is a binary time series.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Bin centers are in absolute time (go_times + CENTERS_REL), same grid as neural data. The photostim onset is also in absolute time, so comparisons are direct.

ii.
```python
centers_abs = go_times[:, None] + CENTERS_REL[None, :]
```

iii. Same coordinate system as neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trials.trial_instruction` ('left'/'right') and `trials.outcome` ('hit'/'miss'/'ignore').

ii.
```python
def derive_choice(instruction: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    instructed = np.where(instruction == "left", 0, 1)
    choice = instructed.copy()
    choice[outcome == "miss"] = 1 - choice[outcome == "miss"]
    choice[outcome == "ignore"] = 2
    return choice.astype(np.uint8)
```

iii. The AI verified this against first response-epoch lick events and found 99.69% agreement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Hit maps to instructed side, miss maps to opposite side, ignore maps to no lick (2). Coded as left=0, right=1, no lick=2. Broadcast across all 80 time bins.

ii.
```python
output_cube[:, 0, :] = choice[:, None]
```

iii. The AI included a cross-check function `event_choice_check` that compared derived choice against actual lick events.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials.outcome` column ('ignore', 'miss', 'hit').

ii.
```python
outcome_text = np.asarray(trials_df.outcome).astype(str)
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.uint8)
```

iii. Straightforward mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Map ignore=0, miss=1, hit=2. Broadcast across all 80 bins.

ii.
```python
output_cube[:, 1, :] = outcome[:, None]
```

iii. Matches the instruction ordering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials.early_lick` column ('no early', 'early').

ii.
```python
early_text = np.asarray(trials_df.early_lick).astype(str)
early = (early_text == "early").astype(np.uint8)
```

iii. Direct mapping from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Map 'early' to 1 and 'no early' to 0. Broadcast across all 80 bins.

ii.
```python
early = (early_text == "early").astype(np.uint8)
output_cube[:, 2, :] = early[:, None]
```

iii. Straightforward binary encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains (n_frames, 3) data = x, y, likelihood with timestamps.

ii.
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
tracking = np.asarray(ts.data[:], dtype=np.float64)
y_all = tracking[:, 1]
likelihood_all = tracking[:, 2]
```

iii. The AI documented this as the only tongue measurement in the file, running at ~300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a "last frame at or before each bin center" alignment approach. For each neural bin center, it finds the last video frame at or before that time using `searchsorted(..., side='right') - 1`. Frames with likelihood < 0.9 are considered not visible. Session-wide 40th and 60th percentiles are computed over all raw visible y values (not bin means). Visible aligned values are categorized against these thresholds; non-visible frames get class 3.

ii.
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.9

visible_all = (
    np.isfinite(y_all) & np.isfinite(likelihood_all)
    & (likelihood_all >= TONGUE_LIKELIHOOD_CUTOFF)
)
q40, q60 = np.percentile(y_all[visible_all], [40, 60])

frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
...
y = y_all[safe_idx]
likelihood = likelihood_all[safe_idx]
visible = (valid_idx & np.isfinite(y) & np.isfinite(likelihood)
           & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF))
category = np.full(centers_abs.shape, 3, dtype=np.uint8)
category[visible & (y < q40)] = 0
category[visible & (y >= q40) & (y <= q60)] = 1
category[visible & (y > q60)] = 2
```

iii. The AI justified the 0.9 threshold by noting the bimodal distribution of likelihood values. It chose last-frame alignment citing the reference code's `align_markers_between_lims` convention. It uses raw frame percentiles rather than bin-mean percentiles.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of all visible raw y values define two thresholds. Visible values below the 40th percentile get class 0, between 40th and 60th (inclusive) get class 1, above 60th get class 2, and non-visible frames get class 3.

ii.
```python
q40, q60 = np.percentile(y_all[visible_all], [40, 60])
category[visible & (y < q40)] = 0
category[visible & (y >= q40) & (y <= q60)] = 1
category[visible & (y > q60)] = 2
```

iii. The AI's boundary condition for class 1 includes both endpoints (`>=` q40 and `<=` q60), while the reference uses `np.digitize` which gives half-open intervals.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural bin center (absolute time), the AI finds the last video frame at or before that time via `searchsorted(timestamps, centers_abs, side='right') - 1`. This is a "sample-and-hold" approach rather than the reference's bin-averaging approach.

ii.
```python
frame_idx = np.searchsorted(timestamps, centers_abs, side="right") - 1
valid_idx = (frame_idx >= 0) & (frame_idx < len(timestamps))
safe_idx = np.clip(frame_idx, 0, len(timestamps) - 1)
y = y_all[safe_idx]
likelihood = likelihood_all[safe_idx]
```

iii. The AI cites the reference code's `align_markers_between_lims` as using a "last sample in each interval" convention.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled:
- **Zero-curated-unit session**: `classification` strings are checked; sessions with no `'good'` units are skipped.
- **Trailing behavioral rows beyond recording**: The `is_good_trials` vector length per unit determines the number of recorded trials, and the behavioral table is truncated accordingly. Population-all-zero trials are additionally removed.
- **Low-confidence tongue tracking**: Frames with likelihood < 0.9 are treated as not visible and get class 3.

ii.
```python
classifications = as_string_array(nwb.units["classification"])
good = np.flatnonzero(classifications == "good")
if len(good) == 0:
    return None, {"file": path.name, "skip_reason": "zero curated units"}
```

```python
recorded_lengths = np.asarray(
    [len(nwb.units["is_good_trials"][int(i)]) for i in good], dtype=int
)
n_recorded_trials = int(recorded_lengths[0])
trials_df = trials_df_all.iloc[:n_recorded_trials].copy()
```

```python
neural_valid = np.any(rates != 0, axis=(1, 2))
```

iii. The AI documented these as issues found during sample validation and critical review.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reported full conversion completed in 178.7 seconds for 173 sessions (~1.0s/session). The dominant costs are reading each NWB file and materializing the spike times array, followed by per-unit searchsorted operations and tongue tracking data loading.

ii. N/A

iii. The AI estimated processing time during sample validation and found it well within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop iterates over each good unit, running `searchsorted` for all trials at once. This cannot be fully vectorized because each unit has a different number of spikes (ragged arrays). No per-trial loop exists for neural data.

ii.
```python
for out_idx, unit_idx in enumerate(good_indices):
    spikes = np.asarray(spike_column[int(unit_idx)], dtype=np.float64)
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_idx, :] = np.diff(edge_positions, axis=1) / BIN_SIZE_S
```

iii. The AI noted that vectorized spike edge searches handle all trial-bin edges at once per unit.

## 10-c. What processing does the code repeat multiple times?

i. Nothing significant is repeated. Each NWB file is opened once. Bin edges are computed once per session. Session percentiles for tongue are computed once per session.

ii. N/A

iii. The AI documented this as a single-pass design.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `event_choice_check` (comparing derived choice against lick events) for every session as a diagnostic. This is not used in the output data but is stored in session info metadata. Additionally, extensive diagnostic metadata (tongue diagnostics, processing times, etc.) is computed and stored.

ii.
```python
choice_check = event_choice_check(nwb, trials_df, go_times, choice)
```

iii. These are validation checks that add modest overhead but don't contribute to the decoder data.
