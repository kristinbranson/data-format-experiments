# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored in `/app/data/sub-*/`. It uses `h5py` (not `pynwb`) to open each NWB file. All 174 NWB files are discovered via `sorted(DATA_DIR.glob("sub-*/*.nwb"))`. Each file is processed by `process_session()`, which reads trials from `intervals/trials`, neural data from `units/`, behavioral events from `acquisition/BehavioralEvents/`, and tongue tracking from `acquisition/BehavioralTimeSeries/`.

ii.
```python
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
with h5py.File(session_path, "r") as nwb:
    trials = nwb["intervals/trials"]
    trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
    ...
    go_start = np.asarray(
        nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:],
        dtype=np.float64,
    )
```

iii. The AI chose `h5py` over `pynwb` for direct HDF5 access. CONVERSION_NOTES.md documents the dataset structure exploration in Step 2 and confirms 174 NWB files across 28 subjects. The AI's trajectory shows it explored the NWB file structure using h5py to understand available groups and datasets.

## 1-b. How are the data split into subjects?

i. The AI uses the parent directory name (e.g., `sub-440956`) as the subject identifier, rather than reading the `subject_id` field from within the NWB file. Unique subjects are collected in encounter order and indexed.

ii.
```python
subject_id = session_path.parent.name
```

```python
subjects_order = list(OrderedDict((sess["subject_id"], None) for sess in processed_sessions).keys())
subject_to_idx = {subject: i for i, subject in enumerate(subjects_order)}
```

iii. CONVERSION_NOTES.md documents 28 subjects matching the papers. The AI uses folder names like `sub-440956` rather than the numeric `subject_id` field inside the NWB file (`440956`), which yields subject identifiers prefixed with `sub-`.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. Each file path stem is used as the session identifier. Sessions are sorted alphabetically by path.

ii.
```python
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = session_path.stem
```

iii. CONVERSION_NOTES.md Step 2 documents 174 NWB files. The AI identifies that one session has all-NaN classification and is dropped, yielding 173 sessions consistent with the papers.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in the NWB file. The AI reads `start_time`, `stop_time`, and other trial columns. Go-cue times come from `acquisition/BehavioralEvents/go_start_times/timestamps`. An assertion checks that the number of go cues matches the number of trials.

ii.
```python
trials = nwb["intervals/trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
n_trials_raw = trial_start.shape[0]
...
go_start = np.asarray(
    nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:],
    dtype=np.float64,
)
...
if go_start.shape[0] != n_trials_raw:
    raise ValueError(f"{session_id}: expected one go cue per trial.")
```

iii. CONVERSION_NOTES.md documents the trial structure in Step 2 with 94990 total raw trials. The one-to-one correspondence between go cues and trials is verified by assertion.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial filters:
1. **Observation window**: Requires the full decoder window `[go - 2.5s, go + 1.5s]` to lie within the shared `obs_intervals` support of all retained good units.
2. **Event validity**: Requires a valid sample_start event before the go cue within the trial, plus finite go_start and response_stop.
3. **All-zero neural**: Drops any remaining trial whose binned neural activity is entirely zero.

Notably, the AI does **not** filter out `free_water` trials.

ii.
```python
common_good_trials = (
    np.isfinite(go_start)
    & ((go_start + OFF_START_S) >= common_obs_start)
    & ((go_start + OFF_END_S) <= common_obs_end)
)
...
event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
keep_trials = common_good_trials & event_valid
...
neural_supported = np.any(neural != 0, axis=(1, 2))
if n_zero_neural_trials:
    keep_idx = keep_idx[neural_supported]
```

iii. CONVERSION_NOTES.md Step 5 documents the rationale: the AI uses the shared obs_intervals window approach rather than matching individual trial start times. The AI also adds event validity checks (requiring valid sample onset) and drops all-zero neural trials as a safety net. The absence of a `free_water` filter is not explicitly justified in the notes, though the AI states it keeps "all decoder-relevant supported trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (via `units/spike_times_index` for ragged indexing) for units with `units/classification == 'good'`. Go-cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` define the alignment.

ii.
```python
classification = read_str_array(nwb["units/classification"])
good_unit_idx = np.flatnonzero(classification == "good")
...
neural = bin_spikes_for_session(
    nwb["units/spike_times"],
    np.asarray(nwb["units/spike_times_index"][:], dtype=np.int64),
    good_unit_idx.astype(np.int64),
    keep_go_start,
)
```

iii. CONVERSION_NOTES.md Step 5 maps `units/spike_times` + go cue timestamps + classification to the neural field.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins spanning [-2.5, 1.5) s relative to the go cue (80 bins total). Bin edges are computed as absolute times, `np.searchsorted` gives cumulative spike counts at each edge, and differencing gives counts per bin. Counts are divided by `BIN_SIZE_S` (0.05) to get firing rates in Hz. The result is stored as `float16`.

ii.
```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
...
counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS + 1
)
fr[:, out_i, :] = (np.diff(counts, axis=1) / BIN_SIZE_S).astype(np.float16)
```

iii. CONVERSION_NOTES.md Step 5 documents the decision to store firing rates (not raw counts) and to use 50ms bins as specified by the instructions. Step 6 notes the use of float16 to reduce output file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. A session with no good units is dropped entirely. This retains 69,453 units across 173 sessions.

ii.
```python
classification = read_str_array(nwb["units/classification"])
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    print(f"  Skipping {session_id}: no good units in NWB classification.")
    return None
```

iii. CONVERSION_NOTES.md Step 4 documents that `classification == 'good'` is the NWB equivalent of the paper's QC classifier output, and notes the small count discrepancy (69,453 vs 69,943 in papers) due to archive version differences.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All data is aligned to the go cue. Absolute bin edges are computed as `go_times[:, None] + BIN_EDGES_REL[None, :]`, and spikes are binned against these absolute edges via `searchsorted`.

ii.
```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
...
counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS + 1
)
```

iii. CONVERSION_NOTES.md Step 5 states all streams are aligned by subtracting per-trial go-cue onset, matching the reference code and decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins spanning [-2.5, 1.5) s. No rebinning is applied; spikes are binned directly from raw spike times into these bins.

ii.
```python
BIN_SIZE_S = 0.05
OFF_START_S = -2.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
```

iii. CONVERSION_NOTES.md Step 5 documents the 50ms bin width and [-2.5, 1.5) window as specified by the decoder task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (tone onset times) and go-cue times. The AI finds the last sample_start event within the trial interval [trial_start, go_start] for each trial.

ii.
```python
sample_start_times = np.asarray(
    nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:],
    dtype=np.float64,
)
sample_start, sample_valid = get_last_events_within(sample_start_times, trial_start, go_start)
```

iii. CONVERSION_NOTES.md Step 5 documents: "For each trial, find the last `sample_start_time` between trial start and go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The sample onset is expressed relative to the go cue (`sample_start_rel = sample_start - go_start`). Then time from tone onset at each bin center is `BIN_CENTERS_REL - sample_start_rel`, which gives seconds since the tone at each bin.

ii.
```python
sample_start_rel = sample_start[keep_idx] - keep_go_start
...
tone_time = make_tone_time_matrix(sample_start_rel)
```

```python
def make_tone_time_matrix(sample_start_rel: np.ndarray) -> np.ndarray:
    return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None]).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 notes that trial-specific sample-start extraction is necessary because replays can shift tone onset earlier than the nominal -1.85s.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same go-cue-relative bin centers (`BIN_CENTERS_REL`), so each bin of the tone-onset input corresponds exactly to the same time interval as the corresponding neural bin.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
```

iii. The alignment is inherent in using the same bin grid for both neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `acquisition/BehavioralEvents/photostim_stop_times/timestamps`, using the first start and last stop within each trial's [trial_start, trial_stop] interval. These are then expressed relative to the go cue.

ii.
```python
photostim_start_times = np.asarray(
    nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"][:],
    dtype=np.float64,
)
photostim_stop_times = np.asarray(
    nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"][:],
    dtype=np.float64,
)
stim_start, _ = get_first_events_within(photostim_start_times, trial_start, trial_stop)
stim_stop, _ = get_last_events_within(photostim_stop_times, trial_start, trial_stop)
```

iii. CONVERSION_NOTES.md Step 5 maps photostim start/stop event timestamps to the photostim input. The AI chose event streams rather than trial-table columns (`photostim_onset`, `photostim_duration`).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying vector: 1 where the bin center falls within [stim_start_rel, stim_stop_rel), 0 elsewhere. Non-stimulated trials have NaN start/stop which produces all zeros.

ii.
```python
def make_photostim_matrix(stim_start_rel, stim_stop_rel):
    mat = np.zeros((stim_start_rel.shape[0], N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_start_rel) & np.isfinite(stim_stop_rel)
    if np.any(valid):
        start = stim_start_rel[valid][:, None]
        stop = stim_stop_rel[valid][:, None]
        active = (BIN_CENTERS_REL[None, :] >= start) & (BIN_CENTERS_REL[None, :] < stop)
        mat[valid] = active.astype(np.float32)
    return mat
```

iii. CONVERSION_NOTES.md Step 5 documents: "Binary vector per trial/bin: 1 if bin center is within the photostim interval relative to go cue, else 0."

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim start and stop are expressed relative to the go cue, and compared against the same `BIN_CENTERS_REL` used for neural data, ensuring bin-level alignment.

ii.
```python
stim_start_rel = stim_start[keep_idx] - keep_go_start
stim_stop_rel = stim_stop[keep_idx] - keep_go_start
```

iii. Same bin grid as neural and other inputs.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`, plus the go cue times and response window end times. The AI finds the first left and right licks in the response window [go_start, min(go_start+1.5, trial_stop)] and assigns choice based on which lick came first.

ii.
```python
left_lick_times = np.asarray(
    nwb["acquisition/BehavioralEvents/left_lick_times/timestamps"][:],
    dtype=np.float64,
)
right_lick_times = np.asarray(
    nwb["acquisition/BehavioralEvents/right_lick_times/timestamps"][:],
    dtype=np.float64,
)
choice = derive_choice_labels(
    left_lick_times, right_lick_times, keep_go_start, keep_response_stop,
)
```

iii. CONVERSION_NOTES.md Step 5 documents: "Derive actual choice from the first lick in the answer window [go_start, min(go_start + 1.5 s, trial_stop)]."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first left and right lick within the response window are found. If only one side has a lick, that's the choice. If both sides have licks, the earlier one is chosen. If neither side has a lick, choice is 2 ("no lick"). Choice is broadcast across all 80 time bins.

ii.
```python
def derive_choice_labels(left_lick_times, right_lick_times, go_start, go_stop):
    left_first, left_valid = get_first_events_within(left_lick_times, go_start, go_stop)
    right_first, right_valid = get_first_events_within(right_lick_times, go_start, go_stop)
    choice = np.full(go_start.shape, 2, dtype=np.int64)  # 2 = no lick
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    both = left_valid & right_valid
    choice[left_only] = 0
    choice[right_only] = 1
    choice[both] = (right_first[both] < left_first[both]).astype(np.int64)
    return choice
```

iii. CONVERSION_NOTES.md Step 4 documents the inconsistency found in `go_stop_times` semantics across sessions, motivating the use of raw lick timestamps with a paper-defined 1.5s answer window.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome`, which contains strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_raw = read_str_array(trials["outcome"])[keep_idx]
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_raw], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 5 maps outcome directly from NWB trial-table string labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String labels are mapped to integers (ignore=0, miss=1, hit=2) and broadcast across all 80 bins.

ii.
```python
outcome_2d = np.broadcast_to(outcome[:, None], (outcome.shape[0], N_BINS))
```

iii. Straightforward categorical encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick`, which contains strings `'no early'` and `'early'`.

ii.
```python
early_raw = read_str_array(trials["early_lick"])[keep_idx]
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int64)
```

iii. Direct mapping from NWB trial-table column.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String labels are mapped to integers (no early=0, early=1) and broadcast across all 80 bins.

ii.
```python
early_2d = np.broadcast_to(early[:, None], (early.shape[0], N_BINS))
```

iii. Straightforward categorical encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (columns: x, y, likelihood) and `/timestamps`.

ii.
```python
tongue_data = np.asarray(
    nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][:],
    dtype=np.float64,
)
tongue_times = np.asarray(
    nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][:],
    dtype=np.float64,
)
tongue_y = tongue_data[:, 1]
tongue_lik = tongue_data[:, 2]
```

iii. CONVERSION_NOTES.md Step 5 identifies this as the side-camera tongue tracking data.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Two steps:
1. **Session-level thresholds**: Frames with `likelihood >= 0.9` are considered visible. The 40th and 60th percentiles of tongue_y across all visible frames in the session define the class boundaries.
2. **Per-bin classification**: For each bin of each trial, the last frame within the bin is taken. If visible, its y value is classified (0: < q40, 1: q40 to q60, 2: > q60). If not visible, class is 3.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
session_visible = (
    np.isfinite(tongue_y) & np.isfinite(tongue_lik) & (tongue_lik >= VISIBILITY_THRESHOLD)
)
if np.any(session_visible):
    q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
```

iii. CONVERSION_NOTES.md Step 5 documents the 0.9 threshold choice and notes the bimodal likelihood distribution. The AI uses all visible frames (not bin means) for percentile computation.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses the last frame in each bin. If the last frame's time falls within the bin and its likelihood >= 0.9, the y value is compared against q40 and q60: class 0 if y < q40, class 1 if q40 <= y <= q60, class 2 if y > q60. Otherwise class 3 (not visible).

ii.
```python
out = np.full((go_times.shape[0], N_BINS), 3, dtype=np.int64)
if np.any(visible):
    out[visible & (last_y < q40)] = 0
    out[visible & (last_y >= q40) & (last_y <= q60)] = 1
    out[visible & (last_y > q60)] = 2
```

iii. The thresholding uses `<=` for the middle class boundary with q60, making it `[q40, q60]` inclusive on both sides, while the reference uses `np.digitize` which gives `[q40, q60)`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Frame times are aligned to the same go-cue-relative bin grid. For each bin, the last frame before the bin's end edge is found via `searchsorted`, and its time is checked against the bin's start edge.

ii.
```python
abs_starts = go_times[:, None] + BIN_EDGES_REL[:-1][None, :]
abs_ends = go_times[:, None] + BIN_EDGES_REL[1:][None, :]
idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS
) - 1
```

iii. CONVERSION_NOTES.md Step 5 documents alignment from `align_markers.py` patterns. The AI uses the "last frame in the bin" approach rather than averaging all frames in the bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three categories:
- **Session with no good units**: The one session with all-NaN classification is dropped (returns None).
- **Trials outside neural support**: Trials whose decoder window doesn't fit within the shared obs_intervals support are excluded. Additionally, trials where no valid sample_start event is found are excluded. Any remaining trial with all-zero neural data is dropped.
- **Missing tongue tracking**: Frames with low likelihood (< 0.9) or non-finite values are treated as invisible. Bins with no visible frame get class 3 ("not visible").

ii.
```python
if good_unit_idx.size == 0:
    return None
...
common_good_trials = (
    np.isfinite(go_start)
    & ((go_start + OFF_START_S) >= common_obs_start)
    & ((go_start + OFF_END_S) <= common_obs_end)
)
...
event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
...
neural_supported = np.any(neural != 0, axis=(1, 2))
```

iii. CONVERSION_NOTES.md documents these edge cases in Step 4 (discrepancies) and Step 10 (review).

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports that spike binning and tongue alignment are the main bottlenecks. Full conversion took ~184 seconds for 173 sessions. Per-session processing is dominated by reading spike data from HDF5 and the per-unit searchsorted loop.

ii. N/A (timing reported in CONVERSION_NOTES.md Step 9)

iii. CONVERSION_NOTES.md Step 7 estimates and Step 9 reports actual timing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `bin_spikes_for_session` iterates over each good unit, performing `searchsorted` per unit. This cannot be fully vectorized due to ragged spike arrays. The tongue alignment uses vectorized searchsorted over all bins at once.

ii.
```python
for out_i, unit_i in enumerate(good_unit_idx):
    spikes = np.asarray(
        spike_times_dataset[unit_starts[unit_i] : spike_index[unit_i]],
        dtype=np.float64,
    )
    ...
    counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(...)
```

iii. The AI notes in CONVERSION_NOTES.md Step 6 that vectorized spike binning per unit and vectorized tongue alignment were implemented as speedups.

## 10-c. What processing does the code repeat multiple times?

i. Nothing is obviously repeated. Each NWB file is opened once. Bin edges are computed once at module level. Session percentiles are computed once per session.

ii. N/A

iii. The AI's code processes each session in a single pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores neural data as `float16`, which loses precision that would be present at `float32`. The code also computes and returns `tongue_y_last` and `tongue_visible` arrays from `align_tongue_bins` which are only used for plotting, not for the final dataset. The `session_stats` dictionary contains extensive diagnostic metadata that is included in the output but not used by the decoder.

ii.
```python
fr = np.empty((go_times.shape[0], good_unit_idx.shape[0], N_BINS), dtype=np.float16)
...
return out, last_y, visible  # last_y and visible only used for plots
```

iii. CONVERSION_NOTES.md notes float16 was chosen to reduce file size but doesn't discuss potential precision implications.
