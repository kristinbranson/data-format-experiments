# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB/HDF5 files organized as one file per session under `data/sub-<subject_id>/`. Files are found via a sorted glob pattern. Each file is opened with `h5py` (not `pynwb`) and session data (units, trials, behavioral events, tracking) is read directly from HDF5 groups.

ii.
```python
DATA_ROOT = Path("/app/data")
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
# ...
for path in paths:
    result = convert_session(path, make_plot=make_plot)
```

Within `convert_session`:
```python
with h5py.File(path, "r") as nwb:
    good_units, annotations = get_good_units(nwb)
    selection = map_observed_trials(nwb, good_units)
    # ...
```

iii. The AI chose h5py over pynwb for direct HDF5 access. The trajectory notes this approach and the CONVERSION_NOTES document the 174 NWB files found and their organization.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). The AI cross-checks this against the NWB internal `general/subject/subject_id` field. Subjects are collected as a sorted unique set.

ii.
```python
subject_id = Path(path).parent.name
source_subject = decode_scalar(nwb["general/subject/subject_id"][()])
if source_subject not in {subject_id, subject_id.removeprefix("sub-")}:
    raise ValueError(...)
# ...
subjects = sorted({x["subject"] for x in converted_sessions})
```

iii. The AI uses the directory name as the subject identifier (e.g., `sub-440956`) rather than the NWB internal subject_id (the numeric ID like `440956`). A cross-check validates consistency between the two.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The session ID is derived from the filename stem with suffix cleanup. Sessions are processed sequentially and those with no good units are skipped.

ii.
```python
session_id = Path(path).stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
with h5py.File(path, "r") as nwb:
    good_units, annotations = get_good_units(nwb)
    if len(good_units) == 0:
        print(f"SKIP {session_id}: no classifier-good units", flush=True)
        return None
```

iii. CONVERSION_NOTES confirm 174 NWB files, one session with 0 classifier-good units skipped, yielding 173 usable sessions matching the paper.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI validates that go-cue event count matches trial table length.

ii.
```python
trial_group = nwb["intervals/trials"]
starts = trial_group["start_time"][:]
stops = trial_group["stop_time"][:]
n_original = len(starts)
# ...
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
if len(go_times) != n_original:
    raise ValueError("Go-cue count does not match trial-table length")
```

iii. The AI uses the trials table directly and validates consistency with go-cue events.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a multi-stage filtering pipeline:
1. Map `obs_intervals` to trial rows (same as reference)
2. Apply `is_good_trials` — require ALL retained good units to be valid for each trial
3. Require the full -2.5 to +1.5 s window to lie within the recording span
4. Exclude trials where ALL neurons have zero spikes (population-all-zero)

The AI does NOT explicitly filter on `free_water`. Instead, the population-all-zero check catches most of those trials indirectly.

ii.
```python
# obs_intervals mapping
all_units_valid = np.all(unit_trial_validity[good_units, :], axis=0)
after_unit_qc = observed_trial_indices[all_units_valid]
# full window check
full_window = (
    (go_times[after_unit_qc] + TIME_EDGES[0] >= recording_start - 1e-9)
    & (go_times[after_unit_qc] + TIME_EDGES[-1] <= recording_stop + 1e-9)
)
selected = after_unit_qc[full_window]
# population-all-zero exclusion
neural_data_present = np.any(rates > 0, axis=(1, 2))
if n_all_zero_neural_trials:
    trial_indices = trial_indices[neural_data_present]
```

iii. CONVERSION_NOTES Step 4 explains: "Retain trials with valid go-cue timestamps and neural recording coverage." Step 10 notes: "2,423 residual population-zero truncated windows are removed." The AI discovered the is_good_trials validity mask and chose to use it because "the target workflow explicitly says to honor valid-period variables."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times) with `units/spike_times_index` for ragged indexing, plus `BehavioralEvents/go_start_times/timestamps` for alignment. Only units with `classification == 'good'` are used.

ii.
```python
spike_index = nwb["units/spike_times_index"][:].astype(np.int64)
previous = np.r_[0, spike_index[:-1]]
all_spikes = nwb["units/spike_times"][:]
```

iii. CONVERSION_NOTES Step 5 documents: spike_times are binned into 80 adjacent half-open bins and divided by 0.05 to get Hz.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins aligned to the go cue. For each good unit, `searchsorted` locates spike positions at all trial bin edges, then `np.diff` gives counts per bin, divided by bin width (0.05s) for Hz. No smoothing, normalization, or baseline subtraction.

ii.
```python
absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
```

iii. CONVERSION_NOTES Step 5: "count absolute spike timestamps in 80 adjacent half-open bins [go-2.5, go+1.5) of width 0.05 s; divide counts by 0.05 to Hz."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. The AI raises a ValueError if a good unit has an empty CCF annotation. A session with no good units is skipped.

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

iii. CONVERSION_NOTES Step 4: "Use classification == 'good'; do not re-threshold the 15 metrics and do not apply the method-paper's later analysis-specific >2-Hz filter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes and events share the same absolute session clock. Bin edges are constructed as `go_time + relative_edges` for each trial. No resampling or interpolation is needed.

ii.
```python
absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
```

iii. CONVERSION_NOTES Step 4: "Subtract each trial's absolute go timestamp conceptually via absolute bin edges; no further synchronization or offset is required."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50ms bins spanning -2.5 to +1.5 s relative to go cue. Bin edges are defined with `np.linspace(-2.5, 1.5, 81)`. No rebinning is applied — spikes are directly counted into these bins.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
BIN_WIDTH_S = 0.05
```

iii. CONVERSION_NOTES Step 5: "Requested bins replace reference 40-ms sliding windows."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times/timestamps` (tone onsets) and `BehavioralEvents/go_start_times/timestamps` (go cues). For each trial, the last sample onset before the go cue is selected.

ii.
```python
def final_tone_onsets(nwb: h5py.File, trial_indices: np.ndarray) -> np.ndarray:
    sample_starts = nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:]
    go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
    selected_go = go_times[trial_indices]
    positions = np.searchsorted(sample_starts, selected_go, side="left") - 1
    tones = sample_starts[positions]
    return tones
```

iii. CONVERSION_NOTES Step 4: "Select the last sample-start event before the final go cue in the trial, i.e. the tone epoch that led into the aligned delay/go."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the absolute bin centers are computed (`go_time + TIME_CENTERS`), then the tone onset is subtracted to give signed time from tone onset.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
time_from_tone = absolute_centers - tone_onsets[:, None]
```

iii. The computation is straightforward — absolute bin centers minus tone onset time.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural data (defined by go cue + relative offsets) are used for the time-from-tone computation, ensuring alignment.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
time_from_tone = absolute_centers - tone_onsets[:, None]
```

iii. Both neural and input share the same go-cue-aligned time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps` — the paired absolute event timestamps.

ii.
```python
event_root = nwb["acquisition/BehavioralEvents"]
photo_starts = event_root["photostim_start_times/timestamps"][:]
photo_stops = event_root["photostim_stop_times/timestamps"][:]
```

iii. CONVERSION_NOTES Step 5: "Binary 1 where absolute bin center lies in any paired half-open laser interval [start, stop), else 0."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Uses a clever event-counting approach: for each bin center, counts how many photostim starts have occurred vs how many stops. If starts > stops, light is on (1), else off (0).

ii.
```python
started = np.searchsorted(photo_starts, absolute_centers, side="right")
stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
photostim_on = (started > stopped).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5: "Evaluate the paired absolute event intervals on decoder time-bin centers; do not infer from power alone."

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Evaluated at the same absolute bin centers as the neural data, using the go-cue-aligned time grid.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
# photostim evaluated at these same centers
```

iii. Same time grid as neural and tone input.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table. Hit maps to the instructed side, miss to the opposite side, ignore to no lick.

ii.
```python
instruction = decode_array(trials["trial_instruction"])[trial_indices]
outcome_text = decode_array(trials["outcome"])[trial_indices]
actual_choice = np.full(len(trial_indices), 2, dtype=np.int64)
hit = outcome_text == "hit"
miss = outcome_text == "miss"
actual_choice[hit & (instruction == "left")] = 0
actual_choice[hit & (instruction == "right")] = 1
actual_choice[miss & (instruction == "left")] = 1
actual_choice[miss & (instruction == "right")] = 0
```

iii. CONVERSION_NOTES Step 4: "Map hit to instructed side, miss to opposite side, ignore to no lick."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as left=0, right=1, no lick=2. Per-trial value is broadcast across all 80 time bins.

ii.
```python
outputs[:, 0, :] = actual_choice[:, None]
```

iii. The choice is a per-trial label broadcast across time.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, containing strings "ignore", "miss", "hit".

ii.
```python
outcome_text = decode_array(trials["outcome"])[trial_indices]
outcome_map = {name: idx for idx, name in enumerate(OUTCOME_VALUES)}
outcomes = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int64)
```

iii. Direct mapping from existing trial labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to ignore=0, miss=1, hit=2 and broadcast across all 80 time bins.

ii.
```python
OUTCOME_VALUES = ["ignore", "miss", "hit"]
outputs[:, 1, :] = outcomes[:, None]
```

iii. Standard categorical encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing "no early" or "early".

ii.
```python
early_text = decode_array(trials["early_lick"])[trial_indices]
early = (early_text == "early").astype(np.int64)
```

iii. Direct mapping from existing trial labels.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1 and broadcast across all 80 time bins.

ii.
```python
early = (early_text == "early").astype(np.int64)
outputs[:, 2, :] = early[:, None]
```

iii. Standard binary encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has columns (x, y, likelihood) with associated timestamps.

ii.
```python
tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
timestamps = tracking["timestamps"][:]
marker = tracking["data"][:]
x = marker[:, 0].astype(np.float64, copy=False)
y = marker[:, 1].astype(np.float64, copy=True)
likelihood = marker[:, 2].astype(np.float64, copy=False)
```

iii. Same source as reference — the side camera tongue tracking.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Multiple steps:
1. Mark frames as visible where likelihood >= 0.9 (and finite)
2. Apply 5-SD velocity outlier correction: compute frame-to-frame speeds, and frames exceeding mean+5*SD are interpolated from neighboring visible frames
3. Compute p40 and p60 percentiles from ALL visible (corrected) y values across the session
4. For each bin center, find the nearest camera frame; if visible, classify y relative to p40/p60; otherwise class 3

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
visible = (np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood)
           & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD))
# 5-SD velocity outlier correction
speeds[valid_pairs] = displacement[valid_pairs] / dt[valid_pairs]
cutoff = np.mean(reference_speeds) + 5 * np.std(reference_speeds)
# interpolation of outliers
y[target] = np.interp(timestamps[target], timestamps[interpolation_basis], y[interpolation_basis])
# percentiles from visible frames
p40, p60 = np.percentile(y[visible], [40, 60])
# nearest-frame sampling
nearest = np.where(choose_left, left_clipped, right_clipped)
classes[sampled_visible & (sampled_y < p40)] = 0
classes[sampled_visible & (sampled_y >= p40) & (sampled_y <= p60)] = 1
classes[sampled_visible & (sampled_y > p60)] = 2
```

iii. CONVERSION_NOTES Step 5: "Use likelihood >=0.9 as visible; repair high-confidence >5-SD velocity outliers by interpolation; compute p40/p60 from all visible corrected session y values; sample nearest frame to each bin center." The velocity correction comes from the method paper's tracking pipeline.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of visible (corrected) y values define two thresholds. y < p40 = class 0, p40 <= y <= p60 = class 1, y > p60 = class 2, not visible = class 3.

ii.
```python
p40, p60 = np.percentile(y[visible], [40, 60])
classes[sampled_visible & (sampled_y < p40)] = 0
classes[sampled_visible & (sampled_y >= p40) & (sampled_y <= p60)] = 1
classes[sampled_visible & (sampled_y > p60)] = 2
```

iii. Follows the instructions for per-session discretization. Percentiles are computed on raw visible frames (not bin means).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (same go-cue-aligned grid as neural data), the nearest camera frame is found. If the frame is visible and close enough (within 1.5x the median frame interval), its y value is classified; otherwise class 3.

ii.
```python
flat_centers = absolute_centers.ravel()
right = np.searchsorted(timestamps, flat_centers, side="left")
# nearest-neighbor logic
median_dt = float(np.median(np.diff(timestamps)))
covered = ((flat_centers >= timestamps[0]) & (flat_centers <= timestamps[-1])
           & (np.abs(timestamps[nearest] - flat_centers) <= 1.5 * median_dt))
```

iii. Uses nearest-frame sampling rather than bin-mean averaging. The 1.5x median interval threshold prevents associating frames that are too far away.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three main cases:
- **Session with no QC labels**: The one session where all classification values are NaN is skipped (0 good units).
- **Partial recording sessions**: Nine sessions where obs_intervals covers fewer trials than the trial table are handled by mapping intervals to trial rows exactly.
- **is_good_trials false entries**: Trials where any good unit is marked invalid are excluded.
- **Truncated spike streams**: Trials where all neurons fire zero spikes are excluded as population-all-zero.
- **Missing tongue frames**: Bins with no nearby visible frame get class 3.
- **Velocity outliers in tongue tracking**: Corrected by interpolation.

ii.
```python
# Session skip
if len(good_units) == 0:
    return None
# is_good_trials filtering
all_units_valid = np.all(unit_trial_validity[good_units, :], axis=0)
# Population-all-zero
neural_data_present = np.any(rates > 0, axis=(1, 2))
```

iii. CONVERSION_NOTES Step 10 documents each edge case in detail.

## 10-a. What are the most time-consuming steps of the code?

i. HDF5 file reading (especially reading the full spike_times array) and the per-unit searchsorted loop dominate. Full conversion takes ~195 seconds for 173 sessions (~1.1 s/session). Pickle writing takes ~15 seconds for 11.2 GiB.

ii. N/A (timing is printed during execution)

iii. CONVERSION_NOTES Step 7 estimates total time well under 15 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `bin_spikes` iterates over good units because each has a different number of spikes (ragged storage). The tongue processing loops over trials in the reference but the AI uses vectorized nearest-frame sampling instead.

ii.
```python
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
    rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
```

iii. The per-unit loop cannot be vectorized due to ragged spike arrays.

## 10-c. What processing does the code repeat multiple times?

i. The AI reads `go_start_times` multiple times within a single session (in `map_observed_trials`, `final_tone_onsets`, `construct_inputs`, `bin_spikes`). Each read is an HDF5 dataset access that could be cached.

ii.
```python
# In map_observed_trials:
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
# In final_tone_onsets:
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
# In construct_inputs:
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
# In bin_spikes:
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

iii. This redundant reading is a minor inefficiency but not a major bottleneck.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI performs several operations that produce metadata/provenance but are not used for decoding:
- Velocity outlier correction for tongue tracking (the reference does not do this)
- Detailed per-session info dictionaries with extensive provenance
- Cross-validation checks (subject ID vs directory name, photostim count matching)

ii.
```python
# Velocity outlier correction
cutoff = np.mean(reference_speeds) + 5 * np.std(reference_speeds)
outlier_frames[1:] = np.isfinite(speeds) & (speeds > cutoff)
# Extensive metadata
info = {
    "session_id": session_id,
    "source_file": os.path.relpath(path, "/app"),
    # ... many more fields
}
```

iii. These add robustness and documentation but don't contribute directly to decoder performance.
