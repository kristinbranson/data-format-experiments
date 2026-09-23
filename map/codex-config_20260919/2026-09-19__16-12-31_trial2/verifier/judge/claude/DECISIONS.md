# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files via `glob` over `sub-*/` directories under the data root. Each file is opened with `h5py` (not `pynwb`) and processed sequentially. A two-pass approach is used: first a `discover_inventory` pass reads all files to collect the global set of subjects and brain regions, then a second `convert_session` pass processes each file.

ii.
```python
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
# First pass:
subjects, brain_regions = discover_inventory(paths)
# Second pass:
for path in paths:
    result = convert_session(path, subject_lookup, region_lookup, ...)
```

```python
def convert_session(path, ...):
    with h5py.File(path, "r") as nwb:
        units = nwb["units"]
        trial_table = nwb["intervals/trials"]
        ...
```

iii. The AI noted in CONVERSION_NOTES Step 2 that the dataset is DANDI 000363 with 174 NWB files in 28 `sub-<id>/` directories, one file per session. Using h5py instead of pynwb was chosen for direct HDF5 access. The two-pass approach is needed to build global region/subject vocabularies before converting individual sessions.

## 1-b. How are the data split into subjects?

i. Each NWB file's `general/subject/subject_id` field is read and verified against the folder name. A global sorted list of unique subject IDs is built during the inventory pass, and each session is mapped to its subject index.

ii.
```python
subject = decode_scalar(nwb["general/subject/subject_id"][()])
folder_subject = Path(path).parent.name.removeprefix("sub-")
if subject != folder_subject:
    raise ValueError(...)
subjects.add(subject)
```

```python
"subject_idx": np.asarray([x["subject_idx"] for x in converted], dtype=np.int32),
```

iii. The AI's CONVERSION_NOTES document that 28 subjects were found, matching the dandiset metadata. The folder-to-NWB subject ID consistency check is an additional validation.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The session ID is derived from the filename by stripping the `_behavior+ecephys` suffix. Sessions are processed in sorted file-path order.

ii.
```python
def session_id_from_path(path: str) -> str:
    name = Path(path).name
    return name.split("_behavior")[0]
```

iii. The AI documented that 174 NWB files correspond to 174 potential sessions; 173 survive after excluding the one session with no classifier-curated units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB intervals/trials table. The AI reads the trial table and go-cue timestamps. Only the first `n_trials_recorded` trials (matching the `is_good_trials` stability matrix width) are considered, which excludes trailing behavior-only trials that lack ephys data.

ii.
```python
trial_table = nwb["intervals/trials"]
n_trials_table = len(trial_table["id"])
stability_all = units["is_good_trials"][:]
stability = stability_all[good_indices]
n_trials_recorded = stability.shape[1]
go_all = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:n_trials_recorded]
trial_starts = trial_table["start_time"][:n_trials_recorded]
```

iii. The AI noted in CONVERSION_NOTES Step 2 that eight files have trailing behavior-table rows beyond ephys observation intervals, and the `is_good_trials` matrix shape identifies the ephys trial count (93,310 recorded trials across 173 sessions).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three stages: (1) Only trials within the `is_good_trials` stability matrix width (ephys-recorded trials); (2) Trials where ALL curated units have `is_good_trials==True` (the `stable_mask`); (3) After spike binning, trials with entirely zero neural population activity are removed. Free-water and auto-water trials are NOT explicitly excluded (they're tracked in metadata but not filtered). Additionally, tone onset validity is checked but effectively all trials pass this.

ii.
```python
stable_mask = np.all(stability, axis=0)
# ...
coverage_mask = np.isfinite(tone_all)
keep = stable_mask & coverage_mask
# ...
# After spike binning:
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    rates = rates[neural_present]
    keep_idx = keep_idx[neural_present]
    # ...
```

iii. The AI justified not applying the paper's `regular` trial mask (which excludes stimulation, early-lick, water, and ignore trials) because those are required decoder variables. The stability mask ensures all curated units have valid recordings for each retained trial. The all-zero neural filter catches remaining trials beyond actual spike coverage. Final count: 90,253 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times) and `units/spike_times_index` (ragged array offsets) for units with `classification == 'good'`. Go-cue times from `BehavioralEvents/go_start_times` define bin edges.

ii.
```python
spike_ends = units["spike_times_index"][:]
spike_values = units["spike_times"]
for out_unit, source_unit in enumerate(good_indices):
    start = 0 if source_unit == 0 else int(spike_ends[source_unit - 1])
    stop = int(spike_ends[source_unit])
    spikes = spike_values[start:stop]
```

iii. Same source variable as the reference — spike_times is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50-ms bins spanning -2.5 to +1.5 s relative to the go cue. For each unit, all trial edges are flattened and `searchsorted` is called once to get running spike counts at each edge; differencing gives counts per bin. Counts are divided by 0.05 s to give firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
absolute_edges = go[:, None] + BIN_EDGES[None, :]
rates = np.empty((n_trials, n_units_good, N_TIME), dtype=np.float32)
for out_unit, source_unit in enumerate(good_indices):
    # ...
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
    counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
    rates[:, out_unit, :] = counts / BIN_WIDTH_S
```

iii. This matches the reference code's `sliding_histogram` approach. The AI validates that rates are non-negative and multiples of 20 Hz (the minimum rate for a single spike in a 50-ms bin).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. No additional firing-rate threshold or individual QC metric filtering is applied. A session with zero good units is skipped entirely (one session: sub-440958_ses-20190216T162508). This yields 69,453 good units across 173 sessions.

ii.
```python
classification = decode_array(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
if len(good_indices) == 0:
    print(f"SKIP {sid}: no classifier-curated units", flush=True)
    return None
```

iii. The AI documented that the classifier QC (from the ChenLiuEtAl2023 spike sorting paper) is the authoritative inclusion rule. The 2-Hz firing rate filter used in the method paper was specific to video-prediction analyses and is not applied here. The 69,453 vs 69,943 paper figure discrepancy is attributed to the published data snapshot.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB timestamps share one absolute session clock. Bin edges are computed as go-cue time plus relative offsets (-2.5 to +1.5 s). Spikes are binned against these absolute edges using `searchsorted`.

ii.
```python
absolute_edges = go[:, None] + BIN_EDGES[None, :]
edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
```

iii. No resampling or interpolation is needed since everything shares the same clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50-ms non-overlapping bins, 80 bins per trial spanning -2.5 to +1.5 s relative to go cue. The bin grid is defined once as 81 edges.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_WIDTH_S = 0.050
BIN_EDGES = np.arange(OFF_START, OFF_END + BIN_WIDTH_S / 2, BIN_EDGES)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = len(BIN_CENTERS)  # 80
```

iii. Matches the decoder task specification exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onset timestamps) and `trial_starts` (trial start times), plus the go-cue times. The **first** `sample_start_times` event at or after each trial's start time is taken as the tone onset.

ii.
```python
def first_tone_onsets(trial_starts, go_times, sample_starts):
    indices = np.searchsorted(sample_starts, trial_starts, side="left")
    tone = sample_starts[indices]
    valid = (tone >= trial_starts - 1e-9) & (tone <= go_times + 1e-9)
    # ...
    return tone
```

iii. The AI justified using the FIRST tone per trial: "tone onset denotes the trial's initial instruction onset." The AI noted that early lick replays the sample epoch, producing multiple sample_start_times events per trial, and explicitly chose the first one between trial start and go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. At each bin center (absolute time = go + BIN_CENTER offset), the elapsed time since tone onset is computed: `absolute_bin_center - tone_onset`. This gives a continuous, time-varying input.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
inputs[:, 0, :] = centers - tone[:, None]
```

iii. No additional processing beyond the subtraction.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers used for the time-from-tone computation are the same centers as the neural firing-rate bins, ensuring perfect alignment.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]  # same grid for neural and inputs
```

iii. Both neural and input time grids are derived from the same go-cue-relative bin structure.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, plus `start_time` (trial start) to convert onset to absolute time. Trials with `N/A` onset are treated as no stimulation.

ii.
```python
onset = numeric_or_nan(trial_table["photostim_onset"][:n_trials_recorded])[keep_idx]
duration = numeric_or_nan(trial_table["photostim_duration"][:n_trials_recorded])[keep_idx]
kept_trial_starts = trial_starts[keep_idx]
stim_start = kept_trial_starts + onset
stim_stop = stim_start + duration
```

iii. The `numeric_or_nan` helper converts string `N/A` values to NaN, cleanly handling unstimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is marked 1 if its center falls within [stim_start, stim_stop), 0 otherwise. NaN start/stop (unstimulated trials) naturally evaluate to False, keeping all bins at 0.

ii.
```python
inputs[:, 1, :] = (
    np.isfinite(stim_start[:, None])
    & np.isfinite(stim_stop[:, None])
    & (centers >= stim_start[:, None])
    & (centers < stim_stop[:, None])
).astype(np.float32)
```

iii. The explicit `np.isfinite` checks ensure unstimulated trials remain all-zero. The result is a binary time series.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The `centers` variable used for photostimulation comparison is the same absolute bin-center array used for neural binning (go + BIN_CENTERS), ensuring alignment.

ii.
```python
centers = centers_all[keep]  # same centers used for neural binning
inputs[:, 1, :] = (... & (centers >= stim_start[:, None]) & (centers < stim_stop[:, None])) ...
```

iii. Same bin grid as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table. There is no direct choice column.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
instructions = decode_array(trial_table["trial_instruction"][:n_trials_recorded])[keep_idx]
choices = choice_codes(outcomes, instructions)
```

iii. Same derivation approach as the reference.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `ignore` maps to `no lick` (2); `hit` maps to the instructed side (left=0, right=1); `miss` maps to the opposite side. The per-trial value is broadcast across all 80 time bins.

ii.
```python
def choice_codes(outcomes, instructions):
    code = np.full(len(outcomes), 2, dtype=np.int8)  # ignore -> no lick
    hit = outcomes == "hit"
    miss = outcomes == "miss"
    code[hit & (instructions == "left")] = 0
    code[hit & (instructions == "right")] = 1
    code[miss & (instructions == "right")] = 0
    code[miss & (instructions == "left")] = 1
    return code
```

```python
outputs[:, 0, :] = choices[:, None]
```

iii. Logically equivalent to the reference's approach using `np.where`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes = decode_array(trial_table["outcome"][:n_trials_recorded])[keep_idx]
outcome_lookup = {name: i for i, name in enumerate(OUTCOME_VALUES)}
outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
```

iii. Direct mapping, same as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings mapped to integers: ignore=0, miss=1, hit=2. Broadcast across 80 time bins.

ii.
```python
OUTCOME_VALUES = ["ignore", "miss", "hit"]
outcome_codes = np.asarray([outcome_lookup[x] for x in outcomes], dtype=np.int8)
outputs[:, 1, :] = outcome_codes[:, None]
```

iii. Same encoding as reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing `'no early'` and `'early'`.

ii.
```python
early_labels = decode_array(trial_table["early_lick"][:n_trials_recorded])[keep_idx]
early_codes = (early_labels == "early").astype(np.int8)
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` maps to 1, `'no early'` maps to 0. Broadcast across 80 time bins.

ii.
```python
early_codes = (early_labels == "early").astype(np.int8)
outputs[:, 2, :] = early_codes[:, None]
```

iii. Same encoding as reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` data (x, y, likelihood) with corresponding timestamps. Column 1 is tongue y, column 2 is likelihood.

ii.
```python
tongue_group = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_t = tongue_group["timestamps"][:]
tongue_data = tongue_group["data"][:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Same source variable as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood < 0.9 are excluded from percentile computation. Session-wide 40th and 60th percentiles of visible tongue y values are computed from **raw visible frames** (not bin means). For each time bin, the nearest preceding camera frame is found via `searchsorted`; if that frame is within 10ms and visible (likelihood >= 0.9), the y value is classified: <q40 → 0, q40<=y<=q60 → 1, >q60 → 2. Otherwise class 3 (not visible).

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
session_visible = (np.isfinite(tongue_y) & np.isfinite(tongue_likelihood)
                   & (tongue_likelihood >= TONGUE_LIKELIHOOD_THRESHOLD))
q40, q60 = np.percentile(tongue_y[session_visible], [40, 60])

# Nearest-preceding frame at each bin center:
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
# ...
tongue_codes[sampled_visible & (sampled_y < q40)] = 0
tongue_codes[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_codes[sampled_visible & (sampled_y > q60)] = 2
```

iii. The AI chose 0.9 likelihood threshold based on standard DLC practice and the bimodal likelihood distribution. Percentiles are computed from raw frames rather than bin means. The nearest-preceding-frame approach mirrors the reference code's `align_embedding_vecs_between_lims` function.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four categories: 0 (< 40th percentile), 1 (40th to 60th percentile, inclusive both ends), 2 (> 60th percentile), 3 (not visible). The boundary convention is: `<q40` → 0, `>=q40 AND <=q60` → 1, `>q60` → 2.

ii.
```python
tongue_codes[sampled_visible & (sampled_y < q40)] = 0
tongue_codes[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_codes[sampled_visible & (sampled_y > q60)] = 2
```

iii. The AI uses explicit conditional comparisons rather than `np.digitize`. The boundary between class 1 and class 2 is `<=q60` (inclusive), unlike `np.digitize` which would place values exactly at q60 in class 2.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (on the same go-cue-relative grid as neural data), the nearest preceding camera frame is found via `searchsorted(tongue_t, center, side="right") - 1`. A frame is considered "current" if its timestamp is within 10ms before the bin center. If the frame is current and the tongue is visible (likelihood >= 0.9), the y value is discretized; otherwise class 3.

ii.
```python
centers_all = go_all[:, None] + BIN_CENTERS[None, :]
frame_idx_all = np.searchsorted(tongue_t, centers_all, side="right") - 1
safe_frame_idx = np.clip(frame_idx_all, 0, len(tongue_t) - 1)
frame_lag = centers_all - tongue_t[safe_frame_idx]
frame_is_current_all = (frame_lag >= -1e-9) & (frame_lag <= 0.010)
```

iii. The 10ms threshold accommodates the ~3.4ms camera frame spacing. Inter-trial camera gaps result in class 3 rather than trial exclusion.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) The session with all-NaN classification/anatomy is skipped (zero good units). (2) Trailing behavior-only trials beyond ephys coverage are excluded via `is_good_trials` matrix width. Trials where some units are unstable are excluded via `stable_mask`. Trials with all-zero neural population after binning are excluded post-hoc. (3) Camera frames with low likelihood or temporal gaps are assigned tongue class 3 (not visible) rather than excluded.

ii.
```python
# All-NaN QC session:
if len(good_indices) == 0:
    return None

# Stability filtering:
stable_mask = np.all(stability, axis=0)

# All-zero neural:
neural_present = np.any(rates != 0, axis=(1, 2))
if not np.all(neural_present):
    rates = rates[neural_present]
    # ...

# Tongue missing frames:
tongue_codes = np.full((n_trials, N_TIME), 3, dtype=np.int8)  # default: not visible
```

iii. The AI documented these cases extensively in CONVERSION_NOTES Steps 9-10, including iterative fixes for camera timestamp gaps and off-by-one observation intervals.

## 10-a. What are the most time-consuming steps of the code?

i. The inventory pass (reading all 174 files for subject/region discovery) takes ~4.65s. Per-session conversion takes 0.26-1.92s depending on unit/trial count, with the full conversion completing in ~177s. Pickle writing is additional time. The two-pass design doubles the file I/O compared to a single-pass approach.

ii. N/A (timing from conversion output)

iii. The AI printed timing per session and total timing. The two-pass approach is the main structural overhead compared to the reference's single-pass design.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in spike binning (one `searchsorted` per unit) cannot be eliminated due to ragged spike arrays. The AI vectorized the trial dimension within each unit call. The tongue frame lookup is vectorized across all bins using `searchsorted`.

ii.
```python
for out_unit, source_unit in enumerate(good_indices):
    # Vectorized over all trials:
    edge_positions = np.searchsorted(spikes, absolute_edges.ravel(), side="left")
    counts = np.diff(edge_positions.reshape(n_trials, N_TIME + 1), axis=1)
```

iii. Same vectorization strategy as the reference.

## 10-c. What processing does the code repeat multiple times?

i. The two-pass design reads every NWB file twice: once in `discover_inventory` (to collect subjects and regions) and once in `convert_session` (for actual processing). Within a session, nothing is recomputed.

ii.
```python
subjects, brain_regions = discover_inventory(paths)  # Pass 1
for path in paths:
    result = convert_session(path, ...)  # Pass 2
```

iii. The inventory pass is a design choice to have global region/subject vocabularies before conversion. The reference avoids this by building vocabularies incrementally during a single pass.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `discover_inventory` pass reads classification and annotation data from every file, including the one that is later skipped. The per-session `session_stats` dictionary collects detailed statistics (auto_water counts, tongue percentiles, example tone timing) that go into metadata but are not used by the decoder. The `video_has_data` computation is performed but not used for filtering (only tracked in metadata).

ii.
```python
# video_has_data is computed but only stored in metadata:
video_has_data = np.any(frame_is_current_all, axis=1)
session_stats = {
    # ...
    "n_trials_with_any_video": int(np.sum(video_has_data[keep_idx])),
    "auto_water_trials_converted": int(np.sum(auto_water != 0)),
    # ...
}
```

iii. These are lightweight computations with negligible performance impact, included for documentation/debugging purposes.
