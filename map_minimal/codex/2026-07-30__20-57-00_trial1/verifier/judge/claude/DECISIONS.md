# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the `/app/data` directory for NWB files organized in subject subdirectories. Each `.nwb` file is opened with `h5py` and processed as one session. The code iterates over sorted subject directories and sorted filenames to ensure deterministic ordering.

ii.
```python
def load_sorted_nwb_paths(data_root: str) -> list[str]:
    paths = []
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for filename in sorted(os.listdir(subject_path)):
            if filename.endswith(".nwb"):
                paths.append(os.path.join(subject_path, filename))
    return paths
```

iii. The AI states in CONVERSION_NOTES.md: "Source NWB files: 174". The data is in NWB format (converted from the original DataJoint/MATLAB export), so using h5py to read NWB files is the correct approach for the available data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the parent directory name of each NWB file (e.g., `/app/data/sub-XXXX/`). All unique subjects are sorted and assigned indices.

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))

# In build_dataset:
subjects = sorted({session["subject"] for session in session_results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI reports 28 subjects, consistent with the dataset spanning multiple mice. The NWB directory structure encodes the subject identity in the parent folder name.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from the filename (without extension).

ii.
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
```

iii. The AI found 174 NWB files (sessions), excluded 1 with zero good units, keeping 173 sessions. This matches the paper's statement of "69,943 good units recorded across 173 behavioral sessions."

## 1-d. How are the data split into trials?

i. The AI uses the first good unit's `obs_intervals` to identify which behavioral trials have electrophysiological coverage. It matches obs_intervals back to the behavioral trial table by exact start/stop times, then only includes those matched trials. This avoids including behavior-only trials that lack neural data.

ii.
```python
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
trial_indices = []
for start, stop in first_unit_obs:
    key = (round(float(start), 4), round(float(stop), 4))
    if key not in trial_lookup:
        raise ValueError(...)
    trial_indices.append(trial_lookup[key])
trial_indices = np.asarray(trial_indices, dtype=np.int64)
```

iii. From CONVERSION_NOTES.md: "For sessions where the behavior table contains more trials than the ephys recording, trials are matched from the first good unit's obs_intervals back to the behavioral trial table. This avoids including behavior-only trials with no neural recording."

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter trials based on behavioral quality criteria. All ephys-covered trials are retained, including early-lick, no-response (ignore), photostimulation, auto-water, and free-water trials. The only session-level filter is excluding sessions with zero good units.

ii.
```python
# In convert_session - no trial filtering is applied:
trial_start = behavior_trial_start[trial_indices]
trial_stop = behavior_trial_stop[trial_indices]
# All trials from obs_intervals are used directly
```

iii. From CONVERSION_NOTES.md: "All ephys-covered trials are retained, including photostim, early-lick, and ignore trials. This differs from some reference analysis masks because the requested decoder outputs explicitly require those conditions."

The reference code's `get_regular_trial_mask()` filters out early lick, auto water, free water, no response, and stimulation trials. The AI justifies keeping them because the instructions require decoding early_lick, outcome (including ignore), and photostimulation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (flat array of all spike timestamps) and `units/spike_times_index` (index array delineating per-unit boundaries), filtered by `units/classification == 'good'`.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
# ...
spike_counts = bin_spike_counts_for_good_units(
    spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
    spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
    good_unit_mask=good_unit_mask,
    ...
)
```

iii. From CONVERSION_NOTES.md: "Units are filtered with classification == 'good', matching the published classifier-based QC workflow."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins within the [-2.5, 1.5)s window relative to go cue onset. Spike counts are then converted to firing rates by dividing by the bin width (0.05s). The result is stored as float16.

ii.
```python
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
# ...
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. From CONVERSION_NOTES.md: "Neural data: spike counts per 50 ms bin, converted to firing rates by dividing by 0.05 s."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are included. No additional quality metrics (ISI violation, amplitude cutoff, presence ratio, etc.) are applied beyond the classifier-based label. Sessions with zero good units are excluded entirely.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. From CONVERSION_NOTES.md: "Session inclusion follows the dataset-level QC rule visible in the NWB files and the papers' spike-QC description: keep sessions with at least one unit whose classification is good." The paper describes using region-specific classifiers trained on manual curation labels to determine which units are "good".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue onset. For each trial, the go cue time is found from `BehavioralEvents/go_start_times`, and spike times are expressed relative to this time. The window [-2.5, 1.5)s around the go cue is extracted.

ii.
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
# In bin_spike_counts_for_good_units:
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
```

iii. The instructions specify: "Temporally align based on Go cue onset. Extract 2.5 s before to 1.5 s after the go cue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (0.05s). Spike times are binned directly into 50ms non-overlapping bins from the raw spike timestamps; no rebinning from a finer resolution is needed.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```

iii. The instructions specify: "Use 50-ms-width bins for computing firing rates." The reference code uses different bin widths (40ms with 3.4ms stride in `preprocess_all_ephys.py`, or 100ms with 50ms stride in the standalone script), but the instructions override this to 50ms.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times/timestamps` (tone/sample onset times) and `BehavioralEvents/go_start_times/timestamps` (go cue times).

ii.
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
# ...
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
```

iii. The AI uses the last sample_start_times event before the go cue for each trial, which accounts for early-lick trial replays where the sample epoch may be presented multiple times.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset time is computed relative to the go cue (`sample_rel = sample_start - go_time`, which is negative). Then at each bin center, the time from tone onset is `bin_center - sample_rel`, yielding a continuously increasing value representing elapsed time since tone onset.

ii.
```python
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. This produces a continuous, time-varying input as specified in the instructions: "Time from tone onset in seconds (continuous, time-varying)."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time from tone onset is computed at the same bin centers used for the neural data (50ms bins from -2.5 to 1.5s relative to go cue). Each bin center's value is the elapsed time since the sample/tone onset, ensuring perfect temporal alignment with the neural data.

ii.
```python
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. Both neural data and this input share the same bin centers (`BIN_CENTERS_S`), ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` in the NWB trials table.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```

iii. The reference code uses `stimulation` array columns `[laser_power, stim_type, laser_on_time, laser_off_time]` from the .mat files. The NWB equivalent fields are `photostim_onset` and `photostim_duration`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostim onset (relative to trial start) is converted to go-cue-relative time. A binary 0/1 value is assigned per bin: 1 if the photostim interval overlaps the bin, 0 otherwise. Trials without photostimulation (N/A onset or duration) get all zeros.

ii.
```python
onset = as_float_or_none(onset_values[trial_idx])
duration = as_float_or_none(duration_values[trial_idx])
if onset is None or duration is None:
    continue
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. From CONVERSION_NOTES.md: "Input photostim_on: binary per bin, 1 if the photostim interval overlaps the bin." This matches the instructions: "Whether photostimulation is on at every time point (discrete, time-varying)."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation timing is converted from trial-start-relative to go-cue-relative coordinates, then binned using the same bin edges as the neural data. This ensures temporal alignment.

ii.
```python
bin_left = BIN_EDGES_S[:-1][None, :]
bin_right = BIN_EDGES_S[1:][None, :]
# ...
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
```

iii. The same `BIN_EDGES_S` array is used for both neural binning and photostim binning, ensuring alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `BehavioralEvents/left_lick_times/timestamps` and `BehavioralEvents/right_lick_times/timestamps` (actual lick event times), with `intervals/trials/trial_instruction` as a fallback for trials with no licks.

ii.
```python
left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)
instructions = decode_array(trial_group["trial_instruction"][:])[trial_indices].astype(str)
```

iii. The AI uses actual lick behavior to determine choice rather than the instructed trial type. This is appropriate for a "choice" variable that reflects the animal's decision.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Three-tier choice determination:
1. First post-go-cue lick: if licks occur after the go cue, the side of the first lick determines choice.
2. First any-trial lick: if no post-go licks but there are licks elsewhere in the trial, the first lick's side is used.
3. Instruction fallback: if no licks at all, the instructed trial side is used.

Left = 0, right = 1.

ii.
```python
left_post = left_all[left_all >= go_time]
right_post = right_all[right_all >= go_time]

if left_post.size or right_post.size:
    left_first = left_post[0] if left_post.size else np.inf
    right_first = right_post[0] if right_post.size else np.inf
    choice[trial_idx] = 0 if left_first < right_first else 1
    source_counter["post_go_lick"] += 1
    continue

if left_all.size or right_all.size:
    # ... fallback to any trial lick
    continue

choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
source_counter["instruction_fallback"] += 1
```

iii. From CONVERSION_NOTES.md: "79,585 trials using the preferred first post-go lick rule, 1,063 trials using first-lick-anywhere fallback, and 12,662 no-lick trials requiring instructed-side fallback."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` in the NWB trials table, which contains string values: "hit", "miss", or "ignore".

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```

iii. The instructions specify: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct string-to-integer mapping: "ignore" -> 0, "miss" -> 1, "hit" -> 2. The outcome is a per-trial categorical value broadcast across all time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
# In output construction:
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
```

iii. Straightforward mapping matching the instructions. The value is constant across time bins within a trial.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. There is no "Distance to reward zone" output in this task. The outcome variable is a per-trial scalar (not time-varying) that is broadcast to all time bins. No temporal alignment is needed beyond assigning the trial's outcome to each bin.

ii.
```python
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
```

iii. The outcome is constant within each trial, so alignment is trivial - each time bin gets the same value.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` in the NWB trials table, which contains string values: "early" or "no early".

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
```

iii. Directly read from the NWB trial metadata.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String-to-integer mapping: "no early" -> 0, "early" -> 1. Per-trial value broadcast across all time bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
# In output:
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
```

iii. Instructions specify: "Early lick (no = 0, yes = 1, per-trial)."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 = y-coordinate) and the corresponding `timestamps`.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. The reference code uses `camera_0_side` -> `tongue_y` from the .mat files; the NWB equivalent is the TongueTracking timeseries, column 1.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session-wide 40th and 60th percentiles are computed over all tongue y-position samples in the session (not just within-trial data). These percentiles define the discretization thresholds.

ii.
```python
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
```

iii. Instructions specify "per-session discretization" using 40th and 60th percentiles.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories:
- 0: y < 40th percentile
- 1: 40th percentile <= y <= 60th percentile (default)
- 2: y > 60th percentile

ii.
```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)  # default = 1
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. Instructions specify: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile." The AI uses strict inequalities (< and >) so values exactly at the 40th or 60th percentile remain in category 1.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is binned to the same 50ms bins as neural data (aligned to go cue). For each bin, the last available tongue tracking sample before the bin's right edge is used. If a bin has no samples, the nearest prior sample is used as a fallback.

ii.
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
```

iii. From CONVERSION_NOTES.md: "Binned values use the last frame inside each 50 ms bin, matching the reference marker-alignment style." The reference `align_markers_between_lims` also uses the last frame within each time window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data scenarios are handled:
- **Sessions with zero good units**: Skipped entirely (1 session excluded).
- **No-lick trials**: Choice falls back to instruction side for trials with no lick events.
- **Missing tongue frames**: If a 50ms bin has no tongue tracking frames, the last available frame before the bin is used (`np.clip(right_idx - 1, 0, ...)`).
- **Missing photostim fields**: Trials where onset or duration is "N/A" or empty get all-zero photostim.
- **obs_intervals mismatch**: Errors are raised if ephys trials cannot be matched to behavioral trials.

ii.
```python
# Missing photostim:
if onset is None or duration is None:
    continue  # leaves zeros

# No-lick fallback:
choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1

# Missing tongue frames:
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)

# Zero good units:
if n_good_units == 0:
    return None
```

iii. From CONVERSION_NOTES.md: "Tongue binning required previous-frame fallback in 149,144 bins across the full dataset." The AI tracked and reported fallback counts as part of sanity checking.

## 10-a. What are the most time-consuming steps of the code?

i. The two most time-consuming steps are:
1. **`bin_spike_counts_for_good_units`**: Iterates over ALL units (including non-good) to find good ones, then processes each good unit's spikes with numpy operations. This is called once per session.
2. **NWB file I/O**: Reading the full `units/spike_times` array (all units, all spikes) into memory for each session.

ii.
```python
# Loads ALL spike times:
spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),

# Iterates per unit:
for unit_idx, unit_end in enumerate(spike_times_index):
    spikes = spike_times_flat[unit_start:unit_end]
    if not good_unit_mask[unit_idx]:
        continue
    # ... process good unit
```

iii. The per-unit loop in spike binning is the computational bottleneck, processing potentially hundreds of units per session across 174 NWB files.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three main loops could be vectorized:
1. **`compute_choice_labels`**: Per-trial Python loop (lines 100-127) searching for licks in each trial. Could use vectorized `searchsorted` and array operations.
2. **`bin_photostim_series`**: Per-trial Python loop (lines 144-153) computing bin overlaps. Already has vectorized bin overlap logic but wraps it in a per-trial loop.
3. **`bin_spike_counts_for_good_units`**: Per-unit loop (lines 197-222). The inner operations are vectorized (numpy), but the outer unit loop is Python.

ii.
```python
# compute_choice_labels - per-trial loop:
for trial_idx in range(len(trial_start)):
    left_all = left_lick_times[(left_lick_times >= start) & (left_lick_times <= stop)]
    # ...

# bin_photostim_series - per-trial loop:
for trial_idx in range(n_trials):
    onset = as_float_or_none(onset_values[trial_idx])
    # ...
```

iii. No justification given for using loops vs. vectorized operations. The choice and photostim loops process ~93K trials total, which adds noticeable overhead.

## 10-c. What processing does the code repeat multiple times?

i.
1. **`build_dataset` called twice**: Once for the full dataset and once for the sample dataset. The sample reuses already-computed session results, so only the aggregation is repeated (lightweight).
2. **Reading all spike times**: The code reads `units/spike_times[:]` (all units) but only processes good units. The non-good units' spike times are loaded but discarded.
3. **Decoding byte strings**: `decode_array` is called multiple times on different NWB fields, each iterating through all values.

ii.
```python
# build_dataset called twice:
full_data, full_summary = build_dataset(session_results)
sample_data, sample_summary = build_dataset(sample_results)
```

iii. The double `build_dataset` call is intentional (sample vs. full), and the overhead is minimal since session conversion (the expensive part) is not repeated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. **Firing rate conversion**: Spike counts are divided by bin_size to produce firing rates (`* (1.0 / BIN_SIZE_S)`). Many decoders work with spike counts directly; this division is a linear scaling that doesn't affect decoding.
2. **Loading all units' spike times**: The entire `units/spike_times` array is read, but only good units are used. Could read only good units' spike time slices.
3. **Sample data creation**: The `sample_data.pkl` file duplicates a subset of the full dataset. If only the full dataset is needed, this processing is wasted.
4. **float16 precision for neural data**: Converting to float16 (`spike_counts[trial_idx].T.astype(np.float16)`) may discard precision unnecessarily.

ii.
```python
# Firing rate conversion (linear scaling):
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))

# Loading all spike times including non-good units:
spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
```

iii. The firing rate conversion preserves interpretability (spikes/s vs raw counts) but doesn't affect decoder performance. The float16 conversion saves memory but may introduce quantization noise for units with very high firing rates.
