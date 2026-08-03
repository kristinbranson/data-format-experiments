# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (HDF5) files stored in a directory tree organized by subject. It scans `data_root` for subject subdirectories, collects all `.nwb` files (sorted alphabetically by subject then filename), and opens each one with `h5py.File`. Each NWB file corresponds to one session. The AI reads behavioral trial tables (`intervals/trials`), event timestamps (`acquisition/BehavioralEvents`), spike times (`units/spike_times`), unit metadata (`units/classification`, `units/anno_name`), observation intervals (`units/obs_intervals`), and tongue tracking data (`acquisition/BehavioralTimeSeries`).

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

iii. The agent identified that the data was stored as NWB files on the DANDI archive and organized in per-subject folders. It inspected the NWB structure using h5py and pynwb before writing the conversion.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by extracting the parent directory name from each NWB file path (e.g., `sub-440956`). A sorted unique list of subjects is built, and each session is mapped to its subject via `subject_idx`.

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))

subjects = sorted({session["subject"] for session in session_results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent noted 28 subjects across 174 NWB files, consistent with the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions with zero `classification == "good"` units are skipped. This results in 173 sessions from 174 NWB files (one session excluded).

ii.
```python
def convert_session(path: str) -> dict[str, Any] | None:
    with h5py.File(path, "r") as f:
        classifications = decode_array(f["units/classification"][:]).astype(str)
        good_unit_mask = classifications == "good"
        n_good_units = int(np.sum(good_unit_mask))
        if n_good_units == 0:
            return None
```

iii. The agent reasoned that since excluding the one zero-good-unit session yields 173 sessions (matching the paper), this is the correct session inclusion criterion. The agent did NOT apply the reference paper's behavioral performance criteria (>65% correct, >=50 correct left and right trials).

## 1-d. How are the data split into trials?

i. Trials are determined by matching the first good unit's `obs_intervals` (observation intervals from the electrophysiology) back to the behavioral trial table (`intervals/trials`). Only trials covered by the electrophysiology recording are kept.

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
```

iii. The agent discovered that some NWB files contain more behavioral trials than ephys-covered trials. It used `obs_intervals` to identify which behavior trials have corresponding neural recordings.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter trials by quality. All ephys-covered trials are retained, including early lick trials, no-response (ignore) trials, auto water trials, free water trials, and photostimulation trials. The reference code's `get_regular_trial_mask` function excludes early lick, auto water, free water, no-response, and stimulation trials, but the AI chose not to apply this filter.

ii. No trial filtering code exists in `convert_data.py`. All trials from `obs_intervals` are kept.

iii. The agent justified this by noting that the decoder task explicitly requires photostimulation as an input and early lick/outcome (including ignore) as outputs, so those trial types must be retained for the decoder to learn from them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (flat array of all spike timestamps) and `units/spike_times_index` (index array demarcating per-unit boundaries). Only units with `units/classification == "good"` are used.

ii.
```python
spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
good_unit_mask=good_unit_mask,
```

iii. The agent confirmed these are the standard NWB fields for spike time data.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins aligned to the go cue, spanning [-2.5s, +1.5s]. Spike counts per bin are computed, then converted to firing rates by dividing by the bin width (0.05s). The result is stored as float16.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)

bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)

# Convert to firing rates
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. The agent followed the instruction to use 50ms bins, diverging from the reference code's 40ms bins with 3.4ms stride. The conversion to firing rates matches the reference `sliding_histogram` approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. No additional unit-level quality filtering (e.g., variance-based filtering from `preprocessing_utils.py`) is applied.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
```

iii. The agent confirmed this matches the reference `qc_mode = 'classifier'` approach used in `preprocess_all_ephys.py`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue time is identified from `acquisition/BehavioralEvents/go_start_times/timestamps`. Spike times are expressed relative to the go cue, then binned into the [-2.5s, +1.5s] window.

ii.
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
```

iii. The instructions explicitly specify go cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (0.05s) non-overlapping bins. There are 80 time bins spanning [-2.5s, +1.5s]. No temporal rebinning is applied; spikes are binned directly at this resolution. The reference code uses 40ms bins with 3.4ms stride, but the instructions override this with 50ms bins.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
```

iii. The agent followed the instruction specification for 50ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (tone/sample onset times) and `acquisition/BehavioralEvents/go_start_times/timestamps` (go cue times).

ii.
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
```

iii. The agent identified `sample_start_times` as the tone onset event in the NWB files.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start_times` event before the go cue (but after trial start) is found. This is converted to a time relative to the go cue (`sample_rel`). Then for each time bin, the time from tone onset is computed as `bin_center - sample_rel`, giving a continuous time-varying signal.

ii.
```python
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. The agent used the `last_event_before_per_trial` function to handle the possibility of multiple sample events before a go cue (due to early lick replays).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same bin centers as the neural data (`BIN_CENTERS_S`), so alignment is exact. Both are relative to the go cue onset.

ii.
```python
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. Shared bin structure ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration` from the behavioral trial table.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```

iii. The agent read these fields from the NWB trial table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, if both `photostim_onset` and `photostim_duration` are non-null, the photostimulation interval is computed in absolute time, then converted to go-cue-relative time. A binary indicator (0/1) is set for each time bin that overlaps with the photostimulation interval.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. The agent interpreted `photostim_onset` as relative to trial start. The binary overlap check creates a time-varying binary input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Uses the same `BIN_EDGES_S` as neural data. The photostim on/off times are converted to go-cue-relative coordinates and compared against bin edges.

ii. Same bin edges as neural data are used in `bin_photostim_series`.

iii. Shared bin structure ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `acquisition/BehavioralEvents/left_lick_times/timestamps`, `acquisition/BehavioralEvents/right_lick_times/timestamps`, the go cue times, and `intervals/trials/trial_instruction` as a fallback.

ii.
```python
left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)
```

iii. The agent used lick event timestamps to determine which side the mouse licked first.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A three-tier fallback hierarchy: (1) First post-go-cue lick determines choice (left=0, right=1). (2) If no post-go lick, the first lick anywhere in the trial is used. (3) If no licks at all, the trial instruction side is used as a fallback. Left=0, Right=1.

ii.
```python
def compute_choice_labels(...):
    # Priority 1: first post-go lick
    if left_post.size or right_post.size:
        choice[trial_idx] = 0 if left_first < right_first else 1
    # Priority 2: any lick in trial
    elif left_all.size or right_all.size:
        choice[trial_idx] = 0 if left_first < right_first else 1
    # Priority 3: instruction fallback
    else:
        choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
```

iii. The agent noted 12,662 trials needed instruction fallback, 1,063 used any-trial-lick, and 79,585 used post-go lick. This handles the "ignore" (no-response) trials that were kept rather than filtered out.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` in the NWB trial table.

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```

iii. Direct mapping from the NWB trial table field.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple categorical mapping: "ignore" -> 0, "miss" -> 1, "hit" -> 2. This matches the instruction specification exactly.

ii. See code snippet above.

iii. No complex processing; direct label mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` in the NWB trial table.

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
```

iii. Direct mapping from the NWB trial table field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Simple categorical mapping: "no early" -> 0, "early" -> 1. Matches the instruction specification (no=0, yes=1).

ii. See code snippet above.

iii. No complex processing; direct label mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 for y-position) and its associated `timestamps`.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. The agent identified the side-camera tongue tracking data in the NWB file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The raw tongue y-position is discretized per session using the 40th and 60th percentiles computed over the entire session's tongue y trace. Values below the 40th percentile map to 0, between 40th-60th to 1, and above 60th to 2.

ii.
```python
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
```

iii. Matches the instruction specification for per-session discretization.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (< 40th percentile), 1 (40th to 60th percentile), 2 (> 60th percentile). Percentiles are computed session-wide over the raw tongue y trace (all time points, not just trial time points).

ii.
```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. The thresholding uses strict inequalities: values exactly at q40 or q60 are assigned to category 1 (middle).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is binned into the same 50ms time bins as neural data. For each bin, the last tongue tracking sample within the bin edges is used. If a bin has no samples, the most recent sample before the bin end is used as a fallback.

ii.
```python
def bin_tongue_y(...):
    trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
    left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
    right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
    sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
    y_binned = tongue_y[sample_idx]
```

iii. The agent noted this matches the reference `align_markers.py` style of using the last frame within each time interval.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Sessions with zero good units are skipped entirely. (2) Trials not covered by electrophysiology are excluded via `obs_intervals` matching. (3) For choice labels, a fallback hierarchy handles trials with no licks. (4) For tongue binning, bins without tongue tracking samples fall back to the nearest previous sample. (5) Missing/null photostim values result in all-zero photostim for that trial.

ii.
```python
# Photostim missing data
onset = as_float_or_none(onset_values[trial_idx])
duration = as_float_or_none(duration_values[trial_idx])
if onset is None or duration is None:
    continue  # leaves zeros

# Tongue fallback
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The agent documented fallback counts in CONVERSION_NOTES.md: 149,144 tongue bins needed fallback, 12,662 trials needed instruction-side choice fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The spike binning function `bin_spike_counts_for_good_units` is the most compute-intensive step, as it iterates over all units and performs per-unit spike assignment to trial-aligned bins. The NWB file I/O (reading large spike time arrays and tongue tracking data) is also significant.

ii.
```python
def bin_spike_counts_for_good_units(...):
    for unit_idx, unit_end in enumerate(spike_times_index):
        # Per-unit spike binning across all trials
```

iii. The spike binning involves vectorized operations per unit but iterates over units in a loop.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The `compute_choice_labels` function uses a Python for-loop over trials to determine lick choice, which could potentially be vectorized. (2) The `bin_photostim_series` function loops over trials. (3) The outer loop in `bin_spike_counts_for_good_units` iterates over units.

ii.
```python
# Trial-level loop in compute_choice_labels
for trial_idx in range(len(trial_start)):
    # per-trial lick analysis

# Trial-level loop in bin_photostim_series
for trial_idx in range(n_trials):
    onset = as_float_or_none(onset_values[trial_idx])
```

iii. These loops process one trial or unit at a time where vectorized searchsorted-style operations could batch the work.

## 10-c. What processing does the code repeat multiple times?

i. The code reads the full `spike_times` and `spike_times_index` arrays once per session, which is efficient. However, the per-session conversion logic (trial matching, event extraction) is repeated identically for every session in a sequential loop. No caching or parallel processing is used.

ii.
```python
for idx, path in enumerate(all_paths, start=1):
    result = convert_session(path)
```

iii. Each session is processed independently, which is correct but sequential.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code computes and stores detailed statistics (`stats` dict) per session including choice source counts, outcome counts, etc., which are used only for logging/metadata and not by the decoder. (2) The code builds a `sample_data.pkl` subset in addition to the full dataset. (3) The code converts spike counts to firing rates using float16, which may lose precision. (4) The tongue y-position percentiles are computed over the entire session's raw trace (including non-trial periods), which may include irrelevant time points.

ii.
```python
# Stats computed but not used by decoder
"stats": {
    "n_trials": int(n_trials),
    "n_good_units": int(len(brain_region_labels)),
    ...
}
```

iii. The stats are useful for validation but are not part of the decoder pipeline.
