# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with a glob pattern `sub-*/*.nwb`. The AI uses `h5py` directly (not `pynwb`) to open each file and access the HDF5 groups for units, trials, and acquisition data. A two-pass approach is used: first `scan_vocab` enumerates subjects and brain regions, then `convert_session` processes each file.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
subjects, brain_regions, paths = scan_vocab(paths)
...
for session, path in enumerate(paths, start=1):
    converted = convert_session(path, subject_to_idx, region_to_idx)
```

Loading one session:
```python
with h5py.File(path, "r") as nwb:
    trials = nwb["intervals/trials"]
    events = nwb["acquisition/BehavioralEvents"]
    units = nwb["units"]
```

iii. The agent confirmed both h5py and pynwb were available, but chose h5py because "the archived files use an older NWB schema, while all fields needed here are standard HDF5 datasets." The glob pattern follows the DANDI directory structure discovered by the agent during exploration.

## 1-b. How are the data split into subjects?

i. Each NWB file records its subject in `general/subject/subject_id`. The `scan_vocab` pass collects all unique subject IDs into a sorted list. During assembly, `subject_to_idx` maps each subject to its index.

ii.
```python
subjects.add(
    nwb["general/subject/subject_id"][()].decode("utf-8").strip()
)
...
subject_to_idx = {name: idx for idx, name in enumerate(subjects)}
```

iii. The subject ID is the canonical animal identifier in the NWB file. The agent found 28 unique subjects across 174 files.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. No grouping or splitting is needed. Each session is identified by `nwb["identifier"]`. Session order follows the sorted file list.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
identifier = nwb["identifier"][()].decode("utf-8").strip()
```

iii. The DANDI dataset stores one session per file, so the file boundary is the session boundary. 173 of 174 files reach the output (one excluded for having no classifier-approved units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The number of behavioral trials is validated against the number of go-cue events. The AI then restricts to ephys-covered trials using `is_good_trials.shape[1]`.

ii.
```python
n_behavior_trials = len(trials["id"])
all_go_times = events["go_start_times/timestamps"][:]
if len(all_go_times) != n_behavior_trials:
    raise ValueError(...)

n_ephys_trials = units["is_good_trials"].shape[1]
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
go_times = all_go_times[source_trial_idx]
```

iii. The agent discovered that some NWB files have more behavioral trials than ephys trials, and used `is_good_trials.shape[1]` to determine the number of ephys-covered trials, taking only the first `n_ephys_trials` trials.

## 1-e. How are trials filtered based on quality controls?

i. Two-stage filtering: (1) Trials beyond the ephys recording are excluded by using `is_good_trials.shape[1]` to determine the number of ephys-covered trials. (2) Acquisition-gap trials where all good units fire zero spikes are excluded. Free water trials are NOT filtered out. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
n_ephys_trials = units["is_good_trials"].shape[1]
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)
...
has_neural_data = np.any(rates != 0, axis=(0, 2))
source_trial_idx = source_trial_idx[has_neural_data]
go_times = go_times[has_neural_data]
rates = rates[:, has_neural_data, :]
```

```python
if len(n) < 2:
    print(f"Skipping {path.name}: fewer than two trials", flush=True)
    continue
```

iii. The agent reasoned that early-lick, ignore, and photostimulation trials should be retained because they are explicitly requested decoder variables. Free water trials were not specifically filtered. The agent initially included all behavioral trials, but after the verifier flagged zero-neural warnings, corrected by implementing the `is_good_trials`-based coverage check and zero-neural removal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, the sorted spike times of each unit. Only units with `classification == 'good'` contribute. Go-cue times (`BehavioralEvents/go_start_times`) are used to place the bin edges.

ii.
```python
all_spikes = units["spike_times"]
ends = units["spike_times_index"][:].astype(np.int64, copy=False)
starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
```

iii. `spike_times` is the only neural representation in the NWB files, so firing rates are computed from it directly.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, spikes are assigned to trial windows via `searchsorted`, then binned into 50ms bins. The rate is computed by adding `1.0 / BIN_SIZE_S` per spike (equivalent to count / bin_width). No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_good_units(units, good_indices, go_times):
    rates = np.zeros((n_units, n_trials, N_TIME), dtype=np.float32)
    ...
    for output_unit, source_unit in enumerate(good_indices):
        spikes = all_spikes[starts[source_unit]:ends[source_unit]]
        trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
        ...
        bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
        np.add.at(
            rates[output_unit],
            (candidate_trials[valid_bin], bin_idx[valid_bin]),
            np.float32(1.0 / BIN_SIZE_S),
        )
    return rates
```

iii. The agent noted "firing rate is spike count divided by bin width" which matches the reference code approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. No thresholds are applied to individual quality metrics. A session with no good units is dropped entirely. This retains 69,453 of ~272,000 units.

ii.
```python
classification = decode_strings(units["classification"][:])
good_indices = np.flatnonzero(classification == "good")
```

In `scan_vocab`:
```python
good = classification == "good"
if np.count_nonzero(good) == 0:
    print(f"Skipping {path.name}: no classifier-approved units", flush=True)
    continue
```

iii. The agent reasoned: "The NWB export carries the paper's classifier result directly (`units/classification == 'good'`), so no quality model needs to be recreated."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and event times are on the same session-absolute clock. Bin edges are computed as go_time + relative offsets. Each spike is assigned to a trial window by searching for the correct go-cue interval, then its relative time determines the bin.

ii.
```python
window_starts = go_times + OFF_START_S
window_ends = go_times + OFF_END_S
...
trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
...
relative = candidate_spikes - go_times[candidate_trials]
bin_idx = np.searchsorted(BIN_EDGES, relative, side="right") - 1
```

iii. Everything in the NWB file is timestamped on one global clock, so aligning to the go cue only requires looking up each trial's go-cue time and taking the window around it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning -2.5s to +1.5s relative to the go cue. The bin grid is defined once and reused for every trial and session.

ii.
```python
BIN_SIZE_S = 0.050
OFF_START_S = -2.5
OFF_END_S = 1.5
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES = OFF_START_S + np.arange(N_TIME + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

iii. The window and 50ms bin width are set by the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset timestamps) and go-cue times. The tone taken for a trial is the last `sample_start_times` entry before the go cue, constrained to be within the trial's start time.

ii.
```python
sample_starts = events["sample_start_times/timestamps"][:]
tone_onsets = final_tone_onsets(trial_starts, go_times, sample_starts)
```

```python
def final_tone_onsets(trial_starts, go_times, sample_starts):
    for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
        stop_idx = np.searchsorted(sample_starts, go, side="right")
        start_idx = np.searchsorted(sample_starts, start, side="left")
        result[trial] = sample_starts[stop_idx - 1]
    return result
```

iii. An early lick replays the sample epoch, so a trial can have more than one tone; the last one before the go cue is selected.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time-from-tone is computed as: bin centers (relative to go cue) plus the go-to-tone gap. This gives a continuous time-varying signal.

ii.
```python
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. No additional processing beyond finding the tone onset time and computing time differences.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The input uses the same bin centers as the neural data (both defined relative to the go cue), so alignment is automatic.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
...
time_from_tone = (
    BIN_CENTERS[None, :] + (go_times - tone_onsets)[:, None]
).astype(np.float32)
```

iii. The bin grid is shared between neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` to compute absolute photostim start/end times.

ii.
```python
photo_onset = parse_optional_floats(trials["photostim_onset"][:][source_trial_idx])
photo_duration = parse_optional_floats(trials["photostim_duration"][:][source_trial_idx])
absolute_photo_start = trial_starts + photo_onset
absolute_photo_end = absolute_photo_start + photo_duration
```

iii. The photostim onset values are stored as strings relative to trial start, with 'N/A' on non-stimulated trials, so they are converted and mapped to absolute times.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 where its center falls between the absolute onset and offset of stimulation, and 0 elsewhere. The result is a binary time-varying input.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
photo_on = (
    (absolute_centers >= absolute_photo_start[:, None])
    & (absolute_centers < absolute_photo_end[:, None])
    & np.isfinite(absolute_photo_start[:, None])
).astype(np.float32)
```

iii. Non-stimulated trials have NaN onset, and the `np.isfinite` check ensures those bins remain 0.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The absolute bin centers (`go_times + BIN_CENTERS`) are used for both neural binning and photostim comparison, ensuring alignment.

ii.
```python
absolute_centers = go_times[:, None] + BIN_CENTERS[None, :]
```

iii. The comparison is done in absolute session time, which shares the same clock as the spikes.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the actual lick event timestamps: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. The first lick after the go cue within a 1.5s response window determines the choice.

ii.
```python
left_licks = events["left_lick_times/timestamps"][:]
right_licks = events["right_lick_times/timestamps"][:]
choice = event_choice(go_times, left_licks, right_licks)
```

```python
def event_choice(go_times, left_licks, right_licks):
    choice = np.full(len(go_times), 2, dtype=np.int8)
    for trial, go in enumerate(go_times):
        li = np.searchsorted(left_licks, go, side="left")
        ri = np.searchsorted(right_licks, go, side="left")
        left_time = left_licks[li] if li < len(left_licks) else np.inf
        right_time = right_licks[ri] if ri < len(right_licks) else np.inf
        response_end = go + 1.5
        if left_time < response_end and left_time <= right_time:
            choice[trial] = 0
        elif right_time < response_end:
            choice[trial] = 1
    return choice
```

iii. The agent used direct behavioral measurements (lick timestamps) rather than deriving choice from instruction x outcome. The three classes are: left=0, right=1, no lick=2.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left and right lick after the go cue are found. If the earliest lick is within 1.5s of the go cue, that direction is the choice. If no lick occurs within the window, choice is "no lick" (2). The result is per-trial and tiled across all 80 time bins.

ii.
```python
output_trial[0, :] = choice[trial]
```

iii. The agent chose to use actual lick events rather than inferring from instruction x outcome.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, holding strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_text = decode_strings(trials["outcome"][:][source_trial_idx])
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int8)
```

iii. The trials table stores outcome explicitly with exactly the three categories the instructions require.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to 0 (ignore), 1 (miss), 2 (hit) via a dictionary. The result is per-trial and tiled across all 80 time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int8)
...
output_trial[1, :] = outcome[trial]
```

iii. Direct mapping, no additional processing.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds strings `'no early'` and `'early'`.

ii.
```python
early_text = decode_strings(trials["early_lick"][:][source_trial_idx])
early = (early_text == "early").astype(np.int8)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to 0 (no) and 1 (yes). The result is per-trial and tiled across all 80 time bins.

ii.
```python
early = (early_text == "early").astype(np.int8)
...
output_trial[2, :] = early[trial]
```

iii. Direct boolean mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = tongue_x, tongue_y, tongue_likelihood, with matching `timestamps`. Column 1 (tongue_y) is the value; column 2 (likelihood) determines visibility.

ii.
```python
tongue = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue["data"][:]
tongue_times = tongue["timestamps"][:]
tongue_y, visible, n_outliers = clean_tongue_y(tongue_data[:, 1], tongue_data[:, 2])
```

iii. This is the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps: (1) Frames with likelihood < 0.9 are marked as not visible. (2) Five-sigma velocity outliers among visible frames are interpolated from neighboring good frames. (3) Session-wide 40th and 60th percentiles of visible y-values give class edges. Each bin is discretized: 0 (< 40th), 1 (40th-60th inclusive), 2 (> 60th), 3 (not visible).

ii.
```python
TONGUE_LIKELIHOOD_CUTOFF = 0.9

def clean_tongue_y(y, likelihood):
    visible = np.isfinite(clean) & np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_CUTOFF)
    ...
    cutoff = speed_sample.mean() + 5.0 * speed_sample.std()
    outlier[1:] = consecutive & (speeds > cutoff)
    ...
    clean[outlier] = np.interp(x[outlier], x[good], clean[good])
    return clean, visible, int(np.count_nonzero(outlier))

q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
```

iii. The agent noted "0.9 is a conservative standard p-cutoff and cleanly separates tracked from occluded tongue samples." The velocity outlier cleaning follows the method paper's preprocessing approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles are computed over visible raw frames (after velocity cleaning). Values below 40th percentile = 0, between 40th and 60th (inclusive both ends) = 1, above 60th = 2, not visible = 3.

ii.
```python
q40, q60 = np.percentile(tongue_y[visible], [40.0, 60.0])
...
tongue_class = np.full((n_trials, N_TIME), 3, dtype=np.int8)
tongue_class[sampled_visible & (sampled_y < q40)] = 0
tongue_class[sampled_visible & (sampled_y >= q40) & (sampled_y <= q60)] = 1
tongue_class[sampled_visible & (sampled_y > q60)] = 2
```

iii. The 40/60 split and per-session scope follow the instructions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Nearest-neighbor sampling: for each bin center time (absolute), the nearest video frame is found using `nearest_indices`. The tongue y-value and visibility at that frame are used for the bin.

ii.
```python
video_idx = nearest_indices(tongue_times, absolute_centers.ravel()).reshape(n_trials, N_TIME)
sampled_y = tongue_y[video_idx]
sampled_visible = visible[video_idx]
```

```python
def nearest_indices(sorted_times, query_times):
    right = np.searchsorted(sorted_times, query_times, side="left")
    right = np.clip(right, 0, len(sorted_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(query_times - sorted_times[left]) <= np.abs(sorted_times[right] - query_times)
    return np.where(choose_left, left, right)
```

iii. The camera timestamps share the global session clock, so the nearest frame to each bin center provides a direct mapping without interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases:
- **Session with no good units**: Dropped entirely during `scan_vocab`.
- **Behavioral-only trials beyond ephys**: Excluded via `is_good_trials.shape[1]`.
- **Acquisition gaps (zero neural data)**: Excluded via `np.any(rates != 0, axis=(0, 2))`.
- **Low-likelihood tongue frames**: Marked not visible (class 3), not imputed.
- **Photostim N/A**: Mapped to NaN, producing all-zero photostim for those trials.

ii.
```python
# Session with no good units
if np.count_nonzero(good) == 0:
    print(f"Skipping {path.name}: no classifier-approved units", flush=True)
    continue

# Behavioral-only trials
n_ephys_trials = units["is_good_trials"].shape[1]
source_trial_idx = np.arange(n_ephys_trials, dtype=np.int64)

# Zero neural data
has_neural_data = np.any(rates != 0, axis=(0, 2))
source_trial_idx = source_trial_idx[has_neural_data]
```

iii. Missing data is either excluded (when nothing was recorded) or represented as an explicit category (tongue not visible).

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and binning spike times dominate. The `bin_good_units` function reads the full spike times array and performs per-unit vectorized binning. The two-pass architecture (scan_vocab + convert_session) means each file is opened twice. Pickling the ~11 GB result is also significant.

ii.
```python
all_spikes = units["spike_times"]
ends = units["spike_times_index"][:].astype(np.int64, copy=False)
...
for output_unit, source_unit in enumerate(good_indices):
    spikes = all_spikes[starts[source_unit]:ends[source_unit]]
```

iii. The work is dominated by I/O and spike binning.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops remain: (1) Per-unit loop in `bin_good_units` processes one unit at a time. (2) Per-trial loop in `event_choice` computes choice one trial at a time. (3) Per-trial loop in `final_tone_onsets` computes tone onset one trial at a time.

ii.
```python
for output_unit, source_unit in enumerate(good_indices):
    spikes = all_spikes[starts[source_unit]:ends[source_unit]]
    ...

for trial, go in enumerate(go_times):
    li = np.searchsorted(left_licks, go, side="left")
    ...

for trial, (start, go) in enumerate(zip(trial_starts, go_times)):
    stop_idx = np.searchsorted(sample_starts, go, side="right")
    ...
```

iii. The per-unit loop is inherent to ragged spike storage. The per-trial loops for choice and tone onset could potentially be vectorized but are not the bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The two-pass architecture opens each NWB file twice: once in `scan_vocab` (to build subject/region vocabularies) and once in `convert_session` (to process the data). The first pass reads `classification` and `anno_name`; the second pass reads them again.

ii.
```python
def scan_vocab(paths):
    for path in paths:
        with h5py.File(path, "r") as nwb:
            classification = decode_strings(nwb["units/classification"][:])
            ...

def convert_session(path, ...):
    with h5py.File(path, "r") as nwb:
        classification = decode_strings(units["classification"][:])
        ...
```

iii. The two-pass approach ensures stable vocabularies before conversion but doubles file I/O for the classification and annotation columns.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The velocity outlier cleaning of tongue y-position is additional processing not present in the reference solution. Depending on perspective, this could be considered unnecessary if the downstream decoder is robust to occasional outliers, or beneficial for cleaner signal. The detailed session_info metadata (choice_counts, outcome_counts, etc.) is computed but only used for logging/metadata.

ii.
```python
tongue_y, visible, n_outliers = clean_tongue_y(tongue_data[:, 1], tongue_data[:, 2])
...
"choice_counts": np.bincount(choice, minlength=3).astype(int).tolist(),
"outcome_counts": np.bincount(outcome, minlength=3).astype(int).tolist(),
```

iii. The velocity outlier cleaning follows the method paper but is not strictly required by the instructions.
