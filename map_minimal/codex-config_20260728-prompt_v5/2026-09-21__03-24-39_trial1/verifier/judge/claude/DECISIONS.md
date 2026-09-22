# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All session files are found with a glob pattern (`/app/data/sub-*/*.nwb`) and sorted. Each file is opened with `h5py` (not pynwb) and processed sequentially. Subject IDs, trials, units, spike times, behavioral events, and tongue tracking are all read from the HDF5 group structure within each file.

ii.
```python
DATA_GLOB = "/app/data/sub-*/*.nwb"
...
for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
```

Loading one session:
```python
with h5py.File(path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"])
    classification = decode_array(f["units/classification"])
    spike_times_flat = f["units/spike_times"][()]
    spike_times_index = f["units/spike_times_index"][()]
    trials = f["intervals/trials"]
    go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

iii. The agent initially explored the data using pynwb but encountered difficulties with electrode metadata access. It switched to h5py for direct HDF5 array access, finding it simpler and more reliable for the flat HDF5 structure of these NWB files.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `general/subject/subject_id`, a numeric string such as `'440956'`. The subject is read for every session and tracked in a dictionary mapping subject IDs to indices. Subjects are added to the list in the order they are first encountered (not sorted).

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"])
...
subject_to_index = {}
subject_idx = []
...
if subject not in subject_to_index:
    subject_to_index[subject] = len(data["subjects"])
    data["subjects"].append(subject)
subject_idx.append(subject_to_index[subject])
```

iii. The agent used the numeric subject_id from the NWB files directly, consistent with the file structure.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no grouping or splitting is needed. Session order follows the sorted file list. Session metadata (filename, subject, trial counts, neuron counts) is stored per session.

ii.
```python
for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
```

```python
session_metadata = {
    "file": os.path.basename(path),
    "subject": subject,
    "n_trials_original": n_trials,
    "n_trials_kept": int(len(kept_trial_indices)),
    "n_neurons": int(keep_units.size),
    ...
}
```

iii. The agent recognized that the dandiset stores one session per file, so file boundaries define session boundaries.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. Go-cue times are loaded from `BehavioralEvents/go_start_times/timestamps`. The number of trials is determined by `len(trials["id"])`.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
trial_start = trials["start_time"][()]
...
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
```

iii. Trials are defined by the NWB trials table. No assertion is made to verify that the number of go-cue times matches the number of trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages. First, at the **session level**, sessions are dropped if they don't meet behavioral performance criteria: control-trial performance > 65%, and at least 50 correct left and 50 correct right control trials. Control trials are defined as non-photostim, non-early-lick, instructed (left/right) trials. Second, at the **trial level**, trials where all kept ALM units have zero spikes across all time bins are removed.

ii. Session-level filter:
```python
def session_behavior_metrics(outcome, trial_instruction, early_lick, photostim_onset):
    is_control = photostim_onset == "N/A"
    no_early = early_lick == "no early"
    instructed = np.isin(trial_instruction, ["left", "right"])
    behavior_trials = no_early & is_control & instructed & np.isin(outcome, ["hit", "miss", "ignore"])
    ...
    return {
        ...
        "passes_filter": (
            n_control_trials > 0
            and performance > MIN_CONTROL_PERFORMANCE
            and left_correct >= MIN_CORRECT_PER_SIDE
            and right_correct >= MIN_CORRECT_PER_SIDE
        ),
    }
```

Trial-level filter:
```python
keep_trials = np.any(rates != 0, axis=(0, 2))
```

iii. The agent found the session-level behavioral criteria in the methods text: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." The trial-level zero-activity filter was added after the validator exposed trials with no spikes: "some exported trials have zero activity across every kept ALM unit. That's almost certainly post-recording behavioral tail rather than legitimate silence."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times of each unit) via `units/spike_times_index` (ragged array offsets). Only units with `classification == 'good'` AND brain region in `{left ALM, right ALM}` contribute. Go-cue times are used to place the bin edges.

ii.
```python
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
spike_starts = np.r_[0, spike_times_index[:-1]]
...
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
keep_units = np.flatnonzero(keep_mask)
...
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
```

iii. The agent identified `spike_times` as the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates. For each kept unit, spikes are assigned to trials and bins using a custom `assign_events_to_windows` function that uses `searchsorted` to find trial windows and then computes bin indices via floor division. Spike counts per bin are divided by bin width (0.05s) to get firing rates in Hz. The result is stored as `float16`.

ii.
```python
def assign_events_to_windows(event_times, window_starts, window_ends, bin_width, n_bins):
    counts = np.zeros((len(window_starts), n_bins), dtype=np.uint16)
    ...
    trial_idx = np.searchsorted(window_starts, event_times, side="right") - 1
    ...
    bin_idx = np.floor((event_times - window_starts[trial_idx]) / bin_width).astype(np.int64)
    ...
    np.add.at(counts, (trial_idx[valid], bin_idx[valid]), 1)
    return counts

rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
    rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)
```

iii. The agent converted spike counts to firing rates by dividing by bin width, consistent with the reference code's `sliding_histogram(..., rate=True)`. Using `float16` was chosen to manage memory given the large dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by two criteria: (1) `classification == 'good'` from the spike-sorting QC classifier, and (2) brain region must be in `{left ALM, right ALM}`. A session with no units surviving both filters is dropped. No additional firing rate or metric thresholds are applied.

ii.
```python
REGIONS_TO_KEEP = {"left ALM", "right ALM"}
...
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
keep_units = np.flatnonzero(keep_mask)
if keep_units.size == 0:
    return None
```

iii. The agent verified that `classification == 'good'` yields 69,453 units matching the paper's QC methodology. The region restriction to ALM was motivated by: (1) the reference code analyzes one brain area at a time, (2) ALM is the behaviorally central, photoinhibited region, and (3) keeping all regions would yield ~12 GB, too large for the downstream trainer.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go-cue times are on the same session-absolute clock. The bin window starts are computed as `go_times + BIN_EDGES[0]` and window ends as `go_times + BIN_EDGES[-1]`. Spikes are then assigned to bins relative to each trial's window start.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
...
counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
```

iii. Everything in the NWB file is timestamped on one global clock, so aligning to the go cue only requires adding the bin offsets to the go-cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Bin centers are computed using the reference code's `sliding_histogram` logic, yielding 81 bin centers from -2.5 to +1.5 at 0.05s stride. Bin edges are derived from centers +/- half the bin width. This produces 81 bins of 50ms width, rather than 80 bins.

ii.
```python
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05

def compute_bin_centers(begin_time, end_time, stride):
    span = (end_time - begin_time) / stride
    if np.allclose(span, math.floor(span) + 1):
        n_bins = math.floor(span) + 2
    else:
        n_bins = math.floor(span) + 1
    return begin_time + np.arange(n_bins) * stride

BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
```

With `BEGIN_TIME=-2.5`, `END_TIME=1.5`, `BIN_STRIDE=0.05`: span = 80.0, `allclose(80, 81)` is True, so `n_bins = 82`. Wait -- let me recheck: `span = 80`, `floor(span) = 80`, `floor(span)+1 = 81`. `allclose(80, 81)` is False, so `n_bins = floor(80)+1 = 81`. So bin centers are `[-2.5, -2.45, ..., 1.45, 1.5]` -- 81 bins.

iii. The agent replicated the reference code's bin-center computation logic from `preprocessing_DJ_2022Aug.py`. This produces bin centers that include both endpoints (-2.5 and 1.5), yielding 81 time bins rather than 80. No smoothing or additional rebinning is applied.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The input named `"time_from_tone_onset_s"` is derived solely from `BIN_CENTERS`, the go-cue-relative bin center array. It is NOT computed per-trial relative to the actual sample tone onset.

ii.
```python
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```

iii. The agent treated the go cue as the relevant "tone onset" for this input, stating: "The 'time from tone onset' input is encoded as time from the Go-cue tone, which is the unambiguous tone aligned by this task." The same BIN_CENTERS array is used for every trial, making this a constant input rather than trial-varying.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. No per-trial processing is involved. The BIN_CENTERS array (which is the same for every trial) is used directly as the first row of the input matrix for each trial.

ii.
```python
BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE).astype(np.float32)
...
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```

iii. Since the agent interpreted "tone onset" as the go-cue tone and data is already aligned to the go cue, the bin centers directly represent time from go-cue tone onset with no additional computation needed.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same BIN_CENTERS array defines both the neural binning grid and the time-from-tone input, so they are inherently aligned.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
...
input_trials.append(
    np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
)
```

iii. Both the neural data and the input use bin centers derived from the same `compute_bin_centers` function.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and the go-cue time used to place them on the trial's time axis.

ii.
```python
photostim_onset = decode_array(trials["photostim_onset"])
photostim_duration = decode_array(trials["photostim_duration"])
...
def parse_photostim_series(trial_start, photostim_onset, photostim_duration, go_time):
    ...
    onset_abs = trial_start + float(photostim_onset)
    offset_abs = onset_abs + float(photostim_duration)
```

iii. The agent examined the NWB trial table and found photostim_onset stored as strings relative to trial start, with 'N/A' for non-stimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is computed per trial. The onset is converted to absolute time (trial_start + onset), the offset is onset + duration, and both are re-expressed relative to the go cue. Bin centers falling within [onset, offset) are set to 1.0; all others remain 0.0.

ii.
```python
def parse_photostim_series(trial_start, photostim_onset, photostim_duration, go_time):
    stim = np.zeros(N_BINS, dtype=np.float32)
    if photostim_onset == "N/A":
        return stim
    onset_abs = trial_start + float(photostim_onset)
    offset_abs = onset_abs + float(photostim_duration)
    rel_on = onset_abs - go_time
    rel_off = offset_abs - go_time
    stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
    return stim
```

iii. The agent computed the photostim series per trial, converting onset/duration to go-cue-relative times and creating a binary mask.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, and bin membership is determined using BIN_CENTERS, which are the same bin centers used for the neural data.

ii.
```python
rel_on = onset_abs - go_time
rel_off = offset_abs - go_time
stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
```

iii. The same BIN_CENTERS array is used for both the neural data binning and the photostim input, ensuring alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from actual lick event times: `left_lick_times/timestamps` and `right_lick_times/timestamps` from `BehavioralEvents`, plus `go_start_times/timestamps` for the go cue. The first lick within 1.5s after the go cue determines the choice.

ii.
```python
left_licks = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()]
right_licks = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()]
...
def first_choice_after_go(left_licks, right_licks, go_time, response_window=1.5):
    left_start = np.searchsorted(left_licks, go_time, side="left")
    left_end = np.searchsorted(left_licks, go_time + response_window, side="left")
    right_start = np.searchsorted(right_licks, go_time, side="left")
    right_end = np.searchsorted(right_licks, go_time + response_window, side="left")
    first_left = left_licks[left_start] if left_start < left_end else np.inf
    first_right = right_licks[right_start] if right_start < right_end else np.inf
    if np.isfinite(first_left) and first_left < first_right:
        return 0
    if np.isfinite(first_right):
        return 1
    return 2
```

iii. The agent investigated the lick-choice relationship by examining NWB event data for miss and ignore trials. It confirmed that miss trials had licks on the wrong side and ignore trials had no licks, validating the approach of using actual lick times rather than deriving from instruction x outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left and right lick times within the 1.5s response window after the go cue are found. The earlier lick determines the choice: left (0), right (1), or no lick (2) if neither side was licked. The value is repeated across all time bins.

ii.
```python
choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
...
output_trials.append(
    np.vstack([
        np.full(N_BINS, choice, dtype=np.int64),
        ...
    ])
)
```

iii. The agent verified that this derivation agrees with the instruction x outcome derivation on hit and miss trials, while being more direct.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds the strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = decode_array(trials["outcome"])
...
outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
```

iii. The trials table stores the outcome explicitly with the three categories the instructions ask for.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. The value is repeated across all time bins.

ii.
```python
outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
...
np.full(N_BINS, outcome_label, dtype=np.int64),
```

iii. Direct mapping with no additional processing.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds the strings `'no early'` and `'early'`.

ii.
```python
early = decode_array(trials["early_lick"])
...
early_label = {"no early": 0, "early": 1}[early[trial_idx]]
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to integers: no early=0, early=1. The value is repeated across all time bins.

ii.
```python
early_label = {"no early": 0, "early": 1}[early[trial_idx]]
...
np.full(N_BINS, early_label, dtype=np.int64),
```

iii. Direct mapping with no additional processing.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = tongue_x, tongue_y, tongue_likelihood, with matching `timestamps`. Column 1 (tongue_y) is the value; column 2 (likelihood) decides whether the tongue is visible.

ii.
```python
track = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"])
track_times = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"]
)
tongue_bins, p40, p60 = build_tongue_bins(track_times, track[:, 1], track[:, 2], go_times)
```

iii. The agent identified this as the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.9 are excluded. The 40th and 60th percentiles are computed over **all visible raw frames** in the session (not over bin means). Each frame within a trial window is assigned to a bin, and classified as 0 (<40th), 1 (40th-60th), or 2 (>60th). Bins with no visible frames remain class 3 ("not visible"). When multiple visible frames fall in the same bin, the **last frame's class overwrites** previous ones (last-write-wins via flat indexing).

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
visible = track_prob > VISIBILITY_THRESHOLD
y_visible = track_y[visible]
p40, p60 = np.percentile(y_visible, [40, 60])
...
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2

flat = tongue_bins.reshape(-1)
flat[trial_idx * N_BINS + bin_idx] = classes
```

iii. The agent used a visibility threshold of 0.9 (noting the bimodal distribution of likelihoods). Percentiles are computed over raw visible frames rather than bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Individual visible frames are classified: y < p40 -> 0, p40 <= y <= p60 -> 1, y > p60 -> 2. When multiple frames fall in one bin, the last frame's class wins. Bins with no visible frames -> 3.

ii.
```python
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2
```

iii. The thresholding uses strict inequalities: `< p40` for class 0, `> p60` for class 2, with class 1 for values exactly at the boundary. This matches the instructions' specification of < 40th, 40th-60th, > 60th percentiles.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues. Each trial's frame range is found by comparing timestamps to `go_times + BIN_EDGES[0]` (window start) and `go_times + BIN_EDGES[-1]` (window end). Frames are assigned to bins based on their offset from the window start divided by bin width.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
...
bin_idx = np.floor((times - window_starts[trial_idx]) / BIN_WIDTH).astype(np.int64)
```

iii. The same bin grid (derived from BIN_EDGES and BIN_WIDTH) is used for both neural and tongue data, ensuring alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **NaN classification values**: The `decode_array` function converts bytes/NaN to strings; units without valid `classification == 'good'` are excluded.
- **Sessions failing behavioral criteria**: Dropped if performance <= 0.65 or fewer than 50 correct trials per side.
- **Trials with zero spikes**: All-zero trials (across all kept ALM units and all time bins) are removed after spike binning.
- **Frames with low tongue visibility**: Frames with likelihood <= 0.9 are excluded; bins with no visible frames get class 3 ("not visible").
- **N/A photostim**: Handled by string comparison; returns all-zero stimulus series.

ii.
```python
# NaN/bytes handling
def decode_array(dataset):
    arr = np.asarray(dataset)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    ...

# Zero-spike trial removal
keep_trials = np.any(rates != 0, axis=(0, 2))

# Tongue visibility
visible = track_prob > VISIBILITY_THRESHOLD
```

iii. The agent handled missing data by excluding it at multiple levels: session-level behavioral filters, unit-level QC and region filters, trial-level zero-spike filters, and frame-level tongue visibility filters.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with h5py and loading the full spike_times array dominates. For each session, the spike times buffer and tongue tracking arrays must be loaded into memory. The per-unit loop over `assign_events_to_windows` adds cost proportional to the number of units. Pickling the output dataset is also significant.

ii.
```python
spike_times_flat = f["units/spike_times"][()]
...
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
```

iii. The agent noted that using every good unit from every trial would create ~12 GB of data, which motivated the ALM-only restriction to keep processing tractable.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in neural binning iterates over each kept unit, calling `assign_events_to_windows` separately for each unit's spike train. The per-trial loop in `first_choice_after_go` is called once per trial rather than vectorized. The `parse_photostim_series` function is also called per trial in a loop.

ii.
```python
for out_idx, unit_idx in enumerate(keep_units):
    spikes = spike_times_flat[spike_starts[unit_idx] : spike_times_index[unit_idx]]
    counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
    rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)

for trial_idx in kept_trial_indices:
    stim_series = parse_photostim_series(...)
    choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
```

iii. The per-unit loop is necessary because each unit has a different number of spikes (ragged storage). The per-trial loops for choice and photostim could potentially be vectorized but are not the bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The `assign_events_to_windows` function is called once per unit, each time performing trial assignment via `searchsorted` on the full spike array slice. The `window_starts` and `window_ends` are computed once and reused. There is no redundant recomputation of major data structures.

ii. N/A

iii. No significant repeated processing was identified.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `session_behavior_metrics` including detailed control-trial statistics (left/right correct counts, performance ratio) which are stored in session metadata but not used by the decoder. Brain region labels are derived from electrode locations via JSON parsing, which is more complex than needed. The `tongue_visible_y_40` and `tongue_visible_y_60` percentile values are stored in session metadata but not used downstream.

ii.
```python
session_metadata = {
    ...
    "tongue_visible_y_40": p40,
    "tongue_visible_y_60": p60,
    **behavior_metrics,
}
```

iii. These extra metadata fields provide transparency about the processing but are not consumed by the decoder.
