# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `data/sub-*/sub-*.nwb` using `pynwb.NWBHDF5IO`. It iterates through 174 NWB files found via glob, processing each one individually in `process_session()`. Each NWB file is one session. Data is loaded sequentially (no parallel I/O).

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```
```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
```

iii. The agent noted that the reference code loads from .mat files, but since the raw data is in NWB format, it adapted accordingly. The glob pattern finds all 174 NWB files across 28 subject directories.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `nwb.subject.subject_id` from each NWB file. The AI collects all unique subject IDs across sessions and builds a sorted list. A `subject_to_idx` mapping is created for indexing.

ii.
```python
subject_id = nwb.subject.subject_id
# ...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
```

iii. The agent identified 28 unique subjects from the NWB data, matching the papers. Subject IDs are extracted per-session and de-duplicated.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes each file independently. Sessions that fail quality criteria are skipped (see 1-e). The final dataset contains 151 sessions (out of 174 NWB files).

ii.
```python
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
    if session_data is not None:
        all_sessions.append(session_data)
```

iii. The agent noted that the paper reports 173 sessions (one has 0 good units), but after applying behavioral filtering criteria (>65% correct rate, >=50 correct L/R trials), only 151 sessions remain.

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials` which contains all trial metadata, and go cue times from `nwb.acquisition['BehavioralEvents'].time_series['go_start_times'].timestamps[:]`. Each trial is indexed by position in the trials table, aligned via go cue timestamps.

ii.
```python
n_trials_total = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
# ...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. The agent found that go_start_times align 1:1 with the trials table. Each go_time corresponds to one trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of filtering:
1. **Session-level**: Sessions must have >=2 good units, >65% correct rate (hits/(hits+misses) on control no-early-lick trials), and >=50 correct left and >=50 correct right trials.
2. **Trial-level**: Trials are filtered by neural recording coverage - the go cue time plus the window [-2.5, +1.5]s must fall within the actual spike time range of the good units.

The AI does NOT apply the reference code's `get_regular_trial_mask` filtering (which excludes early-lick, auto-water, free-water, no-response, and stimulation trials), reasoning that early_lick, outcome, and photostim are decoder outputs/inputs.

ii.
```python
# Session selection criteria
is_control = np.array([str(p) == 'N/A' for p in photostim_power])
is_no_early = np.array([str(e) == 'no early' for e in early_lick])
control_no_early = is_control & is_no_early
is_not_ignore = outcome != 'ignore'
denom = np.sum(control_no_early & is_not_ignore)
hits = np.sum(control_no_early & (outcome == 'hit'))
correct_rate = hits / denom

if correct_rate < 0.65: ...skip...
if correct_left < 50 or correct_right < 50: ...skip...

# Trial-level: recording coverage
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
```

iii. The agent documented that session selection criteria (>65% correct, >=50 correct L/R) match the paper and reference code. The trial-level filtering was developed after discovering that some sessions have partial neural recording coverage (e.g., recording ends at 1107s but trials continue to 3707s). The decision to not apply `get_regular_trial_mask` was explicitly justified: since early_lick, outcome, and photostimulation are decoder variables, filtering them out would remove the signal the decoder needs to learn.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` - the spike times for each unit in the NWB file.

ii.
```python
spike_times_all = nwb.units['spike_times'][:]
spike_times_good = []
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

iii. The agent identified spike_times as the source for neural data, consistent with the reference code's use of spike_times from .mat files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins spanning [-2.5, +1.5]s relative to the go cue (80 bins). Firing rates are computed as spike counts divided by bin width (counts/0.05 = Hz). This uses `np.histogram` with pre-computed bin edges.

ii.
```python
def compute_firing_rates_session(spike_times_list, go_times, bin_width=BIN_WIDTH, ...):
    n_bins = int((window_end - window_start) / bin_width)
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, spikes in enumerate(spike_times_list):
            if len(spikes) > 0:
                left = np.searchsorted(spikes, bin_edges[0])
                right = np.searchsorted(spikes, bin_edges[-1])
                if left < right:
                    counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
                    fr[i, :] = counts / bin_width
```

iii. The agent noted that the reference code uses a Gaussian-smoothed sliding histogram (`sliding_histogram` with bw=0.1s, stride=0.05s), but the task instructions specify 50ms bins. The AI chose simple histogram binning (rectangular bins) rather than Gaussian smoothing, per the instructions' specification of "50-ms-width bins".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `classification` field in `nwb.units`. Only units with `classification == 'good'` are kept. This corresponds to the classifier-based QC described in the spike sorting QC paper.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)
if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close()
    return None
good_indices = np.where(good_mask)[0]
```

iii. The agent identified from the reference code that QC mode is 'classifier' (region-specific classifiers), and the NWB files have a `classification` field with 'good' or 'unlabelled' values. The agent does NOT apply the reference code's `check_fr` function which removes zero-variance neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. Bin edges are computed as `go_time + window_start + np.arange(n_bins + 1) * bin_width`, where go_time comes from `go_start_times`.

ii.
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
# ...
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. The agent confirmed from the reference code and methods that trials are aligned to go cue onset, consistent with the instructions specifying "Go cue onset" alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms (0.050s), producing 80 bins for the 4-second window [-2.5, +1.5]s. No rebinning is applied - spike times are directly binned at 50ms resolution.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins for firing rates
WINDOW_START = -2.5  # seconds before go cue
WINDOW_END = 1.5    # seconds after go cue
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
```

iii. The agent noted the reference code uses different binning (100ms kernel with 50ms stride, or 40ms kernel with 3.4ms stride for video analysis), but followed the task instructions specifying 50ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This is computed from a constant offset. The agent determined that tone onset occurs at -1.85s relative to the go cue. No per-trial tone onset variable is read from the data for this computation.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The agent determined from exploring the NWB data (sample_start_times) and the reference papers that tone onset is consistently at -1.85s relative to go cue. Since this is constant across trials, it's computed from the constant rather than read per-trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `BIN_CENTERS - TONE_ONSET_REL_GO`, where `BIN_CENTERS` are the center times of each 50ms bin relative to the go cue, and `TONE_ONSET_REL_GO = -1.85`. This yields a linearly increasing time series from -0.625s to 3.325s across the 80 bins.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The agent computed that at the start of the window (-2.5s from go cue), time from tone onset = -2.5 - (-1.85) = -0.65s, and at the end (1.5s from go cue), time from tone onset = 1.5 - (-1.85) = 3.35s. The values are continuous and time-varying, matching the instructions.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Since both use the same bin centers (BIN_CENTERS, referenced to go cue), the time-from-tone input is inherently aligned with the neural data at each time bin.

ii.
```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)  # shape: (2, 80)
```

iii. The agent used the same 80 time bins for both neural data and inputs, ensuring alignment by construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `photostim_start_times` and `photostim_stop_times` in `nwb.acquisition['BehavioralEvents']`.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The agent explored the NWB files and found that photostimulation events are stored as start/stop timestamp pairs in BehavioralEvents.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code finds photostim events that overlap with the trial window, then creates a binary time series at each bin center: 1.0 if the bin center falls within any photostim start/stop interval, 0.0 otherwise.

ii.
```python
def get_photostim_timeseries(photostim_start_times, photostim_stop_times, go_time, bin_centers=BIN_CENTERS):
    n_bins = len(bin_centers)
    photostim = np.zeros(n_bins, dtype=np.float32)
    abs_bin_centers = go_time + bin_centers
    for start, stop in zip(photostim_start_times, photostim_stop_times):
        mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
        photostim[mask] = 1.0
    return photostim
```

iii. The agent first filters photostim events to those overlapping the trial window, then marks bin centers that fall within any photostim interval.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation signal uses the same bin centers as the neural data (relative to go cue), ensuring temporal alignment.

ii.
```python
abs_bin_centers = go_time + bin_centers  # same bin_centers used for neural data
```

iii. Alignment is by construction since both neural and photostim use the same go-cue-referenced bin centers.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The "choice" output is derived from `nwb.trials['trial_instruction']`, which records the instructed lick direction ('left' or 'right'), NOT the actual lick direction chosen by the mouse.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
# ...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```
```python
CHOICE_MAP = {'left': 0, 'right': 1}
```

iii. The agent mapped trial_instruction to choice, noting left=0, right=1. The trajectory shows the agent treated `trial_instruction` as the choice variable. However, `trial_instruction` is the stimulus instruction (which tone was played), not the mouse's actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A simple dictionary mapping converts string values to integers: 'left' -> 0, 'right' -> 1. The per-trial value is broadcast to all 80 time bins.

ii.
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
# ...
np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
```

iii. The agent treated choice as a per-trial variable, broadcast across time bins, following the instructions' specification of "per-trial".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `nwb.trials['outcome']`, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = nwb.trials['outcome'][:]
# ...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. The agent directly mapped the NWB outcome field to the required encoding.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A dictionary mapping converts: 'ignore' -> 0, 'miss' -> 1, 'hit' -> 2. The per-trial value is broadcast to all time bins.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64),
```

iii. Follows the instructions' mapping exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `nwb.trials['early_lick']`, which contains 'early' or 'no early'.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
# ...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. The agent directly used the NWB early_lick field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A dictionary mapping converts: 'no early' -> 0, 'early' -> 1. Broadcast to all time bins.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64),
```

iii. Follows the instructions' mapping exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']`, specifically column index 1 (y-coordinate) of the data array, along with the corresponding timestamps.

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
# ...
local_y = tongue_data_all[idx_start:idx_end, 1]  # column 1 = y
```

iii. The agent explored the NWB file structure and identified tongue tracking data at ~300Hz with 3 columns (x, y, likelihood).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, tongue y values are extracted within the trial time window and averaged within each 50ms bin using `np.digitize` and per-bin averaging. Bins without data are set to NaN.

ii.
```python
def get_tongue_y_for_trial(tongue_ts, tongue_data, go_time, ...):
    tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
    # ...
    bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            tongue_y[b] = np.mean(local_y[mask])
    return tongue_y
```

iii. The agent uses all tongue tracking data without likelihood filtering (does not threshold by likelihood to exclude low-confidence detections).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles are computed from all valid (non-NaN) tongue y values across all trials in the session. Categories: 0 if < 40th percentile, 1 if between 40th and 60th, 2 if > 60th. NaN values default to category 1 (middle).

ii.
```python
def discretize_tongue_y(tongue_y_session, percentile_low=40, percentile_high=60):
    all_values = np.concatenate([y[~np.isnan(y)] for y in tongue_y_session if len(y[~np.isnan(y)]) > 0])
    p_low = np.percentile(all_values, percentile_low)
    p_high = np.percentile(all_values, percentile_high)
    # ...
    discrete[valid & (y < p_low)] = 0
    discrete[valid & (y >= p_low) & (y <= p_high)] = 1
    discrete[valid & (y > p_high)] = 2
```

iii. Follows the instructions' percentile thresholds (40th, 60th). NaN bins default to middle category (1).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue data uses the same go-cue-referenced bin edges as neural data, ensuring temporal alignment.

ii.
```python
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
```

iii. Same bin structure as neural data, aligned by construction.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Partial recording coverage**: Trials where the neural recording doesn't cover the full window are excluded via spike-time-range checking.
- **All-zero neural data**: Some trials (2,304 across 85 sessions, ~2.8%) have all-zero neural data. These are kept in the dataset.
- **Missing tongue data**: Bins without tongue tracking data are set to NaN, then discretized as category 1 (middle).
- **Sessions with 0 good units**: Skipped.
- **Missing photostim events**: Default to all-zero (no stimulation).

ii.
```python
# Recording coverage
rec_start, rec_end = get_recording_range(nwb, good_indices)
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)

# Tongue NaN handling
tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
# defaults to category 1 (middle) when discretized
discrete = np.ones(len(y), dtype=np.int64)  # default middle
```

iii. The agent discovered the partial recording issue through debugging (sessions where recording ends mid-session) and handled it by checking actual spike time ranges. The all-zero neural data warnings were investigated and attributed to brief recording gaps.

## 10-a. What are the most time-consuming steps of the code?

i. Firing rate computation is the most time-consuming step, taking 1.6-7.1s per session depending on neuron count and trial count. Data loading (reading NWB files) takes 1.0-1.5s per session. Total conversion time was ~19 minutes for 174 files.

ii.
```python
# Timing is printed:
# "Firing rates computed in {t_fr - t_load:.1f}s"
# "Data loaded in {t_load - t0:.1f}s"
```

iii. The agent estimated and tracked timing, finding firing rate computation dominates (up to 7s for sessions with many neurons).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop in `compute_firing_rates_session` iterates over neurons individually, calling `np.histogram` per neuron per trial. This nested loop (trials x neurons) could be partially vectorized. The tongue y binning loop (`for b in range(n_bins)`) could also be vectorized.

ii.
```python
for trial_idx in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        # per-neuron histogram
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)

for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

iii. The agent used `searchsorted` optimization to narrow the spike range before histogramming, but the per-neuron loop remains.

## 10-c. What processing does the code repeat multiple times?

i. The code reads `spike_times_all` twice from the NWB file - once in `get_recording_range()` and once in the main `process_session()` function. Each call reads the full spike times array.

ii.
```python
# In get_recording_range:
spike_times_all = nwb.units['spike_times'][:]

# In process_session (later):
spike_times_all = nwb.units['spike_times'][:]
```

iii. This redundant read could be avoided by passing the already-loaded spike times to `get_recording_range`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes a detailed brain region simplification mapping (`simplify_brain_region`) with ~70 entries, but many raw annotation names don't match any mapping entry and are passed through unchanged. The verification output shows many unsimplified region names coexisting with simplified ones (e.g., "Anterior olfactory nucleus" alongside "ACA"), resulting in a very large brain_regions list (~140 regions). This fragmented region mapping doesn't affect the decoder but represents unnecessary complexity. Additionally, `subject_desc` is extracted but not used in the final data structure.

ii.
```python
subject_desc = nwb.subject.description  # extracted but unused in final output

def simplify_brain_region(anno_name):
    # 70+ mappings, many not matched
    for key, abbrev in region_map.items():
        if key.lower() in name.lower():
            return abbrev
    return name  # fallback: return original name unchanged
```

iii. The agent created an extensive region mapping but many NWB annotations use different naming conventions than the mapping keys, so many regions pass through unsimplified.
