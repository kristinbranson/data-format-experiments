# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files located in `data/sub-*/sub-*.nwb`. It iterates through all 174 NWB files using `glob.glob`, processing each one via `process_session()`. Each NWB file corresponds to one session. The AI uses the `pynwb` library to open and read each file.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```

```python
def process_session(nwb_file, show_processing=False, session_idx=0):
    io = pynwb.NWBHDF5IO(nwb_file, 'r')
    nwb = io.read()
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code loads from `.mat` files exported from DataJoint, but the NWB files contain the same data. The AI found 174 NWB files across 28 subjects, consistent with the dataset description.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from each NWB file's `nwb.subject.subject_id`. Unique subjects are collected across all sessions and stored as a sorted list. A `subject_idx` array maps each session to its subject index.

ii.
```python
subject_id = nwb.subject.subject_id
# ...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
subject_idx_list = [subject_to_idx[s['subject_id']] for s in all_sessions]
```

iii. The AI documented finding 28 subjects in the data, matching the paper's count.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. After processing, sessions that fail quality criteria are excluded (see 1-e). The remaining sessions are collected into lists indexed by session order.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# Each file = one session
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
    if session_data is not None:
        all_sessions.append(session_data)
```

iii. The AI notes that 151 out of 173 sessions with good units pass the behavioral criteria, with 22 sessions filtered out.

## 1-d. How are the data split into trials?

i. Trials are identified via `nwb.trials` and the corresponding `go_start_times` from `BehavioralEvents`. Each trial has a go cue time that is used for alignment. Trials are filtered by neural recording coverage (whether the full window [-2.5s, +1.5s] around the go cue falls within the recording range).

ii.
```python
n_trials_total = len(nwb.trials)
go_times = beh_events.time_series['go_start_times'].timestamps[:]
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
```

iii. The AI justified filtering by recording coverage to avoid all-zero neural data at the edges of the recording.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three levels of filtering:
- **Session-level**: Sessions must have >65% correct rate on control no-early-lick trials, and at least 50 correct left and 50 correct right trials, and at least 2 good units.
- **Trial-level**: Trials are filtered only by neural recording coverage (whether the full time window is within the recording range). Unlike the reference code's `get_regular_trial_mask`, trials with early licks, auto water, free water, no response, and stimulation are NOT excluded -- because they are decoder outputs/inputs.

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
if correct_rate < 0.65:
    # skip session
if correct_left < 50 or correct_right < 50:
    # skip session
```

iii. The AI explained that it keeps all trials because early_lick, outcome, and photostimulation are decoder outputs/inputs, so excluding them would remove the variation needed for decoding. Session-level filtering matches the paper's criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spike_times` of units classified as `'good'` in the `classification` column of `nwb.units`.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
spike_times_all = nwb.units['spike_times'][:]
spike_times_good = []
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

iii. The AI documented that the QC mode is 'classifier' (region-specific classifiers), matching the reference code's approach.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.histogram` to compute firing rates (spikes/second). The bins span from -2.5s to +1.5s relative to go cue onset, producing 80 time bins per trial.

ii.
```python
def compute_firing_rates_session(spike_times_list, go_times, bin_width=BIN_WIDTH,
                                  window_start=WINDOW_START, window_end=WINDOW_END):
    n_bins = int((window_end - window_start) / bin_width)
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, spikes in enumerate(spike_times_list):
            if len(spikes) > 0:
                counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
                fr[i, :] = counts / bin_width
```

iii. The AI noted that the instructions specify 50ms bins. The reference code uses `sliding_histogram` with `bw=0.1s, stride=0.05s` (100ms-wide bins sliding by 50ms) by default, or `bw=0.04, stride=0.0034` for the Sherlock scripts. The AI chose non-overlapping 50ms bins as specified in the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons based on the `classification` column in the NWB units table, keeping only units labeled `'good'`. No additional QC filtering (e.g., zero-variance check via `check_fr`) is applied.

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

iii. The AI documented that QC uses the 'classifier' mode matching the reference code. The reference `check_fr` function removes zero-variance neurons, but the AI does not apply this additional filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. The `go_start_times` timestamps from `BehavioralEvents` serve as the alignment reference (time 0). For each trial, bin edges are constructed relative to the go cue time.

ii.
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
# ...
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. The instructions specify "Temporally align based on Go cue onset", and the reference code also aligns to go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms (0.050s), resulting in 80 bins spanning [-2.5s, +1.5s]. No temporal rebinning is applied -- spike times are directly binned into 50ms bins.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins for firing rates
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
```

iii. The instructions specify "50-ms-width bins". The reference code default uses `bw=0.1, stride=0.05` (100ms bins, 50ms stride), while the Sherlock scripts use `bw=0.04, stride=0.0034`. The AI chose 50ms non-overlapping bins per the instructions rather than matching the reference code's bin width.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The tone onset time is computed as a fixed offset from the go cue: -1.85 seconds. This is hardcoded as `TONE_ONSET_REL_GO = -1.85`. No raw variable is read for per-trial tone onset.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The AI derived the -1.85s offset from methods.txt: the sample epoch consists of three 150ms tones with 100ms inter-tone intervals (total 650ms), followed by a 1.2s delay. Total = 1.85s before go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The input is computed as `BIN_CENTERS - TONE_ONSET_REL_GO`, where `BIN_CENTERS` are the centers of each 50ms bin relative to the go cue, and `TONE_ONSET_REL_GO = -1.85`. This produces a linearly increasing time series in seconds, identical for all trials.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The result is a continuous, time-varying signal that equals 0 at tone onset and increases linearly.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Since both the neural data bins and the time-from-tone input use the same `BIN_CENTERS` array (defined relative to go cue), they are inherently aligned. Each time bin's "time from tone onset" value corresponds to the center of the same neural data bin.

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. No separate alignment step is needed since both use the same temporal grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. These are the actual laser on/off times recorded during the experiment.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the code identifies photostim events that overlap with the trial window. It then creates a binary time series (0/1) at each bin center, setting bins to 1 when the bin center falls within a photostim start-stop interval.

ii.
```python
def get_photostim_timeseries(photostim_start_times, photostim_stop_times,
                              go_time, bin_centers=BIN_CENTERS):
    photostim = np.zeros(n_bins, dtype=np.float32)
    abs_bin_centers = go_time + bin_centers
    for start, stop in zip(photostim_start_times, photostim_stop_times):
        mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
        photostim[mask] = 1.0
    return photostim
```

```python
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
if np.any(stim_mask):
    trial_stim_starts = photostim_starts_all[stim_mask]
    trial_stim_stops = photostim_stops_all[stim_mask]
    photostim_ts = get_photostim_timeseries(trial_stim_starts, trial_stim_stops, go_time)
else:
    photostim_ts = np.zeros(N_TIMEBINS, dtype=np.float32)
```

iii. The AI noted that photostim events are not per-trial in the NWB file but are global timestamps, so they must be matched to trials by overlap with trial windows.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary signal uses the same `BIN_CENTERS` as neural data, converted to absolute times by adding the go cue time. This ensures alignment with neural data bins.

ii.
```python
abs_bin_centers = go_time + bin_centers
```

iii. Same temporal grid as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Lick direction choice is derived from the `trial_instruction` column of `nwb.trials`.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
# ...
CHOICE_MAP = {'left': 0, 'right': 1}
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. The AI maps 'left' to 0 and 'right' to 1 as specified in the instructions.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The `trial_instruction` string is mapped to an integer (left=0, right=1) and broadcast as a constant across all 80 time bins for that trial.

ii.
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
output_choice.append(choice)
# ...
out_arr = np.stack([
    np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
    ...
], axis=0)
```

iii. This is the trial instruction (left/right lick port), used as the "choice" output. Note: the AI uses `trial_instruction` rather than actual lick behavior; for "hit" trials these are the same, but for "miss" trials the actual lick may differ.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `outcome` column of `nwb.trials`.

ii.
```python
outcome = nwb.trials['outcome'][:]
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. Maps the three outcome categories to integers matching the instruction specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome string is mapped to an integer (ignore=0, miss=1, hit=2) and broadcast as a constant across all time bins.

ii.
```python
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
output_outcome.append(out)
# ...
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64),
```

iii. Straightforward categorical mapping.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The instructions ask about "Outcome", not "Distance to reward zone". There is no distance-to-reward-zone variable in this dataset. Outcome is a per-trial scalar broadcast to all time bins, so alignment is trivial -- the same value occupies all time bins.

ii.
```python
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64)
```

iii. N/A -- this question appears to reference a different dataset. The Outcome output is per-trial and constant across time.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the `early_lick` column of `nwb.trials`.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. Binary categorization matching the instruction specification.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The early_lick string is mapped to an integer (no early=0, early=1) and broadcast across all time bins.

ii.
```python
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
output_early_lick.append(el)
# ...
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64),
```

iii. Straightforward binary mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `Camera0_side_TongueTracking` in `BehavioralTimeSeries`. Specifically, column index 1 (y-coordinate) of the tracking data.

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
# ...
local_y = tongue_data[idx_start:idx_end, 1]  # column 1 = y
```

iii. The tongue tracking data has 3 columns (x, y, likelihood), recorded at ~300Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, tongue y-position samples are binned into the same 50ms bins as neural data. Within each bin, tongue y values are averaged. Bins with no tongue data are set to NaN.

ii.
```python
def get_tongue_y_for_trial(tongue_ts, tongue_data, go_time, bin_centers=BIN_CENTERS, bin_width=BIN_WIDTH):
    tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
    # ...
    abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
    bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            tongue_y[b] = np.mean(local_y[mask])
    return tongue_y
```

iii. Average within each time bin, using NaN for missing data.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, all valid (non-NaN) tongue y values across all trials are pooled. The 40th and 60th percentiles are computed. Values below 40th percentile become 0, between 40th-60th become 1, above 60th become 2. NaN time bins default to category 1 (middle).

ii.
```python
def discretize_tongue_y(tongue_y_session, percentile_low=40, percentile_high=60):
    all_values = []
    for y in tongue_y_session:
        valid = y[~np.isnan(y)]
        if len(valid) > 0:
            all_values.append(valid)
    all_values = np.concatenate(all_values)
    p_low = np.percentile(all_values, percentile_low)
    p_high = np.percentile(all_values, percentile_high)

    tongue_y_discrete = []
    for y in tongue_y_session:
        discrete = np.ones(len(y), dtype=np.int64)  # default middle
        valid = ~np.isnan(y)
        if np.any(valid):
            discrete[valid & (y < p_low)] = 0
            discrete[valid & (y >= p_low) & (y <= p_high)] = 1
            discrete[valid & (y > p_high)] = 2
        tongue_y_discrete.append(discrete)
    return tongue_y_discrete
```

iii. The instructions specify: 0 = <40th percentile, 1 = 40th-60th, 2 = >60th, with per-session discretization.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are aligned to the same time bins as neural data using the go cue time as reference. The bin edges match exactly.

ii.
```python
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
```

iii. Same temporal grid ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Sessions with <2 good units are skipped
- Sessions with <2 valid trials are skipped
- Trials outside recording coverage are excluded
- Missing tongue data in bins produces NaN, which defaults to category 1 (middle) after discretization
- Unknown `trial_instruction` or `outcome` values default to 0 via `.get(str(...), 0)`
- 2,304 trials (2.8%) across 85 sessions have all-zero neural data due to recording gaps; these are kept

ii.
```python
if n_good < 2:
    return None
if len(valid_trial_indices) < 2:
    return None
tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
discrete = np.ones(len(y), dtype=np.int64)  # default middle for NaN
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. The AI documented that 2,304 zero-data trials are not significantly correlated with specific trial characteristics and have minimal impact on training (2.8% of data).

## 10-a. What are the most time-consuming steps of the code?

i. According to timing information in the output, firing rate computation is the most time-consuming step, taking up to 7.1 seconds per session. Data loading takes 1.0-1.5 seconds per session. The full conversion takes approximately 19 minutes for 174 files.

ii.
```python
t_load = time.time()
print(f'  Data loaded in {t_load - t0:.1f}s')
fr_all = compute_firing_rates_session(spike_times_good, valid_go_times)
t_fr = time.time()
print(f'  Firing rates computed in {t_fr - t_load:.1f}s')
```

iii. The AI documented timing estimates in CONVERSION_NOTES.md Step 7, showing firing rate computation as the dominant cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- The inner loop over neurons in `compute_firing_rates_session` iterates per-neuron to compute histograms
- The `get_tongue_y_for_trial` function loops over bins to compute mean tongue y
- The `get_photostim_timeseries` function loops over start/stop pairs

ii.
```python
# Per-neuron loop in firing rate computation
for i, spikes in enumerate(spike_times_list):
    if len(spikes) > 0:
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
        fr[i, :] = counts / bin_width

# Per-bin loop in tongue y
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

iii. The per-neuron histogram loop is the biggest efficiency concern, as `np.histogram` is called once per neuron per trial (n_neurons * n_trials times total). A vectorized bincount approach or bulk histogram could speed this up.

## 10-c. What processing does the code repeat multiple times?

i. The `BIN_CENTERS` and `time_from_tone` arrays are computed once globally and reused, which is efficient. However, `spike_times_all = nwb.units['spike_times'][:]` is read twice in `process_session` -- once for `get_recording_range` and once for extracting good unit spike times.

ii.
```python
# First read in get_recording_range:
def get_recording_range(nwb, good_indices):
    spike_times_all = nwb.units['spike_times'][:]

# Second read in process_session:
spike_times_all = nwb.units['spike_times'][:]
```

iii. Reading all spike times from the NWB file twice is redundant and wasteful, especially since this is a large data read from HDF5.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `simplify_brain_region` function performs extensive string matching with a 60+ entry dictionary for every neuron, mapping detailed CCF annotations to abbreviations. While this is used in the output, the mapping is more elaborate than needed. The `subject_desc` field is extracted but only stored in metadata and not used for decoder training. The `correct_rate` is computed for session filtering and stored in metadata but not used downstream.

ii.
```python
subject_desc = nwb.subject.description  # extracted but not critical
# ...
region_map = {
    'Secondary motor area': 'MOs',
    'Primary motor area': 'MOp',
    # ... 60+ entries
}
```

iii. The brain region simplification is cosmetic -- the downstream decoder uses the region indices regardless of naming. The metadata fields add minor overhead.
