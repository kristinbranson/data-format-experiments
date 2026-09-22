# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by listing subject directories (`sub-*`) and globbing for `.nwb` files within each. Each file is opened with `pynwb.NWBHDF5IO` and processed individually in `process_session()`. Subjects, trials, units, and behavioral events are read from within each NWB file.

ii.
```python
def get_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
        for f in nwb_files:
            all_files.append((subj, f))
    return all_files
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    session_id = nwb.identifier
    trials = nwb.trials
    ...
```

iii. The AI's CONVERSION_NOTES.md documents that the dataset contains 28 subject directories with 174 NWB files. Standard `pynwb` is used as the reader.

## 1-b. How are the data split into subjects?

i. Each NWB file records the animal's `subject_id` via `nwb.subject.subject_id`. The AI collects all subject IDs across sessions and builds a sorted list of unique subjects. Subject index for each session maps into this sorted list.

ii.
```python
subject_id = nwb.subject.subject_id
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The AI uses the numeric `subject_id` field from the NWB file directly. This yields 28 unique subjects.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The AI processes each file independently and identifies sessions by `nwb.identifier`. No grouping or splitting is needed.

ii.
```python
session_id = nwb.identifier
...
all_session_ids.append(result['session_id'])
```

iii. The CONVERSION_NOTES document that 174 NWB files exist, of which 173 have good units and are retained.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). The AI reads trial columns directly (e.g., `trials['trial_instruction'][:]`, `trials['outcome'][:]`). Go cue times are from `BehavioralEvents/go_start_times`.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
early_lick = trials['early_lick'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI uses the trials table directly without additional derivation of trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in two ways: (1) excludes `auto_water` and `free_water` trials, and (2) excludes trials whose neural data window (`go_cue + ALIGN_START` to `go_cue + ALIGN_END`) falls outside the recording time range derived from `obs_intervals`. Sessions with fewer than 2 valid trials are dropped.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)
for i in range(n_trials):
    if valid_trial_mask[i]:
        window_start = go_times[i] + ALIGN_START
        window_end = go_times[i] + ALIGN_END
        if window_start < rec_min or window_end > rec_max:
            valid_trial_mask[i] = False
```

```python
def get_recording_time_range(units, good_indices):
    obs = np.array(units['obs_intervals'][good_indices[0]])
    min_time = obs[0, 0]
    max_time = obs[-1, 1]
    return min_time, max_time
```

iii. The CONVERSION_NOTES explain: "Trial curation for decoder: exclude auto_water and free_water; exclude trials outside recording range." The AI uses the overall recording time range rather than matching individual trial start times against `obs_intervals` entries.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` for units classified as `'good'`. Go cue times from `BehavioralEvents/go_start_times` are used to set the bin edges.

ii.
```python
classification = units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
...
good_spike_times = []
for idx in good_indices:
    st = units['spike_times'][idx]
    good_spike_times.append(np.sort(st))
```

iii. The AI reads spike times for each good unit individually and sorts them.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue using `np.histogram`. The spike counts are divided by bin width to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_cue_times, n_neurons, ...):
    n_bins = int((align_end - align_start) / bin_width)
    bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
    fr_all = []
    for t in range(n_trials):
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        bin_edges = go_cue_times[t] + bin_offsets
        for i, spikes in enumerate(spike_times_list):
            ...
            counts, _ = np.histogram(spikes_window, bins=bin_edges)
            fr[i] = counts / bin_width
        fr_all.append(fr)
    return fr_all
```

iii. The AI documents that spike rates are computed in 50ms bins as specified by the task requirements.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Sessions with zero good units are skipped. No additional quality metric thresholds are applied.

ii.
```python
classification = units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
if n_good == 0:
    print(f"  Session {session_id}: No good units, skipping")
    return None
```

iii. CONVERSION_NOTES: "Neuron: classification == 'good'" matching the QC classifier described in the spike sorting white paper. The AI reports 69,453 total good units across 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go cue times are on the same session-absolute clock. Bin edges are computed as offsets from each trial's go cue time. No additional alignment or interpolation is needed.

ii.
```python
bin_edges = go_cue_times[t] + bin_offsets
...
counts, _ = np.histogram(spikes_window, bins=bin_edges)
```

iii. The AI aligns to the go cue as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin width is 50 ms, producing 80 bins spanning -2.5 s to +1.5 s relative to the go cue. This matches the instructions. No rebinning is applied since spikes are binned directly from raw spike times.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins
ALIGN_START = -2.5
ALIGN_END = 1.5
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)  # 80 bins
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

iii. The AI follows the instruction specification of 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (the tone onsets) from `BehavioralEvents`, and the go cue time for each trial. The AI finds the last sample_start_time before the go cue (with a small 0.01s tolerance).

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
prev_samples = sample_start_times[sample_start_times < go_time + 0.01]
if len(prev_samples) > 0:
    tone_onset = prev_samples[-1]
else:
    tone_onset = go_time - 1.85
```

iii. The AI identifies that early licks can replay the sample epoch, so multiple sample_start_times may exist per trial, and selects the last one before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is computed as: `(go_time + BIN_CENTERS) - tone_onset`. This gives a continuous, time-varying input representing seconds since the tone.

ii.
```python
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. No additional processing beyond computing the offset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same `BIN_CENTERS` array which is defined as offsets from the go cue. The time_from_tone values are computed at these same bin centers, so they are inherently aligned with the neural data.

ii.
```python
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. Same bin grid is used for both neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents`, combined with `photostim_power` from the trials table to determine which trials have stimulation.

ii.
```python
has_photostim_events = 'photostim_start_times' in be.time_series
if has_photostim_events:
    photostim_event_starts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_event_stops = be.time_series['photostim_stop_times'].timestamps[:]
...
photostim_power_str = trials['photostim_power'][:]
```

iii. The AI uses the event-based photostim times rather than the per-trial `photostim_onset` and `photostim_duration` columns in the trials table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each stimulation trial, the AI iterates through all photostim events to find those overlapping with the trial window. A bin is marked as 1 if *any part* of it overlaps with a photostim event (bin overlap check), rather than checking if the bin center falls within the stimulation period.

ii.
```python
if has_photostim_events and photostim_power_str[trial_idx] != 'N/A':
    trial_window_start = go_time + ALIGN_START
    trial_window_end = go_time + ALIGN_END
    for ps_idx in range(len(photostim_event_starts)):
        ps_start = photostim_event_starts[ps_idx]
        ps_stop = photostim_event_stops[ps_idx]
        if ps_stop < trial_window_start or ps_start > trial_window_end:
            continue
        for b in range(N_TIMEBINS):
            bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
            bin_end_abs = bin_start_abs + BIN_WIDTH
            if bin_start_abs < ps_stop and bin_end_abs > ps_start:
                photostim_binary[b] = 1.0
```

iii. The AI uses a bin-overlap approach rather than a bin-center approach for determining if photostimulation is on.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim events are timestamped on the same session-absolute clock. The AI computes absolute bin edges from the go cue and checks for overlap with photostim event times directly.

ii. Same code as 4-b, using absolute time comparisons.

iii. Alignment is inherent since all times are on the same clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') in the trials table. There is no direct choice column.

ii.
```python
def determine_choice(trial_instruction, outcome):
    if outcome == 'ignore':
        return 2  # no lick
    elif outcome == 'hit':
        return 0 if trial_instruction == 'left' else 1
    elif outcome == 'miss':
        return 1 if trial_instruction == 'left' else 0
    return 2
```

iii. The AI correctly derives choice from the instruction and outcome: hit means licked the instructed side, miss means licked the opposite side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no_lick. It is a per-trial value that is expanded to all time bins by repeating.

ii.
```python
choice = determine_choice(trial_instruction[trial_idx], outcome[trial_idx])
...
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. The encoding matches the instructions: left, right, no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = trials['outcome'][:]
...
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome[trial_idx], 0)
```

iii. No derivation needed; the trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. Per-trial values are expanded across all time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome[trial_idx], 0)
...
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. Straightforward encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early_lick = trials['early_lick'][:]
...
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. Direct mapping from the trials table field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Coded as 0=no, 1=yes. Per-trial value expanded across all time bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
...
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. Simple binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has columns for tongue_x, tongue_y, and tongue_likelihood, with matching timestamps.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```

iii. This is the only tongue tracking data in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI: (1) bins tongue data into 50ms bins per trial, (2) within each bin, keeps only frames with likelihood >= 0.9, (3) computes mean tongue_y of visible frames per bin, (4) collects all visible bin-mean values across all trials in the session, (5) computes 40th and 60th percentiles of those values, (6) discretizes each bin: 0 if < p40, 1 if between p40 and p60 inclusive, 2 if > p60, 3 if not visible.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9

def get_tongue_y_for_trial(tongue_ts, tongue_data, go_cue_time, ...):
    ...
    for b in range(n_bins):
        mask = bin_idx == b
        frames = data_window[mask]
        visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
        visible_binned[b] = visible.mean()
        if visible.sum() > 0:
            tongue_y_binned[b] = frames[visible, 1].mean()
    return tongue_y_binned, visible_binned
```

```python
# Percentiles from all visible bin-mean values across trials
if len(tongue_y_all_visible) > 0:
    tongue_y_arr = np.array(tongue_y_all_visible)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)
```

```python
tongue_discrete[visible_mask] = np.where(ty < p40, 0,
                                         np.where(ty <= p60, 1, 2))
```

iii. CONVERSION_NOTES: "Tongue visibility: 0.9 DLC likelihood threshold." The AI computes percentiles from trial-based visible bin means rather than session-wide bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses `np.where` with boundaries at p40 and p60: values < p40 get class 0, values between p40 and p60 (inclusive) get class 1, values > p60 get class 2, and bins with no visible frames get class 3.

ii.
```python
tongue_discrete[visible_mask] = np.where(ty < p40, 0,
                                         np.where(ty <= p60, 1, 2))
```

iii. The 4-class scheme matches the instructions (below 40th, 40th-60th, above 60th, not visible).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are on the same session-absolute clock. For each trial, the AI finds frames within the trial window using `searchsorted`, assigns them to bins using `np.digitize` against the same bin edges used for neural data, and computes mean tongue_y per bin.

ii.
```python
bin_edges = go_cue_time + align_start + np.arange(n_bins + 1) * bin_width
...
left_idx = np.searchsorted(tongue_ts, bin_edges[0])
right_idx = np.searchsorted(tongue_ts, bin_edges[-1])
...
bin_idx = np.digitize(ts_window, bin_edges) - 1
```

iii. Same bin grid as neural data ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units are skipped (returns None). (2) Trials outside recording range or with auto_water/free_water are excluded. (3) Tongue bins with no visible frames (likelihood < 0.9) are assigned class 3 ('not visible').

ii.
```python
if n_good == 0:
    return None
...
valid_trial_mask = (auto_water == 0) & (free_water == 0)
...
if window_start < rec_min or window_end > rec_max:
    valid_trial_mask[i] = False
...
tongue_discrete = np.full(N_TIMEBINS, 3, dtype=np.int64)
```

iii. The AI handles the one session without QC labels by checking for zero good units. The AI does not appear to have a special `_text()` handler for non-string classification values — it relies on `units['classification'][:]` comparing directly to `'good'`, which may work if NaN != 'good' evaluates correctly.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports total processing takes ~3.5s per session (~10 min total). The most time-consuming steps are reading unit spike times (per-unit reads) and computing firing rates (nested trial x neuron loops using `np.histogram`).

ii.
```python
# Per-unit spike time reading
for idx in good_indices:
    st = units['spike_times'][idx]
    good_spike_times.append(np.sort(st))

# Nested loops for firing rates
for t in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes_window, bins=bin_edges)
```

iii. CONVERSION_NOTES: "Script: ~3.5s per session, ~10 min total"

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The firing rate computation has a doubly-nested loop (trials x neurons) that could be improved. The reference solution vectorizes across trials by flattening all edge arrays and using a single `searchsorted` per unit. The tongue processing also has a per-bin loop within each trial.

ii.
```python
# Outer loop over trials, inner loop over neurons
for t in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes_window, bins=bin_edges)
```

```python
# Per-bin loop in tongue processing
for b in range(n_bins):
    mask = bin_idx == b
```

iii. The AI named the function `compute_firing_rates_vectorized` but it still uses nested loops over trials and neurons.

## 10-c. What processing does the code repeat multiple times?

i. The photostimulation computation iterates through ALL photostim events for every stim trial, even though most events belong to other trials. This is redundant searching. The per-trial processing loop also reconstructs bin edges for each trial separately instead of computing them once.

ii.
```python
for ps_idx in range(len(photostim_event_starts)):
    ps_start = photostim_event_starts[ps_idx]
    ...
```

iii. No documentation of this inefficiency in CONVERSION_NOTES.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `correct_rate` per session for reporting purposes, reads `auto_water` and `photostim_power` columns that influence filtering but not the output format directly. The `visible_binned` array from tongue processing is used only for thresholding visibility but not stored in the output.

ii.
```python
control_mask_all = (photostim_power_str == 'N/A') & (early_lick == 'no early')
ctrl_out = outcome[control_mask_all]
n_hit = (ctrl_out == 'hit').sum()
n_miss = (ctrl_out == 'miss').sum()
correct_rate = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0
```

iii. The correct_rate computation is purely informational and not included in the output data structure.
