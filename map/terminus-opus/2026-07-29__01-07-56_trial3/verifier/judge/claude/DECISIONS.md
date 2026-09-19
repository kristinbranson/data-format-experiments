# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files using `pynwb`. It globs for all `.nwb` files under `data/sub-*/` and iterates over each file, opening it with `pynwb.NWBHDF5IO`. From each file it extracts trials, units, behavioral events, and behavioral time series.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```

```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
```

iii. The AI noted that the dataset is in NWB format (DANDI:000363) with 174 NWB files across 28 subjects, and that pynwb is the standard reader. This matches the reference approach.

## 1-b. How are the data split into subjects?

i. The AI reads `nwb.subject.subject_id` from each NWB file and collects unique subject IDs. The sorted set of unique IDs forms the `subjects` list, and each session is mapped to its index via `subject_to_idx`.

ii.
```python
subject_id = nwb.subject.subject_id
...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
```

iii. This is the standard approach and matches the reference. The subject ID is the numeric DANDI identifier (e.g. '440956').

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session, so no splitting is needed. Sessions are processed in sorted file order. Files are identified by their path.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
```

iii. This matches the reference approach. One NWB file = one session.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). The AI reads trial columns (`trial_instruction`, `outcome`, `early_lick`, etc.) and go cue times from `BehavioralEvents/go_start_times`.

ii.
```python
n_trials_total = len(nwb.trials)
trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. This matches the reference approach of using the NWB trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies THREE types of filtering:
1. **Session-level behavioral filtering**: Sessions with <65% correct rate or <50 correct left/right trials are dropped entirely (22 sessions removed, leaving 151).
2. **Recording coverage filtering**: Trials where the go cue window falls outside the range of actual spike times are excluded.
3. **No unit-count minimum beyond 2**: Sessions with <2 good units are skipped.

The AI does NOT filter by `obs_intervals` or `free_water`. Instead, it uses the actual spike time range to determine recording coverage.

ii.
```python
rec_start, rec_end = get_recording_range(nwb, good_indices)
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
...
if correct_rate < 0.65:
    print(f'  Skipping: correct rate {correct_rate:.3f} < 0.65')
    io.close()
    return None

if correct_left < 50 or correct_right < 50:
    print(f'  Skipping: correct_left={correct_left}, correct_right={correct_right} (need >=50)')
    io.close()
    return None
```

iii. The AI documented session selection criteria from the paper (>65% correct, >=50 correct L/R) in CONVERSION_NOTES.md Step 3. The reference solution does NOT apply session-level behavioral filtering — it keeps all 173 sessions with good units. The recording coverage filter uses spike time range rather than obs_intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` for units classified as `'good'`. Go cue times from `BehavioralEvents/go_start_times` provide the alignment event.

ii.
```python
spike_times_all = nwb.units['spike_times'][:]
spike_times_good = []
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

iii. This matches the reference — both use spike_times from good-classified units.

## 2-b. How is the `neural` data processed?

i. For each trial, bin edges are computed from the go cue time. For each neuron, spikes within the trial window are histogrammed into 50ms bins using `np.histogram`, then divided by bin width to get firing rates in Hz.

ii.
```python
def compute_firing_rates_session(spike_times_list, go_times, ...):
    for trial_idx in range(n_trials):
        go_time = go_times[trial_idx]
        bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
        for i, spikes in enumerate(spike_times_list):
            counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
            fr[i, :] = counts / bin_width
```

iii. The processing is functionally equivalent to the reference (both bin spikes into 50ms windows and convert to Hz). However, the AI loops over both trials AND neurons, whereas the reference vectorizes across trials using a flattened edge array.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with fewer than 2 good units are skipped.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)
if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close()
    return None
```

iii. The reference uses the same `classification == 'good'` filter but with a threshold of 0 (no good units → skip), not 2. The AI's threshold of 2 is slightly more restrictive but unlikely to matter in practice as the dropped session has 0 good units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. Bin edges are computed as offsets from the go cue time for each trial: `go_time + WINDOW_START + np.arange(n_bins + 1) * bin_width`.

ii.
```python
go_time = go_times[trial_idx]
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. This matches the reference — both align to go cue onset with the same window [-2.5, 1.5]s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50ms, giving 80 bins over the 4s window. No rebinning is applied — spike times are binned directly into 50ms bins.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins for firing rates
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
```

iii. This matches both the instructions and the reference.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI uses a FIXED constant `TONE_ONSET_REL_GO = -1.85` as the tone onset relative to the go cue, rather than reading actual tone onset times from the data.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md that trial timing is "sample -1.85 to -1.20" based on the paper. The reference solution reads actual `sample_start_times` per trial, finding the last tone before the go cue, which accounts for early-lick replays where the sample epoch is repeated.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI subtracts the fixed tone onset offset from the bin centers: `BIN_CENTERS - TONE_ONSET_REL_GO`. Since both are constants, this produces the same time_from_tone for EVERY trial.

ii.
```python
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. This is incorrect — time_from_tone should vary per trial based on the actual tone onset time. Early-lick trials have the sample epoch replayed, shifting the effective tone onset. The reference computes a per-trial value using actual sample_start_times.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. The input uses the same `BIN_CENTERS` as the neural data, so they share the same time grid. Since tone onset is a fixed offset, alignment is trivially consistent.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The alignment method is correct in principle (shared bin centers), but the fixed tone offset means the values are wrong for early-lick trials.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents` time series, which are session-level event streams containing all photostimulation events.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The reference uses `photostim_onset` and `photostim_duration` from the trials table instead, which are per-trial values. Both approaches should yield equivalent results, though the trial-table approach is more direct.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI finds photostim events overlapping the trial window, then creates a binary time series where bins with centers falling between start and stop times are set to 1.

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

iii. Functionally similar to the reference. One minor difference: the AI uses `<=` for the stop time boundary while the reference uses `<`. This could include one extra bin at the boundary.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostimulation uses the same bin centers (relative to go cue) as the neural data, ensuring temporal alignment.

ii.
```python
abs_bin_centers = go_time + bin_centers
```

iii. This matches the reference approach of using go-cue-relative bin centers.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI maps `trial_instruction` directly to choice values: 'left' → 0, 'right' → 1. It does NOT derive the actual lick direction from instruction and outcome.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. This is INCORRECT. The instructions ask for "Lick direction choice", which is the animal's actual lick direction, not the instructed direction. The reference derives choice from instruction × outcome: hit → lick matches instruction, miss → lick is opposite, ignore → no lick. The AI's approach also lacks a "no lick" category entirely.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI simply looks up the trial_instruction in a dictionary. There is no derivation or combination with outcome. The choice is stored as a per-trial value broadcast across all 80 time bins. `output_values[0]` is `['left', 'right']` — only 2 categories, missing 'no lick'.

ii.
```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
...
out_arr = np.stack([
    np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
    ...
])
```

iii. The reference has 3 categories: left(0), right(1), no lick(2). The AI only has 2. This means ignore trials (no lick) are incorrectly assigned the instructed direction as their "choice".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds strings 'ignore', 'miss', 'hit'.

ii.
```python
outcome = nwb.trials['outcome'][:]
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. This matches the reference — both read outcome directly from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: ignore→0, miss→1, hit→2. Per-trial values are broadcast across all 80 time bins.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
...
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64),
```

iii. This matches the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. This matches the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to integers: 'no early'→0, 'early'→1. Per-trial values are broadcast across all time bins.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
```

iii. This matches the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has columns (x, y, likelihood) with timestamps. The AI uses column 1 (tongue_y).

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```

iii. This matches the reference — same data source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, tongue y values are averaged within 50ms bins using `get_tongue_y_for_trial`. Then `discretize_tongue_y` computes session-wide 40th and 60th percentiles of ALL non-NaN tongue y values and assigns categories 0 (below 40th), 1 (40th-60th), 2 (above 60th). NaN bins (no frames) default to category 1 (middle).

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
    ...
    discrete = np.ones(len(y), dtype=np.int64)  # default middle
    discrete[valid & (y < p_low)] = 0
    discrete[valid & (y >= p_low) & (y <= p_high)] = 1
    discrete[valid & (y > p_high)] = 2
```

iii. Key differences from reference:
1. **No likelihood filtering**: The AI does not filter frames by tracking likelihood before computing tongue y. Low-likelihood frames (tongue not visible) are included in bin averages and percentile computation.
2. **No "not visible" category**: The AI assigns NaN bins to category 1 (middle) by default, rather than a separate "not visible" category (3) as specified in the instructions.
3. **Percentiles over trial bin means only**: The AI computes percentiles from the per-trial bin-averaged values, not from session-wide bin means as the reference does.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses the 40th and 60th percentiles of all non-NaN tongue y bin means across the session to create 3 categories. Bins with no data default to 1 (middle).

ii.
```python
discrete[valid & (y < p_low)] = 0
discrete[valid & (y >= p_low) & (y <= p_high)] = 1
discrete[valid & (y > p_high)] = 2
```

iii. The instructions specify 4 categories: 0 (<40th), 1 (40th-60th), 2 (>60th), 3 (not visible). The AI only implements 3, mapping "not visible" bins to category 1 instead of a separate category 3.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue y is extracted for each trial using the same window (go_time + BIN_CENTERS ± BIN_WIDTH/2) as the neural data, with tongue timestamps digitized into the same 80 bins.

ii.
```python
def get_tongue_y_for_trial(tongue_ts, tongue_data, go_time, ...):
    t_start = go_time + bin_centers[0] - bin_width / 2
    t_end = go_time + bin_centers[-1] + bin_width / 2
    abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
    bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
```

iii. The alignment approach is consistent with the neural data — both use the same go-cue-relative time grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases:
- Sessions with <2 good units are skipped
- Sessions failing behavioral criteria (correct rate, trial counts) are skipped
- Trials outside recording range are excluded
- Tongue tracking bins with no frames get NaN (then defaulted to category 1)

The AI does NOT handle:
- Sessions where `classification` is NaN (the reference handles this by converting non-string entries to '')
- `free_water` trials (the reference explicitly excludes these)
- Low-likelihood tongue frames (the reference discards these)

ii.
```python
if n_good < 2:
    return None
...
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
```

iii. The AI's approach is less thorough than the reference. The reference explicitly handles NaN classification, free_water trials, and low-likelihood tongue frames. The AI's recording-range filter is a rougher proxy for obs_intervals.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, the most time-consuming step is computing firing rates. Some sessions take 7+ seconds for this step. Total conversion took ~1144s (19 min) for 174 files.

ii.
```python
# From conversion_full_out.txt:
# Firing rates computed in 7.1s (for session with 526 units, 520 trials)
```

iii. The double loop (over trials AND neurons) in `compute_firing_rates_session` makes the neural binning the dominant cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's `compute_firing_rates_session` has a double loop: outer loop over trials, inner loop over neurons. The reference vectorizes the trial dimension by flattening all edge arrays and only looping over neurons. The `get_tongue_y_for_trial` also loops per-bin and is called per-trial.

ii.
```python
for trial_idx in range(n_trials):
    for i, spikes in enumerate(spike_times_list):
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
```

iii. The trial loop is the most impactful — the reference processes all trials at once per neuron using a single searchsorted on the flattened edge array.

## 10-c. What processing does the code repeat multiple times?

i. The spike times are read once per session via `nwb.units['spike_times'][:]`, but then a second time in `get_recording_range` which also reads spike times for all good units. This is redundant I/O.

ii.
```python
def get_recording_range(nwb, good_indices):
    spike_times_all = nwb.units['spike_times'][:]
    ...

def process_session(nwb_file, ...):
    ...
    rec_start, rec_end = get_recording_range(nwb, good_indices)
    ...
    spike_times_all = nwb.units['spike_times'][:]  # read again
```

iii. The data is read from NWB twice: once in `get_recording_range` and once in the main processing. This could be consolidated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session-level behavioral statistics (`correct_rate`, `correct_left`, `correct_right`) for session filtering. These statistics are stored in metadata but the session filtering itself is unnecessary per the reference — the reference includes all 173 sessions with good units. This filtering discards 22 sessions that could have been used.

ii.
```python
denom = np.sum(control_no_early & is_not_ignore)
hits = np.sum(control_no_early & (outcome == 'hit'))
correct_rate = hits / denom
correct_left = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'left'))
correct_right = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'right'))
```

iii. The behavioral session filtering removes valid sessions from the decoder training data. The reference paper mentions these criteria in the context of their analysis, but the decoder task instructions do not require session-level behavioral filtering.
