# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from NWB files in the `data/` directory. It lists subject directories (`sub-*`), then finds `.nwb` files within each. Each NWB file is opened with `pynwb.NWBHDF5IO` and processed via `process_session()`. Units, trials, behavioral events, and tongue tracking are read from within each file.

ii.
```python
def get_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files
```

```python
io = NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The AI notes in CONVERSION_NOTES.md that there are 174 NWB files across 28 subjects. It uses pynwb as the standard NWB reader.

## 1-b. How are the data split into subjects?

i. The AI uses the directory name (e.g., `sub-440956`) as the subject identifier, extracted from the file listing. Subject IDs are tracked as sessions are processed and assigned indices in order of first appearance.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    ...
    result = process_session(nwb_path, subject_id)
    ...
    if subject_id not in subjects_seen:
        subjects_seen[subject_id] = len(subjects_seen)
        subjects_list.append(subject_id)
```

iii. The AI tracks subjects by directory name (e.g., `sub-440956`) rather than the NWB field `nwb.subject.subject_id`. These are equivalent since the directory is named after the subject ID.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Files are sorted within each subject directory. The session name is the NWB filename basename.

ii.
```python
files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
'session_name': os.path.basename(nwb_path),
```

iii. The AI correctly treats each NWB file as a session. The conversion notes state 174 NWB files, 173 with good units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). Go cue times come from `BehavioralEvents/go_start_times`. Each trial is indexed by position in the trials table.

ii.
```python
trials = nwb.trials
outcomes = trials['outcome'][:]
instructions = trials['trial_instruction'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
...
go_start_times = events.time_series['go_start_times'].timestamps[:]
```

iii. The AI reads trial data directly from NWB trials table columns.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials on three criteria: (1) `auto_water == 0`, (2) `free_water == 0`, and (3) trials must be within obs_intervals (valid recording). A session is dropped if fewer than 2 trials survive.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]

if len(trial_indices) < 2:
    io.close()
    return None
```

iii. The AI's CONVERSION_NOTES.md states: "EXCLUDE: auto_water, free_water (confound behavior, not decoder variables)." The reference code only excludes `free_water` and trials outside `obs_intervals`, but does NOT exclude `auto_water`. The AI additionally excludes `auto_water` trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, read per-unit for each good unit. Go cue times from `BehavioralEvents/go_start_times` define the alignment.

ii.
```python
spike_times_per_unit = []
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. The AI uses spike times as the only neural variable, which is correct.

## 2-b. How is the `neural` data processed?

i. For each trial and each neuron, spikes within the trial window are assigned to 50ms bins and counted. Counts are divided by bin size to get firing rates in Hz. This is done per-trial in a function `compute_firing_rates()`.

ii.
```python
def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_neurons = len(spike_times_list)
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end
    for i, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        if len(spikes_in_window) == 0:
            continue
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(fr[i], bin_indices, 1.0)
    fr /= bin_size
    return fr
```

iii. The AI processes each trial independently in a loop, calling `compute_firing_rates` once per trial. The reference uses a vectorized approach processing all trials at once per neuron with `searchsorted`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps units with `classification == 'good'`, then additionally drops units whose `anno_name` does not map to one of the 14 predefined coarse brain regions.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
...
anno_names = nwb.units['anno_name'][:][good_mask]
region_indices = []
valid_neuron_mask = np.ones(n_good, dtype=bool)
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
        region_indices.append(-1)
    else:
        region_indices.append(COARSE_REGIONS.index(region))
...
good_unit_indices = good_unit_indices[valid_neuron_mask]
```

iii. The AI notes in CONVERSION_NOTES.md: "classification=='good' only (all have anno_name)." The additional filtering based on region mapping is implicit — neurons with unmappable annotations are silently dropped. The reference keeps ALL good units regardless of brain region annotation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue. The trial window is defined as `go_cue_time + T_START` to `go_cue_time + T_END` (-2.5s to +1.5s relative to go cue).

ii.
```python
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
...
bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
```

iii. Alignment to go cue is correct per the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s = 80 time bins. No rebinning beyond direct spike counting into these bins.

ii.
```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5     # seconds before go cue
T_END = 1.5        # seconds after go cue
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. Matches the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onsets) from `BehavioralEvents`, and the go cue time of each trial. The last sample onset before the go cue is used.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
...
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
```

```python
def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]
```

iii. Correct approach — accounts for early lick replays by taking the last tone before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Bin centers are computed relative to the go cue, then the tone onset is subtracted to get time from tone. If no tone is found, a fallback of `go_cue - 1.85` is used.

ii.
```python
def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time
    time_from_tone = bin_centers - tone_rel
    return time_from_tone.astype(np.float32)
```

```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
```

iii. The fallback of 1.85s before go cue is based on the typical task timing (sample starts ~1.85s before go cue). The reference code does not have this fallback — it assumes `sample_start_times` always has at least one entry before the go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and this input use bin centers defined as `t_start + bin_size/2 + k*bin_size` relative to the go cue, so they share the same time axis.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```

iii. Correctly aligned via shared bin center definition.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents` — these are absolute timestamps of photostimulation events across the entire session.

ii.
```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The reference solution uses `photostim_onset` and `photostim_duration` from the trials table instead. The AI uses the event-based timestamps, which is a different but potentially valid approach.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI finds photostim events that overlap the trial window, then creates a binary time series where bins with centers falling between start and stop times are set to 1.

ii.
```python
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
ps_starts = photostim_start_abs[stim_mask]
ps_stops = photostim_stop_abs[stim_mask]
photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                              T_START, T_END, BIN_SIZE_S)
```

```python
def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops, ...):
    ...
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim
```

iii. The AI uses `<=` for the stop comparison (inclusive), while the reference uses `<` (exclusive). This is a minor difference. The AI uses absolute event times rather than per-trial onset+duration from the trials table.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Bin centers are computed in absolute time (`bin_centers + go_cue_time`) and compared against absolute photostim start/stop times. The bin grid is the same as for neural data.

ii.
```python
abs_centers = bin_centers + go_cue_time
```

iii. Aligned via the same bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from `trial_instruction` — left instruction = 0, right instruction = 1. It does NOT account for the animal's actual lick direction or whether the animal licked at all.

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The reference derives choice from `trial_instruction × outcome`: hit means the animal licked the instructed side, miss means it licked the opposite side, and ignore (no lick) gets a third code (2). The AI's approach maps instruction to choice, which is incorrect — "choice" should reflect what the animal actually did, not what it was told to do. For miss trials, the AI assigns the wrong direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps instruction directly: `'left' → 0`, `'right' → 1`. There are only 2 output values: `['left', 'right']`. It is repeated across all 80 time bins.

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
...
full_output[0, :] = out_dict['choice']
```

```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The AI is missing a third class for "no lick" (ignore trials), and on miss trials assigns the instructed direction rather than the opposite direction. The reference has 3 classes: `['left', 'right', 'no lick']`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which holds `'hit'`, `'miss'`, or `'ignore'`.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome_str = outcomes[ti]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

iii. Directly from the NWB trials table, same as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values mapped to integers: ignore=0, miss=1, hit=2. Repeated across all 80 time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
...
full_output[1, :] = out_dict['outcome']
```

iii. Same encoding as the reference. The `.get(..., 0)` fallback would map unknown outcomes to ignore (0), but this never occurs in practice.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, with values `'no early'` and `'early'`.

ii.
```python
early_lick = trials['early_lick'][:]
...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early' → 0`, anything else → 1. Repeated across all time bins.

ii.
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
...
full_output[2, :] = out_dict['early_lick']
```

iii. Same as reference, just slightly different coding style.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, columns: (x, y, likelihood). Column 1 (y) is the value, column 2 (likelihood) gates visibility.

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. Same data source as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood < 0.9 are set to NaN. For each trial, tongue y values are binned into 50ms bins and averaged. The session-wide valid tongue y values are collected for percentile computation.

ii.
```python
y_in_window[like_in_window < 0.9] = np.nan
```

```python
# Bin tongue y-position
sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
counts = np.bincount(valid_bins, minlength=n_bins)
has_data = counts > 0
tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
```

iii. The AI uses a likelihood threshold of 0.9, while the reference uses 0.5. Since the likelihood values are effectively binary (~89% below 0.01, ~10.5% at or above 0.99), the difference in thresholds has only a small practical effect, but it's a different decision. The AI computes percentiles from raw valid time points rather than from 50ms bin means over the whole session, which is a meaningful difference.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of all valid (non-NaN) tongue y values are computed. Values below p40 get class 0, between p40 and p60 get class 1, above p60 get class 2. There is NO "not visible" class — NaN bins default to class 0.

ii.
```python
if len(tongue_y_session) > 0:
    tongue_y_arr = np.array(tongue_y_session)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)

tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

```python
'output_values': [
    ...
    ['below_40th', '40th_to_60th', 'above_60th'],
]
```

iii. Key differences from reference: (1) Percentiles are computed from individual valid time points, not from 50ms bin means of the whole session. (2) No "not visible" class (class 3) — bins with no visible tongue default to 0 (same as "below 40th percentile"). (3) The reference uses `np.digitize` which naturally produces 0/1/2 boundary handling, while the AI uses manual comparisons with `<=` for p60 boundary. (4) The reference has 4 output values; the AI only has 3.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are in absolute time (same clock as spikes and go cues). The trial window is defined by `go_cue + T_START` to `go_cue + T_END`, and tongue data within this window is binned into the same 50ms bins as neural data.

ii.
```python
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
...
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. Same alignment principle as neural data and other inputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units are skipped (returns None). (2) Trials outside obs_intervals or with auto_water/free_water are excluded. (3) Tongue frames with low likelihood are set to NaN; bins with no valid tongue frames default to class 0 (rather than a separate "not visible" class).

ii.
```python
if n_good == 0:
    io.close()
    return None
```

```python
if n_neurons == 0:
    io.close()
    return None
```

```python
y_in_window[like_in_window < 0.9] = np.nan
...
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
```

iii. The AI handles missing data by exclusion or NaN masking. Notably, the AI does NOT have a separate "not visible" class for tongue data, meaning bins without visible tongue are indistinguishable from "below 40th percentile" tongue positions.

## 10-a. What are the most time-consuming steps of the code?

i. Per the conversion output, full conversion took ~2300s (~38 minutes) for 174 sessions. The per-trial spike binning loop is the main bottleneck since `compute_firing_rates` is called once per trial with a loop over neurons inside it.

ii.
```python
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The AI notes processing time was ~6.5s per session, with ~38 min total. The reference code takes ~247s (~4 min) for the same data.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `compute_firing_rates` function loops over neurons and processes one trial at a time. The reference vectorizes across all trials at once per neuron using `searchsorted` on a flattened edge array. The per-trial loop calling `compute_firing_rates` is the main inefficiency.

ii.
```python
for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, ...)

# Inside compute_firing_rates:
for i, st in enumerate(spike_times_list):
    mask = (st >= abs_start) & (st < abs_end)
    ...
    np.add.at(fr[i], bin_indices, 1.0)
```

iii. The double loop (over trials, then over neurons) is the main performance issue. The reference processes all trials simultaneously per neuron.

## 10-c. What processing does the code repeat multiple times?

i. The spike times for each unit are re-filtered and re-binned for every trial (`compute_firing_rates` is called once per trial, and each call iterates over all neurons). The reference reads the spike time buffer once per neuron and uses a single `searchsorted` call across all trials.

ii.
```python
for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The per-trial spike filtering `(st >= abs_start) & (st < abs_end)` scans the full spike array for each neuron on each trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The coarse brain region mapping (`map_anno_to_region`) with its 14 hardcoded categories is complex and custom-built. While all computed fields go into the output, the region mapping drops neurons that don't match any of the 14 categories, which is unnecessary and loses data.

ii.
```python
def map_anno_to_region(anno):
    # ~100 lines of string matching rules
    ...
    return None  # If nothing matched
```

iii. The elaborate region mapping is not required — the reference simply uses the first part of the fine CCF annotation as the region label. The AI's approach is overly complex and loses neurons whose annotations don't match any rule.
