# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `/app/data/sub-*/` directories using `os.listdir` to iterate over subject directories and then over `.nwb` files within each. Each file is opened with `pynwb.NWBHDF5IO` and processed once. This is functionally equivalent to the reference's glob-based approach.

ii.
```python
def get_nwb_files(data_dir='/app/data'):
    nwb_files = []
    for subdir in sorted(os.listdir(data_dir)):
        sub_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(sub_path) or not subdir.startswith('sub-'):
            continue
        for f in sorted(os.listdir(sub_path)):
            if f.endswith('.nwb'):
                nwb_files.append(os.path.join(sub_path, f))
    return nwb_files
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
```

iii. The AI noted in CONVERSION_NOTES.md that NWB is the published format and uses pynwb as its standard reader. The approach finds all 174 NWB files across 28 subject directories.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject identifier from the file path directory name (e.g., `sub-440956`), rather than from the NWB file's internal `nwb.subject.subject_id` field. This means subject IDs include the `sub-` prefix.

ii.
```python
subject_id = nwb_path.split('/')[-2]  # e.g., "sub-440956"
```

```python
all_subjects = sorted(set(s['subject_id'] for s in all_sessions_data))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The CONVERSION_NOTES.md documents 28 subjects. The AI's approach uses the directory name rather than the NWB internal subject ID, yielding identifiers like `sub-440956` instead of `440956`.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session, so no further splitting is needed. Session IDs are derived from the filename (without `.nwb` extension), e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`.

ii.
```python
session_id = os.path.basename(nwb_path).replace('.nwb', '')
```

iii. The AI correctly identified that each NWB file is one session. 173 sessions are retained (one dropped for having 0 good units).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.intervals['trials']`). Go cue times come from `BehavioralEvents/go_start_times`. The AI does not explicitly assert that the number of go cues matches the number of trials.

ii.
```python
trials = nwb.intervals['trials']
n_trials_total = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI uses the trials table directly. Unlike the reference, there is no assertion checking that `len(go_times) == len(trials)`.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using three criteria: (1) `auto_water == 0`, (2) `free_water == 0`, and (3) the trial's analysis window `[go + t_start, go + t_end]` must fall within the recording time range (intersection of all good units' obs_intervals). Sessions with fewer than 2 valid trials are dropped.

ii.
```python
valid_trials = (auto_water == 0) & (free_water == 0)
recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
valid_trials = valid_trials & recording_covered
```

```python
def get_recording_time_range(nwb, good_indices):
    rec_start = -np.inf
    rec_end = np.inf
    for idx in good_indices:
        obs = nwb.units['obs_intervals'][idx]
        unit_start = obs[0, 0]
        unit_end = obs[-1, 1]
        rec_start = max(rec_start, unit_start)
        rec_end = min(rec_end, unit_end)
    return rec_start, rec_end
```

iii. The AI noted in CONVERSION_NOTES.md that auto_water and free_water trials are "not genuine behavioral trials" and should be excluded, and that trials outside the recording window should be excluded to avoid zero neural data. This differs from the reference which uses `obs_intervals` matching against trial start times and only filters `free_water`. The AI's approach yields 89,532 trials vs. the reference's 90,860.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times of good units (`units['spike_times']`), accessed per-unit via indexing. Go cue times (`BehavioralEvents/go_start_times`) define the alignment.

ii.
```python
spike_times_list = []
for idx in good_indices:
    spike_times_list.append(units['spike_times'][idx])
```

iii. The AI identified spike_times as the source for neural data, same as the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning [-2.5, 1.5] s relative to go cue (80 bins). For each neuron and trial, `np.searchsorted` finds spikes in the window, then `np.histogram` counts spikes per bin. Counts are divided by bin width to get firing rates in Hz.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_times, t_start, t_end, bin_width):
    n_bins = int(round((t_end - t_start) / bin_width))
    bin_edges = t_start + np.arange(n_bins + 1) * bin_width
    ...
    for j, spk_times in enumerate(spike_times_list):
        for i in range(n_trials):
            go = go_times[i]
            lo = np.searchsorted(spk_times, go + t_start)
            hi = np.searchsorted(spk_times, go + t_end)
            if hi > lo:
                relative_spikes = spk_times[lo:hi] - go
                counts, _ = np.histogram(relative_spikes, bins=bin_edges)
                fr_all[i, j, :] = counts / bin_width
```

iii. The approach is functionally equivalent to the reference's vectorized searchsorted method, but uses a double loop (over neurons and trials) with `np.histogram` instead of the reference's single loop over neurons with flattened edges and `np.diff(searchsorted)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with zero good units are skipped.

ii.
```python
classifications = units['classification'][:]
good_mask = classifications == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
if n_good == 0:
    return None
```

iii. The AI correctly uses the `classification` column from the QC classifier. However, unlike the reference which uses a `_text()` helper to handle NaN values in the classification column (converting non-strings to ''), the AI directly compares `classifications == 'good'`. This could potentially cause issues with NaN entries, though in practice the one affected session (all NaN classifications) appears to be handled because NaN != 'good' evaluates to True in numpy, yielding zero good units and triggering the skip.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go cue times are on the same session-absolute clock. For each trial, the go cue time is used to define the analysis window, and spike times are converted to go-cue-relative times before binning.

ii.
```python
go = go_times[i]
lo = np.searchsorted(spk_times, go + t_start)
hi = np.searchsorted(spk_times, go + t_end)
if hi > lo:
    relative_spikes = spk_times[lo:hi] - go
    counts, _ = np.histogram(relative_spikes, bins=bin_edges)
```

iii. The alignment is correct - spikes are extracted in the window [go-2.5, go+1.5] and binned relative to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5, 1.5] s relative to go cue. No rebinning is applied - spikes are directly binned at this resolution.

ii.
```python
t_start = -2.5
t_end = 1.5
bin_width = 0.05
n_bins = int(round((t_end - t_start) / bin_width))  # 80
bin_edges = t_start + np.arange(n_bins + 1) * bin_width
```

iii. Matches the task specification of 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (the tone onset events) and the go cue times. For each trial, the AI finds sample events within `[trial_start, go)` and takes the last one. A fallback of `go - 1.85` is used if no sample event is found.

ii.
```python
def get_tone_onset_per_trial(go_times, trial_starts, sample_start_times):
    tone_onsets = np.full(len(go_times), np.nan)
    for i in range(len(go_times)):
        go = go_times[i]
        ts = trial_starts[i]
        samp_in_trial = sample_start_times[(sample_start_times >= ts) & (sample_start_times < go)]
        if len(samp_in_trial) > 0:
            tone_onsets[i] = samp_in_trial[-1]
        else:
            tone_onsets[i] = go - 1.85
    return tone_onsets
```

iii. The AI correctly identified that early licks can replay the sample epoch, so the last tone before the go cue is the relevant one. The fallback to `go - 1.85` is a safety net. The reference uses `searchsorted` across all sample events (not restricted to within-trial), which is simpler and doesn't need a fallback.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The bin centers (relative to go cue) are offset by the tone-to-go gap: `bin_center - tone_onset_rel`, where `tone_onset_rel = tone_onset - go_time`.

ii.
```python
tone_onset_rel = tone_onsets[i] - go_times_valid[i]  # negative value
inputs[0, :] = bin_centers - tone_onset_rel  # = bin_centers + (go - tone)
```

iii. This is algebraically equivalent to the reference's `CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural binning are used for the time-from-tone computation, ensuring alignment.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
...
inputs[0, :] = bin_centers - tone_onset_rel
```

iii. Both neural and input use the same bin centers relative to go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` used to convert to absolute times and go cue used for alignment.

ii.
```python
photostim_onset_str = trials['photostim_onset'][:]
photostim_duration_str = trials['photostim_duration'][:]
...
if photostim_onset_valid[i] != 'N/A':
    stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
    stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
    stim_dur = float(photostim_duration_valid[i])
    stim_end_go_rel = stim_onset_go_rel + stim_dur
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1 where the bin center falls within [onset, onset+duration) relative to go cue, 0 otherwise. Non-stimulated trials ('N/A') remain all zeros.

ii.
```python
inputs[1, :] = ((bin_centers >= stim_onset_go_rel) & (bin_centers < stim_end_go_rel)).astype(np.float32)
```

iii. Same logic as the reference.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulus onset/offset are expressed relative to the go cue, and bin centers are also relative to go cue, so alignment is inherent.

ii. Same bin_centers used for neural binning and photostim comparison.

iii. Correct alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the actual lick event times: `left_lick_times` and `right_lick_times` from BehavioralEvents, taking the first lick after the go cue within the trial window. This differs from the reference, which derives choice from `trial_instruction` x `outcome`.

ii.
```python
def determine_lick_choice(go_time, trial_stop, left_lick_times, right_lick_times):
    left_after = left_lick_times[(left_lick_times > go_time) & (left_lick_times < trial_stop)]
    right_after = right_lick_times[(right_lick_times > go_time) & (right_lick_times < trial_stop)]
    first_left = left_after[0] if len(left_after) > 0 else np.inf
    first_right = right_after[0] if len(right_after) > 0 else np.inf
    if first_left < first_right:
        return 0  # left
    elif first_right < first_left:
        return 1  # right
    else:
        return 2  # no lick
```

iii. The AI noted in CONVERSION_NOTES.md (Step 5) "Determine from first lick after go cue (left_lick_times vs right_lick_times)." The reference derives choice from instruction x outcome logic instead. Both approaches should yield the same result in most cases since a hit means licking the instructed side and a miss means licking the other side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The choice is coded as 0=left, 1=right, 2=no_lick. It is a per-trial value repeated across all 80 time bins.

ii.
```python
choice = determine_lick_choice(go_times_valid[i], trial_stops_valid[i],
                                left_lick_times, right_lick_times)
outputs[0, :] = choice
```

iii. The encoding matches the reference (left=0, right=1, no_lick=2).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which holds 'hit', 'miss', or 'ignore'.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}
outcome = outcome_map.get(outcomes_valid[i], 2)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers: hit=0, miss=1, ignore=2. This is a **different ordering** from the reference which uses ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}
outcome = outcome_map.get(outcomes_valid[i], 2)
outputs[1, :] = outcome
```

The AI's output_values:
```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ['hit', 'miss', 'ignore'],
    ...
]
```

iii. The AI chose a different mapping order than the reference (which uses ignore=0, miss=1, hit=2). However, the output_values labels correctly describe each code, so the decoder should still work correctly - it's just a different arbitrary labeling convention.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' or 'no early'.

ii.
```python
early_lick = trials['early_lick'][:]
...
early = 1 if early_lick_valid[i] == 'early' else 0
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no early, 1=early. Repeated across all 80 time bins.

ii.
```python
early = 1 if early_lick_valid[i] == 'early' else 0
outputs[2, :] = early
```

iii. Same logic as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data: tongue_x, tongue_y, tongue_likelihood, with corresponding timestamps.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a likelihood threshold of **0.9** (vs. reference's 0.5). Percentiles (40th, 60th) are computed on **raw visible frames** across the whole session (vs. reference's 50ms-binned means). Per trial, frames are assigned to bins using `searchsorted`, and within each bin, visible frames' y-values are averaged and compared against the percentile thresholds.

ii.
```python
# Session-wide percentiles on raw frames
visible_mask = tongue_data[:, 2] >= 0.9
if np.any(visible_mask):
    y_visible = tongue_data[visible_mask, 1]
    p40 = np.percentile(y_visible, 40)
    p60 = np.percentile(y_visible, 60)
```

```python
# Per-bin classification
visible = trial_lk_v[mask] >= likelihood_threshold
if np.any(visible):
    mean_y = np.mean(trial_y_v[mask][visible])
    if mean_y < p40:
        result[i, b] = 0
    elif mean_y <= p60:
        result[i, b] = 1
    else:
        result[i, b] = 2
```

iii. Two key differences from the reference: (1) likelihood threshold 0.9 vs. 0.5, and (2) percentiles computed on raw frames vs. 50ms bin means. The 0.9 threshold is mentioned in CONVERSION_NOTES.md Step 5 as a design choice. The raw-frame percentile computation differs from the reference's binned-mean percentile approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Categories: 0 = below p40, 1 = p40 to p60 (inclusive of p60 boundary), 2 = above p60, 3 = not visible. The AI uses `<=` for the p60 boundary: `mean_y <= p60` maps to class 1, while the reference uses `np.digitize` which uses `<` boundaries.

ii.
```python
if mean_y < p40:
    result[i, b] = 0
elif mean_y <= p60:
    result[i, b] = 1
else:
    result[i, b] = 2
```

iii. The AI uses `<= p60` for class 1, while the reference uses `np.digitize(m[ok], edges)` where edges = [p40, p60], giving class 0 for < p40, class 1 for p40 <= x < p60, class 2 for >= p60. The boundary handling differs slightly.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock. For each trial, frames within the go-cue-aligned window are found via searchsorted, assigned to bins, and averaged.

ii.
```python
abs_edges = go + bin_edges
idx_start = np.searchsorted(tongue_timestamps, abs_edges[0])
idx_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
...
bin_indices = np.searchsorted(abs_edges, trial_ts, side='right') - 1
```

iii. Uses the same time window and bin structure as neural data, ensuring alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases handled:
- Sessions with no good units: skipped (returns None)
- Trials outside recording range: filtered via recording coverage check
- Tongue frames with low likelihood: excluded from bin averages; bins with no visible frames get class 3

ii.
```python
if n_good == 0:
    return None
```
```python
recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
```
```python
visible = trial_lk_v[mask] >= likelihood_threshold
```

iii. The AI also has a fallback for missing tone onsets (`go - 1.85`). The CONVERSION_NOTES.md documents that 2 edge-case trials still have all-zero neural data after filtering. Unlike the reference, the AI does not have special handling for NaN values in the classification column (no `_text()` helper), though this doesn't cause issues in practice.

## 10-a. What are the most time-consuming steps of the code?

i. The AI reports total processing time of ~590s for 174 sessions (~3.4s/session average). The firing rate computation is the bottleneck (1-3s per session), followed by data loading. The double loop over neurons and trials in `compute_firing_rates_vectorized` is less efficient than the reference's single-loop approach.

ii. From CONVERSION_NOTES.md: "Load data 0.3-0.7s, Firing rates 1.8-3.1s, Tongue y 0.2-0.3s"

iii. The AI identified the firing rate computation as the main bottleneck and optimized it from the initial version using searchsorted.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The double loop over neurons and trials in `compute_firing_rates_vectorized` could be partially vectorized. The reference vectorizes the trial dimension by flattening all trial edges into one array per unit, requiring only a single loop over units. The per-trial tongue y computation loop could also potentially be vectorized.

ii.
```python
for j, spk_times in enumerate(spike_times_list):
    for i in range(n_trials):  # This inner loop could be eliminated
        ...
```

iii. The AI's "vectorized" function name is somewhat misleading - it still has a nested loop over neurons x trials rather than the reference's single loop over neurons with vectorized trial handling.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and processed once. No processing appears to be redundantly repeated. The per-unit spike time reads (`units['spike_times'][idx]`) are done individually rather than via a bulk ragged-array read as in the reference, which may involve more I/O overhead.

ii.
```python
for idx in good_indices:
    spike_times_list.append(units['spike_times'][idx])
```

iii. The individual unit reads are less efficient than the reference's bulk read of the ragged spike_times buffer.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads `auto_water` which is used for filtering but the reference does not filter on this. The AI also reads `left_lick_times` and `right_lick_times` to determine lick choice, while the reference derives choice from trial_instruction x outcome without needing lick times. The AI reads `trial_stops` for the lick choice determination window.

ii.
```python
auto_water = trials['auto_water'][:]
left_lick_times = be.time_series['left_lick_times'].timestamps[:]
right_lick_times = be.time_series['right_lick_times'].timestamps[:]
```

iii. These extra data loads add I/O overhead but the data is used (just for a different approach than the reference).
