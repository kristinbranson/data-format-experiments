# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly open NWB files as HDF5, reading from paths like `intervals/trials/`, `units/`, and `acquisition/BehavioralEvents/`. All NWB files are found via `glob.glob('data/sub-*/sub-*.nwb')` and sorted. Each file is processed by `process_session()`.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
f = h5py.File(nwb_path, 'r')
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
```

iii. The agent explored the raw HDF5 path structure in preliminary Bash probes and confirmed the data was accessible without the PyNWB API. h5py was used from the first prototype onward without debating pynwb as an alternative.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from the NWB filename by parsing the `sub-<id>` prefix, rather than reading a field from within the NWB file. Unique subjects are collected and sorted.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The agent noted there were 174 NWB files across 28 unique subjects. Subject IDs are numeric strings derived from the filename (e.g., `440956`).

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The sorted glob of NWB files defines the session list. Sessions that fail filtering criteria are skipped.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
for nwb_path in nwb_files:
    result = process_session(nwb_path)
    if result is None:
        n_skipped += 1
        continue
```

iii. The agent recognized that each NWB file represents a single session. Session metadata (filename) is stored in `metadata['session_files']`.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The number of trials equals the length of `intervals/trials/id`. Go cue times are asserted to match the trial count.

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_cue_times) == n_trials
```

iii. The agent verified go cue count matches trial count in each session.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways: (1) `auto_water` and `free_water` trials are excluded, and (2) trials must fall within the recording observation window (`recording_valid`). The observation window is computed as the intersection of all selected units' `obs_intervals` (max start, min end), with a 0.1s tolerance. Additionally, sessions are filtered at the session level: performance must exceed 65% on control trials, and there must be at least 50 correct left and 50 correct right trials. Early lick and ignore trials are kept.

ii.
```python
# Session-level filtering
is_control = ((early_lick == 'no early') & (auto_water == 0) &
              (free_water == 0) & (ps_onset_str == 'N/A'))
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
if performance <= MIN_PERFORMANCE:
    return None
if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None

# Trial-level filtering
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
```

iii. The agent reasoned that the decoder task requires early_lick and outcome (including ignore) as outputs, so those trials should not be excluded. Session performance criteria (>65%, >=50 correct per direction) were taken directly from methods.txt. The `auto_water` exclusion was added based on checking data (14 auto_water trials per session). The `recording_valid` filter was discovered through debugging all-zero neural data in late trials of some sessions, which revealed that some NWB files contain behavioral trials beyond the recording period.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (with `units/spike_times_index` for ragged array indexing) and go cue times from `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
unit_spike_times = []
for ui in unit_indices:
    start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
    end_idx = spike_times_index[ui]
    unit_spike_times.append(spike_times_data[start_idx:end_idx])
```

iii. The agent identified spike_times as the neural representation in the NWB file and used the standard ragged array access pattern.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins using `np.histogram`, then divided by bin width to convert to firing rates in Hz. This is done per-unit, per-trial.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```

iii. The agent noted that while the reference code uses 100ms bandwidth with 50ms stride (Gaussian-weighted), the decoder task specifies 50ms bins, so the task specification took precedence.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Additionally, units must have a valid brain region mapping (via `map_anno_name_to_region`). Units with unmappable or empty `anno_name` are excluded. A session with no qualifying units is dropped.

ii.
```python
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
good_mask = classification == 'good'

unit_regions = []
unit_indices = []
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)
```

iii. The agent identified `classification` as the QC classifier verdict from the spike sorting quality control paper. It also added a region-validity filter, excluding ~490 units (<1%) with unmappable annotation names.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time. The bin edges are defined relative to the go cue (-2.5s to +1.5s). The function `compute_firing_rates` subtracts go_cue_time from spike times, then histograms relative times into BIN_EDGES.

ii.
```python
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
...
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    counts, _ = np.histogram(rel_times, bins=bin_edges)
```

iii. Alignment to go cue onset is specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins spanning -2.5s to +1.5s relative to the go cue, giving 80 time bins. Bin edges are computed via `np.linspace`. No rebinning is applied; spikes are directly histogrammed into these bins.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The 50ms bin width and -2.5s to +1.5s window are directly from the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onsets) in `acquisition/BehavioralEvents/sample_start_times/timestamps` and the go cue time. The last sample_start before the go cue is used as the tone onset.

ii.
```python
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time
    return None
```

iii. The agent verified that the tone onset to go cue gap is approximately 1.85s (sample 0.65s + delay 1.2s). Early lick trials can have replayed sample epochs, so the last sample_start before the go cue is the correct tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset relative to go cue is computed, then for each bin center, the time from tone onset is `BIN_CENTERS - tone_onset_rel`. Since `tone_onset_rel` is negative (tone is before go cue), this gives positive values that increase over time. A fallback of -1.85 is used if no sample_start is found before the go cue.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. The agent computed this as elapsed seconds since tone onset at each bin center.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same BIN_CENTERS array, which is defined relative to the go cue. The time_from_tone values are computed at bin centers, so they are inherently aligned with the neural data bins.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. Alignment is inherent since both use the same bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `intervals/trials/photostim_onset` (onset relative to trial start, stored as string), `intervals/trials/photostim_duration` (duration), `intervals/trials/start_time` (trial start time), and go cue times.

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. The agent initially misunderstood photostim_onset as an absolute time and had to debug the issue; it was found to be relative to trial start. The fix was `ps_abs_onset = trial_start_times[ti] + ps_onset_val`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Photostimulation is a binary time series per trial. For stimulated trials, the absolute onset is computed as `trial_start + photostim_onset`, converted to go-cue-relative, and bins where `BIN_CENTERS` falls between onset and offset are set to 1. Non-stimulated trials (onset == 'N/A') remain all zeros.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float64)
if ps_onset_str[ti] != 'N/A':
    ps_onset_val = float(ps_onset_str[ti])
    ps_dur = float(ps_dur_str[ti])
    ps_abs_onset = trial_start_times[ti] + ps_onset_val
    ps_start_rel = ps_abs_onset - gc
    ps_end_rel = ps_start_rel + ps_dur
    photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                       (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The agent confirmed this places photostim at approximately -1.2s to -0.7s relative to go cue (late delay epoch), consistent with the methods text.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, and BIN_CENTERS is also relative to the go cue, so they share the same time axis.

ii.
```python
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                   (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. Alignment is inherent since photostim timing is converted to be go-cue-relative, matching the neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from `trial_instruction` (`left` or `right`), which is the instructed side, NOT the animal's actual lick direction.

ii.
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
...
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. The agent used `trial_instruction` (the cue side) as choice. This does NOT account for the animal's actual response; on miss trials (where the animal licked the wrong side) or ignore trials (no lick), the "choice" reported is the instructed side, not the actual behavior.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 = left, 1 = right based on `trial_instruction`. There is no "no lick" category. The value is broadcast across all 80 time bins.

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
...
out = np.array([
    np.full(N_BINS, output_choice[i], dtype=np.int64),
    ...
])
```

iii. The agent listed `output_values` for choice as `['left', 'right']` with only two categories, omitting the "no lick" category specified in the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `intervals/trials/outcome`, which contains `'ignore'`, `'miss'`, or `'hit'`.

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
...
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:
    outcome = 2
```

iii. The three outcome categories map directly to the instructions' specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: 0 = ignore, 1 = miss, 2 = hit. The value is broadcast across all 80 bins.

ii.
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:
    outcome = 2
...
np.full(N_BINS, output_outcome[i], dtype=np.int64)
```

iii. Direct mapping of the three categories as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, which contains `'no early'` or `'early'`.

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
...
el = 1 if early_lick[ti] == 'early' else 0
```

iii. Direct read from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 = no, 1 = yes. Broadcast across all 80 bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
...
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. Simple binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `data` of shape `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` and matching `timestamps`. Column 1 (tongue_y) is used.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]
```

iii. The agent identified this as the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. All tongue tracking data is used without confidence filtering. Per-trial, tongue y values within the trial window are averaged into 50ms bins. Session-wide percentiles at p40 and p60 are computed over all valid (non-NaN) bin values across all trials. Each bin is then discretized: 0 = below p40, 1 = p40-p60, 2 = above p60. NaN bins (no tracking data) default to category 1 (middle).

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    # Uses all tongue tracking data (no confidence filtering)
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
    return tongue_y_binned

# Session-wide percentiles
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]
p40 = np.percentile(valid_tongue_y, 40)
p60 = np.percentile(valid_tongue_y, 60)

# Discretize
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

iii. The agent initially filtered by confidence threshold (0.9), but found this resulted in ~80% of bins being "middle" category. After investigation, the agent removed confidence filtering entirely, reasoning that all tracking data should be used and the decoder can learn to predict categories even if some represent the retracted baseline. Percentiles are computed over all valid values across all trials in the session (not over per-bin means).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three visible categories: 0 = below p40, 1 = p40-p60, 2 = above p60. There is NO explicit "not visible" (category 3) class. NaN bins default to category 1 (middle) because the default array is initialized to `np.ones(N_BINS, dtype=np.int64)`.

ii.
```python
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
valid_mask = ~np.isnan(ty)
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

iii. The agent's output_values for tongue_y_position lists only `['low', 'middle', 'high']` with three categories, not four. The "not visible" class (3) specified in the instructions is missing.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are searched for frames within the trial's time window (go_cue + T_START to go_cue + T_END), aligned relative to the go cue, then binned into the same 50ms bins as neural data.

ii.
```python
ts_rel = ts_window - go_cue_time
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
    if np.any(bin_mask):
        tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

iii. The same bin edges are used for both neural and tongue data, ensuring alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with no good units with valid regions are dropped. (2) Trials outside the recording observation window are excluded via `recording_valid`. (3) Tongue bins with no tracking data are set to NaN and default to middle category (1). The agent also handles NaN classification values by decoding bytes/strings, which would not match 'good' and thus be excluded.

ii.
```python
if len(unit_indices) == 0:
    return None
...
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
...
tongue_y_binned = np.full(n_bins, np.nan)
# NaN bins default to middle category
ty_disc = np.ones(N_BINS, dtype=np.int64)
```

iii. The agent discovered the obs_intervals issue through debugging and implemented recording_valid filtering. Missing tongue data defaults to category 1 rather than being flagged as a separate class.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Reading each NWB file with h5py and loading spike_times data. (2) Computing firing rates per-unit per-trial via `compute_firing_rates`, which loops over every unit and every trial individually with `np.histogram`. (3) Computing tongue y-position per-trial with per-bin loops.

ii.
```python
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The nested loop over trials and units is the dominant computational cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The double loop over trials and units in the firing rate computation is the main candidate. The reference solution vectorizes the trial dimension by flattening bin edges across all trials and using a single `searchsorted` per unit. The AI's code calls `np.histogram` once per unit per trial (n_units x n_trials calls). The tongue y per-bin loop within `extract_tongue_y_for_trial` could also be vectorized.

ii.
```python
# AI: double loop (trials x units)
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)

# Reference: single loop over units, trials vectorized
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
    rates[r] = np.diff(pos, axis=1)
```

iii. The AI's approach processes one trial at a time per unit, missing the optimization of batching all trials together.

## 10-c. What processing does the code repeat multiple times?

i. Within `compute_firing_rates`, each call to `np.histogram` independently processes the same spike times array for each trial of the same unit. The full spike time array is filtered with a mask each time, rather than pre-slicing once. Additionally, `sample_start_times` is filtered for each trial independently.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time  # recomputed for every trial
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])  # recomputed
```

iii. The subtraction and masking of spike times is repeated for every trial for the same unit.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The session performance calculation (including control trial identification, performance percentage, and correct-per-direction counts) is done for all sessions, including those that pass the filter. This work is not part of the output data. The `auto_water` trial exclusion may also be unnecessary since free_water captures the relevant cases. The large REGION_MAPPING dictionary processes many region names that may not appear in the data.

ii.
```python
is_control = ((early_lick == 'no early') & (auto_water == 0) &
              (free_water == 0) & (ps_onset_str == 'N/A'))
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
```

iii. The session filtering computation is inherently necessary for the filtering step, though it adds processing. The region mapping is done once at startup.
